// kk_renderer.c — eigener, schmaler Render-Pfad (Strangler Phase 1).
//
// Ziel: pl_render_image für den Kuckuck-Prod-Input (AVPlayer-Hybrid) Schritt
// für Schritt ablösen, um den Pass-Graph selbst zu besitzen (Pacing-Hebel:
// Pass-Fusion 10–15 -> ~3–4; später AOT-Shader für Binary-Größe). Die
// Farbmathematik bleibt Stock: pl_shader_decode_color / _sample_polar /
// _color_map_ex / _dither werden weiter aus libplacebo aufgerufen — nur der
// Graph drumherum ist hier neu.
//
// STRANGLER-PRINZIP: kk_render_image() behandelt NUR Fälle, die kk_can_handle()
// freigibt, und liefert sonst false -> der Host (hybrid_render.c) fällt auf das
// bewährte pl_render_image() zurück. Damit kann Prod nie regressieren, und der
// migrierte Anteil wächst kontrolliert, jede Stufe einzeln on-device A/B-geprüft
// (IQ-Harness + Gegentest gegen den Stock-Pfad).
//
// Input-Contract (verifiziert aus hybrid_render.c, 2026-06-23):
//   - immer 2 Planes, biplanar 4:2:0 (Luma r8/r16 + Chroma rg8/rg16)
//   - Chroma-Location immer PL_CHROMA_LEFT
//   - Matrix 709/601/240M/2020-NC, Transfer BT.1886/PQ/HLG
//   - kein Frame-Mix, kein Deinterlace (macht mpv-bwdif upstream)
//   - Target: Swapchain RGB, sRGB (SDR) oder hdr10/nits (HDR)

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <libplacebo/dispatch.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>
#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/dithering.h>
#include <libplacebo/shaders/custom.h>

#include "kk_renderer.h"

struct kk_renderer {
    pl_gpu gpu;
    pl_dispatch dp;

    // Eigene, über die public API gehaltene Shader-State-Objekte (in
    // pl_renderer sind das rr->sampler_main / rr->tone_map_state / etc.).
    pl_shader_obj sampler_lut;   // Chroma-Upsample-LUT
    pl_shader_obj main_lut;      // Haupt-Upscaler-LUT (polar)
    pl_shader_obj downscaler_lut; // Downscaler-LUT (ortho, beide Achsen teilen sie)
    pl_shader_obj tone_map_state;
    pl_shader_obj dither_state;

    // Zwischentexturen (bei Größenwechsel via pl_tex_recreate neu):
    //   chroma_tex: Chroma bilinear auf Luma-Auflösung hochgesampelt
    //   rgb_tex:    zusammengesetztes + decode_color-tes RGB (Source-Res)
    pl_tex chroma_tex;
    pl_tex rgb_tex;
    pl_tex scale_tmp;            // Y-Pass-Zwischentextur fürs Downscale-Ping-Pong

    pl_tex hook_fbos[16];        // Scratch-Pool für get_tex (Anime4K ~10 saved tex)
    int    hook_fbo_used;        // pro Frame zurückgesetzt
};

struct kk_renderer *kk_renderer_create(pl_log log, pl_gpu gpu)
{
    if (!gpu)
        return NULL;
    struct kk_renderer *kr = calloc(1, sizeof(*kr));
    if (!kr)
        return NULL;
    kr->gpu = gpu;
    kr->dp  = pl_dispatch_create(log, gpu); // eigener Dispatch, wir besitzen ihn
    if (!kr->dp) {
        free(kr);
        return NULL;
    }
    return kr;
}

void kk_renderer_destroy(struct kk_renderer **pkr)
{
    struct kk_renderer *kr = *pkr;
    if (!kr)
        return;
    pl_shader_obj_destroy(&kr->sampler_lut);
    pl_shader_obj_destroy(&kr->main_lut);
    pl_shader_obj_destroy(&kr->downscaler_lut);
    pl_shader_obj_destroy(&kr->tone_map_state);
    pl_shader_obj_destroy(&kr->dither_state);
    pl_tex_destroy(kr->gpu, &kr->chroma_tex);
    pl_tex_destroy(kr->gpu, &kr->rgb_tex);
    pl_tex_destroy(kr->gpu, &kr->scale_tmp);
    for (int i = 0; i < (int)(sizeof(kr->hook_fbos)/sizeof(kr->hook_fbos[0])); i++)
        pl_tex_destroy(kr->gpu, &kr->hook_fbos[i]);
    pl_dispatch_destroy(&kr->dp);
    free(kr);
    *pkr = NULL;
}

// Welche Frames der kk-Pfad heute übernimmt. KONSERVATIV halten und nur
// erweitern, nachdem die jeweilige Stufe on-device gegen pl_render_image
// verifiziert wurde. Alles, was hier false ergibt, läuft über den Stock-Pfad.
//
// Erste Slice (diese Datei): SDR (BT.1886), kein User-Hook, kein Deband,
// und 1:1 (KEIN Scaling) — so ist das Output-Sampling ein exakter Copy und der
// erste IQ-Harness-Diff gegen Stock sauber matchbar. Scaling kommt als eigene
// Slice mit dem echten Down-/Upscaler nach (sonst würde Bilinear≠Stock-Scaler
// den Diff verfälschen).
static bool kk_can_handle(const struct pl_frame *image,
                          const struct pl_frame *target,
                          const struct pl_render_params *params)
{
    if (image->num_planes != 2)                 return false; // nur biplanar 4:2:0
    // SDR (BT.1886/sRGB) und HDR (PQ/HLG) — Tonemap+Peak via color_map/detect_peak.
    // Hooks: RGB/MAIN (CAS, Anime4K-Restore) + PRE_KERNEL (Anime4K). LUMA-Hooks
    // (FSRCNNX/ArtCNN, Plane-Stage + Resize) -> Stock.
    for (int n = 0; n < params->num_hooks; n++)
        if (params->hooks[n]->stages & ~(unsigned)(PL_HOOK_RGB | PL_HOOK_PRE_KERNEL))
            return false;
    if (params->deband_params)                  return false;
    if (params->lut || target->lut)             return false;
    if (params->distort_params)                 return false;
    if (params->cone_params)                    return false;
    // Chroma 2x + Haupt-Upscale brauchen einen polaren Scaler (sample_polar).
    // Nicht-polare Upscaler (z.B. HD-Light-Lanczos) bräuchten Upscale-Ortho.
    if (!params->upscaler || !params->upscaler->polar) return false;
    // Downscale: separabel via ortho-Ping-Pong -> downscaler muss existieren.
    if (!params->downscaler) return false;
    return true;
}

// rgba16f-FBO (re)anlegen. storable=true erlaubt Compute (für den polaren
// Chroma-Sampler); storable=false zwingt Raster (Merge-Pass braucht gültiges
// gl_FragCoord -> kein Compute-Upgrade).
static bool kk_get_fbo(struct kk_renderer *kr, pl_tex *tex, int w, int h, bool storable)
{
    pl_fmt fmt = pl_find_named_fmt(kr->gpu, "rgba16f");
    if (!fmt)
        return false;
    return pl_tex_recreate(kr->gpu, tex, &(struct pl_tex_params) {
        .w = w, .h = h, .format = fmt,
        .renderable = true, .sampleable = true, .storable = storable,
    });
}

// Luma + Chroma -> RGB in kr->rgb_tex. Spiegelt pass_read_image für genau den
// 2-Plane-4:2:0-Fall:
//   (1) Chroma (rg) bilinear auf Luma-Auflösung in kr->chroma_tex. Die
//       Sample-Rect übernimmt den Chroma-Shift aus plane->shift_x exakt wie
//       renderer.c:1735 (rect = (luma_crop - shift) * ratio).
//   (2) Merge: vec4(Y, Cb, Cr, 1) per texelFetch beider Planes (jetzt beide
//       Luma-Res, 1:1) + pl_shader_decode_color (Matrix/Levels/Bits aus repr).
//
// !!! A/B-GATE #1 — drei trennbare Fehlerbilder beim IQ-Harness-Diff gegen Stock:
//   - horizontaler Farbsaum  -> Chroma-Shift (sx/ratio) falsch
//   - Bild vertikal gespiegelt -> gl_FragCoord-Y-Origin (texelFetch unten/oben)
//   - global falsche Farben   -> Matrix/Levels/Bits in decode_color
// Scratch-Textur für User-Shader-Hooks (pl_hook_params.get_tex). Wie Stocks
// get_hook_tex: rgba16f, renderable+sampleable+storable, pro Frame recycelt.
static pl_tex kk_get_hook_tex(void *priv, int width, int height)
{
    struct kk_renderer *kr = priv;
    int cap = (int)(sizeof(kr->hook_fbos)/sizeof(kr->hook_fbos[0]));
    if (kr->hook_fbo_used >= cap)
        return NULL;
    pl_tex *slot = &kr->hook_fbos[kr->hook_fbo_used++];
    if (!kk_get_fbo(kr, slot, width, height, true))
        return NULL;
    return *slot;
}

// User-Shader-Hooks einer Stage anwenden (spiegelt pass_hook). Genutzt für
// PL_HOOK_RGB (mpv "MAIN", custom_mpv.c:854 — CAS, Anime4K-Restore) und
// PL_HOOK_PRE_KERNEL (mpv "PREKERNEL" — Anime4K). Saved/Bind-Texturen managed
// libplacebo intern via get_tex. Resize wird vom Aufrufer über die Output-Größe
// aufgefangen. *psh wird durch das Hook-Ergebnis ersetzt; repr/color aktualisiert.
static bool kk_apply_hooks(struct kk_renderer *kr, pl_shader *psh, int w, int h,
                           enum pl_hook_stage stage, const struct pl_frame *image,
                           struct pl_color_repr *repr, struct pl_color_space *color,
                           const struct pl_render_params *params)
{
    for (int n = 0; n < params->num_hooks; n++) {
        const struct pl_hook *hook = params->hooks[n];
        if (!(hook->stages & stage))
            continue;

        struct pl_hook_params hp = {
            .gpu = kr->gpu, .dispatch = kr->dp,
            .get_tex = kk_get_hook_tex, .priv = kr,
            .stage = stage,
            .rect = { 0, 0, w, h },
            .repr = *repr, .color = *color,
            .orig_repr = &image->repr, .orig_color = &image->color,
            .components = 4,
            .src_rect = { 0, 0, w, h },
            .dst_rect = { 0, 0, w, h },
        };

        if (hook->input == PL_HOOK_SIG_TEX) {
            pl_tex hin = kk_get_hook_tex(kr, w, h);
            if (!hin || !pl_dispatch_finish(kr->dp, pl_dispatch_params(
                    .shader = psh, .target = hin)))
                return false;
            hp.tex = hin;
        } else if (hook->input == PL_HOOK_SIG_COLOR) {
            hp.sh = *psh;
        }

        struct pl_hook_res res = hook->hook(hook->priv, &hp);
        if (res.failed)
            return false;

        if (res.output == PL_HOOK_SIG_COLOR) {
            *psh = res.sh;
        } else if (res.output == PL_HOOK_SIG_TEX) {
            *psh = pl_dispatch_begin(kr->dp);
            pl_shader_sample_direct(*psh, pl_sample_src( .tex = res.tex ));
        }
        // Resize an dieser (resizable) Stage wird über die rgb_tex-Größe nach
        // dem Bake aufgefangen (CAS resized nicht).
        *repr  = res.repr;
        *color = res.color;
    }
    return true;
}

// use_linear/use_sigmoid: wie pass_scale_main — der Scaler arbeitet in Linear-
// licht (Downscale) bzw. Sigmoid-Linear (Upscale). Wird hier VOR dem Bake von
// rgb_tex angewandt (= Stock linearisiert die Quelle, bevor der Scaler sie
// als Textur sampelt). Bei true ist der Output prelinearized fürs color_map.
static bool kk_build_rgb(struct kk_renderer *kr,
                         const struct pl_frame *image,
                         const struct pl_render_params *params,
                         bool use_linear, bool use_sigmoid)
{
    const struct pl_plane *luma   = &image->planes[0];
    const struct pl_plane *chroma = &image->planes[1];
    pl_tex ltex = luma->texture, ctex = chroma->texture;
    const int w = ltex->params.w, h = ltex->params.h;
    const int cw = ctex->params.w, ch = ctex->params.h;

    if (!kk_get_fbo(kr, &kr->chroma_tex, w, h, true) ||  // storable: Polar-Compute
        !kk_get_fbo(kr, &kr->rgb_tex, w, h, false))      // raster: gl_FragCoord-Merge
        return false;

    // (1) Chroma auf Luma-Res. ratio<1 (typ. 0.5), Shift aus der Plane. Scaler =
    // params->upscaler (ewa_lanczossharp) — exakt wie Stock für SAMPLER_PLANE/UP
    // (renderer.c:643, plane_upscaler=NULL -> upscaler). Bilinear wäre ~0.94 SSIM.
    float rx = (float) cw / w, ry = (float) ch / h;
    float sx = chroma->shift_x, sy = chroma->shift_y;
    pl_shader csh = pl_dispatch_begin(kr->dp);
    struct pl_sample_src csrc = {
        .tex   = ctex,
        .rect  = { (0 - sx) * rx, (0 - sy) * ry, (w - sx) * rx, (h - sy) * ry },
        .new_w = w, .new_h = h,
    };
    if (!pl_shader_sample_polar(csh, &csrc, pl_sample_filter_params(
            .filter = *params->upscaler,
            .lut    = &kr->sampler_lut))) {
        pl_dispatch_abort(kr->dp, &csh);
        return false;
    }
    if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(
            .shader = &csh, .target = kr->chroma_tex)))
        return false;

    // (2) Merge beider Planes (beide w×h, 1:1) -> vec4(Y,Cb,Cr,1), dann decode.
    pl_shader sh = pl_dispatch_begin(kr->dp);
    pl_shader_custom(sh, &(struct pl_custom_shader) {
        .description = "kk yuv merge (biplanar 4:2:0)",
        .output      = PL_SHADER_SIG_COLOR,
        .body =
            "ivec2 kkp = ivec2(gl_FragCoord.xy);                       \n"
            "color = vec4(texelFetch(kk_luma,   kkp, 0).r,             \n"
            "             texelFetch(kk_chroma, kkp, 0).rg, 1.0);      \n",
        .num_descriptors = 2,
        .descriptors = (struct pl_shader_desc[]) {
            { .desc = { .name = "kk_luma",   .type = PL_DESC_SAMPLED_TEX },
              .binding = { .object = ltex,         .sample_mode = PL_TEX_SAMPLE_NEAREST } },
            { .desc = { .name = "kk_chroma", .type = PL_DESC_SAMPLED_TEX },
              .binding = { .object = kr->chroma_tex, .sample_mode = PL_TEX_SAMPLE_NEAREST } },
        },
    });

    struct pl_color_repr repr = image->repr;
    pl_shader_decode_color(sh, &repr, NULL);

    // RGB/MAIN-Stage-Hooks (CAS, Anime4K-Restore) — VOR Linearize, wie Stock
    // (PL_HOOK_RGB:1959, vor pass_scale_main). Aktualisiert sh + repr/col.
    struct pl_color_space col = image->color;
    if (!kk_apply_hooks(kr, &sh, w, h, PL_HOOK_RGB, image, &repr, &col, params))
        return false;

    // Linearize/Sigmoidize VOR dem Bake, falls der Scaler in Linearlicht läuft
    // (Stock pass_scale_main:2051-2060). rgb_tex hält dann prelinearisiertes RGB.
    if (use_linear || use_sigmoid)
        pl_shader_linearize(sh, &col);
    if (use_sigmoid)
        pl_shader_sigmoidize(sh, params->sigmoid_params);

    // PRE_KERNEL-Stage-Hooks (Anime4K) — nach Linearize/Sigmoid, unmittelbar vor
    // dem Scaler (Stock pass_scale_main:2062). Anime4K-Mode-A resized nicht
    // (HEIGHT MAIN.h) -> rgb_tex bleibt w×h; der Main-Scaler macht den 2x-Upscale.
    if (!kk_apply_hooks(kr, &sh, w, h, PL_HOOK_PRE_KERNEL, image, &repr, &col, params))
        return false;

    if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(
            .shader = &sh, .target = kr->rgb_tex)))
        return false;

    // HDR-Peak-Detection (Stock hdr_update_peak): Compute-Side-Effect schreibt
    // in tone_map_state, das color_map_ex liest. allow_delayed=false -> muss vor
    // dem color_map liegen (eigener Compute-Dispatch hier auf rgb_tex). dcsp-
    // Transfer = LINEAR falls rgb_tex linearisiert wurde (Downscale).
    if (params->peak_detect_params && pl_color_space_is_hdr(&image->color)) {
        struct pl_color_space dcsp = image->color;
        if (use_linear || use_sigmoid)
            dcsp.transfer = PL_COLOR_TRC_LINEAR;
        pl_shader dsh = pl_dispatch_begin(kr->dp);
        pl_shader_sample_direct(dsh, pl_sample_src( .tex = kr->rgb_tex ));
        if (!pl_shader_detect_peak(dsh, dcsp, &kr->tone_map_state,
                                   params->peak_detect_params)) {
            pl_dispatch_abort(kr->dp, &dsh);
            return false; // -> Fallback (z.B. kein SSBO/Storable)
        }
        if (!pl_dispatch_compute(kr->dp, pl_dispatch_compute_params(
                .shader = &dsh, .width = w, .height = h)))
            return false;
    }
    return true;
}

bool kk_render_image(struct kk_renderer *kr,
                     const struct pl_frame *image,
                     const struct pl_frame *target,
                     const struct pl_render_params *params)
{
    if (!kr || !kk_can_handle(image, target, params))
        return false; // -> Host fällt auf pl_render_image zurück

    kr->hook_fbo_used = 0; // Hook-Scratch-Pool pro Frame zurücksetzen

    pl_tex target_tex = target->planes[0].texture;
    const int tw = target_tex->params.w, th = target_tex->params.h;

    // Ziel-Renderfläche (aspect-fit-Rect) -> Scale-Richtung.
    pl_rect2d rc = {
        .x0 = (int) roundf(target->crop.x0), .y0 = (int) roundf(target->crop.y0),
        .x1 = (int) roundf(target->crop.x1), .y1 = (int) roundf(target->crop.y1),
    };
    if (!pl_rect_w(rc)) { rc.x0 = 0; rc.x1 = tw; }
    if (!pl_rect_h(rc)) { rc.y0 = 0; rc.y1 = th; }
    const int dst_w = abs(pl_rect_w(rc)), dst_h = abs(pl_rect_h(rc));
    const int src_w = (int) (image->crop.x1 - image->crop.x0);
    const int src_h = (int) (image->crop.y1 - image->crop.y0);

    bool is_hdr = pl_color_space_is_hdr(&image->color);
    bool up   = dst_w > src_w || dst_h > src_h;
    bool down = dst_w < src_w || dst_h < src_h;
    // HDR: Sigmoid aus (clippt [0,1], pass_scale_main:2038); use_linear bleibt
    // bei Downscale, weil rgba16f float ist (pass_scale_main:2040-2041).
    bool use_sigmoid = up && params->sigmoid_params && !is_hdr; // pass_scale_main:1997
    bool use_linear  = down;                                     // pass_scale_main:1998
    bool prelinearized = use_linear || use_sigmoid;

    // --- Stufe 1+2: Decode (Luma+Chroma -> RGB), ggf. linearisiert ---------
    if (!kk_build_rgb(kr, image, params, use_linear, use_sigmoid))
        return false; // -> Fallback auf Stock

    // --- Stufe 2b: Scale (rgb_tex -> dst). 1:1=direct, Up=polar, Down=ortho.
    // no_compute am fused Pass: er finished auf das (evtl. nicht-storable) Target.
    pl_shader sh;
    if (down) {
        // Separables Downscale, Y zuerst dann X (renderer.c:746-771): hermite,
        // anti-aliased (no_widening=false), beide Achsen teilen downscaler_lut.
        if (!kk_get_fbo(kr, &kr->scale_tmp, src_w, dst_h, true)) // storable: Compute-Ortho
            return false;
        pl_shader ysh = pl_dispatch_begin(kr->dp);
        struct pl_sample_src ysrc = {
            .tex = kr->rgb_tex, .components = 4,
            .rect = { 0, 0, src_w, src_h }, .new_w = src_w, .new_h = dst_h, // nur Y skaliert
        };
        if (!pl_shader_sample_ortho2(ysh, &ysrc, pl_sample_filter_params(
                .filter = *params->downscaler, .antiring = params->antiringing_strength,
                .lut = &kr->downscaler_lut))) {
            pl_dispatch_abort(kr->dp, &ysh);
            return false;
        }
        if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(
                .shader = &ysh, .target = kr->scale_tmp)))
            return false;

        sh = pl_dispatch_begin(kr->dp);
        struct pl_sample_src xsrc = {
            .tex = kr->scale_tmp, .components = 4,
            .rect = { 0, 0, src_w, dst_h }, .new_w = dst_w, .new_h = dst_h, // nur X skaliert
        };
        if (!pl_shader_sample_ortho2(sh, &xsrc, pl_sample_filter_params(
                .filter = *params->downscaler, .antiring = params->antiringing_strength,
                .lut = &kr->downscaler_lut, .no_compute = true))) {
            pl_dispatch_abort(kr->dp, &sh);
            return false;
        }
    } else if (up) {
        sh = pl_dispatch_begin(kr->dp);
        struct pl_sample_src ssrc = { .tex = kr->rgb_tex, .new_w = dst_w, .new_h = dst_h };
        if (!pl_shader_sample_polar(sh, &ssrc, pl_sample_filter_params(
                .filter = *params->upscaler, .lut = &kr->main_lut, .no_compute = true))) {
            pl_dispatch_abort(kr->dp, &sh);
            return false;
        }
    } else {
        sh = pl_dispatch_begin(kr->dp);
        pl_shader_sample_direct(sh, pl_sample_src( .tex = kr->rgb_tex,
                                                   .new_w = dst_w, .new_h = dst_h ));
    }
    if (use_sigmoid)
        pl_shader_unsigmoidize(sh, params->sigmoid_params);

    // --- Stufe 3: Color-Map (Tone-Map / Transfer) -------------------------
    pl_shader_color_map_ex(sh, params->color_map_params, pl_color_map_args(
        .src           = image->color,
        .dst           = target->color,
        .prelinearized = prelinearized,
        .state         = &kr->tone_map_state,
    ));

    // --- Stufe 4: Dither (≤1080p / force_dither) --------------------------
    int depth = target->repr.bits.color_depth;
    if (depth && (depth < 16 || params->force_dither) && params->dither_params) {
        struct pl_dither_params dp = *params->dither_params;
        pl_shader_dither(sh, depth, &kr->dither_state, &dp);
    }

    // --- Stufe 5: Output -> Swapchain (aspect-fit crop, blend) ------------
    return pl_dispatch_finish(kr->dp, pl_dispatch_params(
        .shader       = &sh,
        .target       = target_tex,
        .rect         = rc,
        .blend_params = params->blend_params,
    ));
}

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
#include <time.h>

#include <libplacebo/dispatch.h>
#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>
#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/sampling.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/dithering.h>
#include <libplacebo/shaders/custom.h>

#include "kk_renderer.h"

// TEMP Pass-Timing (env KUCKUCK_PASS_TIMING): isoliert pro Pass via pl_gpu_finish
// + CPU-Monotonic-Clock (Metal-pl_timer misst nur CB-Ebene -> chroma==rgb). Das
// Finish serialisiert -> Lap-Summe > echter pipelined Total, aber die Pass-AUFTEILUNG
// ist exakt (echten Total misst hybrid.log ohne Laps). g_timing: -1 ungeprüft, 0/1.
static int g_timing = -1;
static uint64_t g_last_ns, g_acc_luma, g_acc_chroma, g_acc_rgb, g_acc_scale;
static int g_lap_n;
static uint64_t kk_mono_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static void kk_lap(pl_gpu gpu, uint64_t *acc) {
    if (g_timing != 1) return;
    pl_gpu_finish(gpu);
    uint64_t now = kk_mono_ns(); *acc += now - g_last_ns; g_last_ns = now;
}

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
    pl_tex deband_luma;          // debandete Luma-Plane (native Res), falls deband an
    pl_tex deband_chroma;        // debandete Chroma-Plane (native Res), falls deband an
    pl_tex luma_tex;             // LUMA-Hook-Ergebnis (ArtCNN/FSRCNNX), falls aktiv
    pl_tex chroma_tex;
    pl_tex chroma_tmp;           // X-Pass-Zwischentextur fürs ortho-Chroma-Upsample
    pl_tex rgb_tex;
    pl_tex scale_tmp;            // Pass-1-Zwischentextur fürs Scale-Ortho-Ping-Pong

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
    pl_tex_destroy(kr->gpu, &kr->deband_luma);
    pl_tex_destroy(kr->gpu, &kr->deband_chroma);
    pl_tex_destroy(kr->gpu, &kr->luma_tex);
    pl_tex_destroy(kr->gpu, &kr->chroma_tex);
    pl_tex_destroy(kr->gpu, &kr->chroma_tmp);
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
    // Hooks: RGB/MAIN (CAS, Anime4K) + PRE_KERNEL (Anime4K) + LUMA_INPUT
    // (ArtCNN/FSRCNNX; aktiver 2x-CNN -> kk_luma_hooked liefert die resized Luma,
    // Merge/Scale laufen auf der neuen Res). Andere Stages (CHROMA/NATIVE/OUTPUT/…) -> Stock.
    for (int n = 0; n < params->num_hooks; n++)
        if (params->hooks[n]->stages &
            ~(unsigned)(PL_HOOK_RGB | PL_HOOK_PRE_KERNEL | PL_HOOK_LUMA_INPUT))
            return false;
    if (params->lut || target->lut)             return false;
    if (params->distort_params)                 return false;
    if (params->cone_params)                    return false;
    // Up- und Downscale via polar (1 Pass) oder separabel-ortho (Ping-Pong) je
    // nach filter->polar; beide Scaler müssen existieren (auch für Chroma-Upsample).
    if (!params->upscaler || !params->downscaler) return false;
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

// Debandet eine Plane in *out (Stock plane_deband, renderer.c:1769): pl_shader_deband
// mit scale=normalize (Aufrufer nutzt drepr fürs decode) + grain_neutral der Plane,
// auf nativer Plane-Auflösung. comps=1 (Luma) bzw 2 (Chroma). Liefert *out / NULL.
static pl_tex kk_deband(struct kk_renderer *kr, pl_tex *out, pl_tex src,
                        int sw, int sh_, int comps, float scale,
                        const float neutral[3], const struct pl_deband_params *dp)
{
    if (!kk_get_fbo(kr, out, sw, sh_, true))
        return NULL;
    struct pl_deband_params p = *dp;
    p.grain_neutral[0] = neutral[0];
    p.grain_neutral[1] = neutral[1];
    p.grain_neutral[2] = neutral[2];
    pl_shader dsh = pl_dispatch_begin(kr->dp);
    pl_shader_deband(dsh, pl_sample_src( .tex = src, .components = comps, .scale = scale ), &p);
    if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader = &dsh, .target = *out)))
        return NULL;
    return *out;
}

// LUMA-INPUT-Hooks (ArtCNN/FSRCNNX) auf die Luma-Plane anwenden. Liefert die zu
// mergende Luma-Textur. *resized=true, wenn ein Hook die Luma vergrößert (aktiver
// CNN-Upscale, WHEN>1.3x) -> Aufrufer fällt auf Stock zurück (Plane-Geometrie-
// Änderung nicht migriert). Inerter Hook (WHEN false -> SIG_NONE) = raw Luma.
static pl_tex kk_luma_hooked(struct kk_renderer *kr, pl_tex ltex, int w, int h,
                             const struct pl_frame *image,
                             const struct pl_render_params *params, bool *resized)
{
    *resized = false;
    bool has = false;
    for (int n = 0; n < params->num_hooks; n++)
        if (params->hooks[n]->stages & PL_HOOK_LUMA_INPUT) has = true;
    if (!has)
        return ltex;

    pl_shader sh = pl_dispatch_begin(kr->dp);
    pl_shader_sample_direct(sh, pl_sample_src( .tex = ltex, .components = 1 ));
    struct pl_color_repr repr = image->repr;
    struct pl_color_space col = image->color;
    bool modified = false;
    int cur_w = w, cur_h = h;  // wächst, wenn ein Hook (ArtCNN/FSRCNNX) hochskaliert

    for (int n = 0; n < params->num_hooks; n++) {
        const struct pl_hook *hook = params->hooks[n];
        if (!(hook->stages & PL_HOOK_LUMA_INPUT))
            continue;
        struct pl_hook_params hp = {
            .gpu = kr->gpu, .dispatch = kr->dp, .get_tex = kk_get_hook_tex, .priv = kr,
            .stage = PL_HOOK_LUMA_INPUT, .rect = { 0, 0, cur_w, cur_h }, .repr = repr, .color = col,
            .orig_repr = &image->repr, .orig_color = &image->color, .components = 1,
            .src_rect = { 0, 0, cur_w, cur_h }, .dst_rect = { 0, 0, cur_w, cur_h },
        };
        pl_tex hin = NULL;
        if (hook->input == PL_HOOK_SIG_TEX) {
            hin = kk_get_hook_tex(kr, cur_w, cur_h);
            if (!hin || !pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader = &sh, .target = hin)))
                return NULL;
            hp.tex = hin;
        } else if (hook->input == PL_HOOK_SIG_COLOR) {
            hp.sh = sh;
        }
        struct pl_hook_res res = hook->hook(hook->priv, &hp);
        if (res.failed)
            return NULL;
        int ow = cur_w, oh = cur_h;
        if (res.output == PL_HOOK_SIG_NONE) {
            // unverändert: bei SIG_TEX-Input wurde sh in hin gebacken -> aus hin neu
            if (hook->input == PL_HOOK_SIG_TEX) {
                sh = pl_dispatch_begin(kr->dp);
                pl_shader_sample_direct(sh, pl_sample_src( .tex = hin ));
            }
            continue;
        } else if (res.output == PL_HOOK_SIG_TEX) {
            ow = res.tex->params.w; oh = res.tex->params.h;
            sh = pl_dispatch_begin(kr->dp);
            pl_shader_sample_direct(sh, pl_sample_src( .tex = res.tex ));
        } else { // SIG_COLOR
            sh = res.sh;
            if (!pl_shader_output_size(sh, &ow, &oh)) { ow = cur_w; oh = cur_h; }
        }
        repr = res.repr; col = res.color; modified = true;
        if (ow != cur_w || oh != cur_h) {  // aktiver Upscaler-Hook -> neue Luma-Res
            *resized = true;
            cur_w = ow; cur_h = oh;
        }
    }

    if (!modified) { // alle Hooks inert -> raw Luma direkt (kein Extra-Pass)
        pl_dispatch_abort(kr->dp, &sh);
        return ltex;
    }
    // Bake bei der finalen (evtl. resized) Luma-Res. Caller liest luma_tex->params.w/h.
    if (!kk_get_fbo(kr, &kr->luma_tex, cur_w, cur_h, false) ||
        !pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader = &sh, .target = kr->luma_tex)))
        return NULL;
    return kr->luma_tex;
}

// use_linear/use_sigmoid: wie pass_scale_main — der Scaler arbeitet in Linear-
// licht (Downscale) bzw. Sigmoid-Linear (Upscale). Wird hier VOR dem Bake von
// rgb_tex angewandt (= Stock linearisiert die Quelle, bevor der Scaler sie
// als Textur sampelt). Bei true ist der Output prelinearized fürs color_map.
// dst_w/dst_h + is_hdr: nötig, um up/down (und damit use_linear/use_sigmoid) NACH
// einem evtl. Luma-Resize (aktiver 2x-CNN wie ArtCNN) zu bestimmen — der Resize kann
// die Scale-Richtung kippen. Gibt die echten rgb_tex-Dims (out_w/out_h = resized
// Luma-Res) + use_linear/use_sigmoid zurück, die der Aufrufer für Scale/unsigmoid braucht.
static bool kk_build_rgb(struct kk_renderer *kr,
                         const struct pl_frame *image,
                         const struct pl_render_params *params,
                         int dst_w, int dst_h, bool is_hdr,
                         int *out_w, int *out_h,
                         bool *out_linear, bool *out_sigmoid)
{
    const struct pl_plane *luma   = &image->planes[0];
    const struct pl_plane *chroma = &image->planes[1];
    pl_tex ltex = luma->texture, ctex = chroma->texture;
    const int w = ltex->params.w, h = ltex->params.h;
    const int cw = ctex->params.w, ch = ctex->params.h;
    if (g_timing == 1) g_last_ns = kk_mono_ns(); // Pass-Timing-Startmarke

    // (0) Deband pro Plane (Stock plane_deband, VOR Hooks). Normalisiert die Plane
    // -> decode_color nutzt das angepasste drepr. neutral = grain-Neutralpunkt.
    pl_tex ltex_d = ltex, ctex_d = ctex;
    struct pl_color_repr decode_repr = image->repr;
    if (params->deband_params) {
        int bits = image->repr.bits.sample_depth;
        float os = bits ? (float)(1ull<<bits) / ((1ull<<bits) - 1.0f) : 1.0f;
        float nl = (pl_color_levels_guess(&image->repr) == PL_COLOR_LEVELS_LIMITED)
                       ? 16.0f/256.0f * os : 0.0f;
        float nc = pl_color_system_is_ycbcr_like(image->repr.sys) ? 0.5f * os : nl;
        float dscale = pl_color_repr_normalize(&decode_repr); // passt decode_repr.bits an
        if (!kk_deband(kr, &kr->deband_luma, ltex, w, h, 1, dscale,
                       (float[3]){ nl, 0, 0 }, params->deband_params) ||
            !kk_deband(kr, &kr->deband_chroma, ctex, cw, ch, 2, dscale,
                       (float[3]){ nc, nc, 0 }, params->deband_params))
            return false;
        ltex_d = kr->deband_luma;
        ctex_d = kr->deband_chroma;
    }

    // LUMA-INPUT-Hooks (ArtCNN/FSRCNNX) vor dem Merge. Ein aktiver 2x-CNN vergrößert
    // die Luma -> luma_src ist mw×mh (resized); inert -> raw Luma (w×h). luma_src wird
    // im Merge statt ltex gesampelt; Chroma/Merge/rgb laufen auf mw×mh (die echte
    // Luma-Res), die Chroma-INPUT-Math (rx/rect) bleibt bei der Original-Res w/cw.
    bool luma_resized = false;
    pl_tex luma_src = kk_luma_hooked(kr, ltex_d, w, h, image, params, &luma_resized);
    if (!luma_src)
        return false; // Hook-Fehler -> Fallback auf Stock
    const int mw = luma_src->params.w, mh = luma_src->params.h;
    kk_lap(kr->gpu, &g_acc_luma); // = Deband + LUMA-Hooks (ArtCNN/FSRCNNX)

    // up/down (und damit sigmoid/linear) gegen die ECHTE Luma-Res mw×mh bestimmen —
    // ein 2x-CNN-Resize kann die Scale-Richtung kippen (z.B. 720p->1080: vorher up,
    // nach 2x-CNN 1440->1080 = down). Treibt das Prelinearize unten + den Scaler oben.
    bool up = dst_w > mw || dst_h > mh;
    bool down = dst_w < mw || dst_h < mh;
    bool use_sigmoid = up && params->sigmoid_params && !is_hdr;
    bool use_linear  = down;
    *out_w = mw; *out_h = mh; *out_linear = use_linear; *out_sigmoid = use_sigmoid;

    if (!kk_get_fbo(kr, &kr->chroma_tex, mw, mh, true) ||  // storable: Polar-Compute
        !kk_get_fbo(kr, &kr->rgb_tex, mw, mh, false))      // raster: gl_FragCoord-Merge
        return false;

    // (1) Chroma auf Luma-Res mit params->upscaler (Stock SAMPLER_PLANE/UP,
    // renderer.c:643). Shift aus der Plane. Polarer Filter (ewa) -> 1 Pass;
    // separabler Filter (lanczos, HD-Light) -> ortho X-dann-Y (Shift aufgeteilt).
    // CHROMA-LIGHT (Default AN, env KUCKUCK_CHROMA_LIGHT=0 -> voller ewa-Upscaler):
    // Chroma mit pl_filter_bilinear (separabel, 2-Tap-Triangle) statt dem teuren
    // polaren ewa_lanczossharp. Läuft über den bewährten ortho-Pfad (X+Y). Chroma
    // ist halbaufgelöst + perzeptuell tolerant -> ~4ms -> ~1-1.5ms, kaum sichtbar.
    const char *clenv = getenv("KUCKUCK_CHROMA_LIGHT");
    bool chroma_light = !clenv || clenv[0] != '0';
    const struct pl_filter_config *cf = chroma_light ? &pl_filter_bilinear : params->upscaler;
    float rx = (float) cw / w, ry = (float) ch / h;
    float sx = chroma->shift_x, sy = chroma->shift_y;
    // Output-Res = mw×mh (echte Luma-Res); rect bleibt in Chroma-INPUT-Koordinaten
    // (Original-Res-Mapping cw/w + Shift) — unabhängig von der Output-Res.
    if (cf && cf->polar) {
        pl_shader csh = pl_dispatch_begin(kr->dp);
        if (!pl_shader_sample_polar(csh, pl_sample_src(
                .tex = ctex_d, .new_w = mw, .new_h = mh,
                .rect = { (0-sx)*rx, (0-sy)*ry, (w-sx)*rx, (h-sy)*ry }),
                pl_sample_filter_params(.filter = *cf, .lut = &kr->sampler_lut))) {
            pl_dispatch_abort(kr->dp, &csh); return false;
        }
        if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader=&csh, .target=kr->chroma_tex)))
            return false;
    } else {
        // X-Pass: cw->mw (Shift sx), Y 1:1 -> chroma_tmp (mw×ch)
        if (!kk_get_fbo(kr, &kr->chroma_tmp, mw, ch, true)) return false;
        pl_shader cx = pl_dispatch_begin(kr->dp);
        if (!pl_shader_sample_ortho2(cx, pl_sample_src(
                .tex = ctex_d, .components = 2, .new_w = mw, .new_h = ch,
                .rect = { (0-sx)*rx, 0, (w-sx)*rx, ch }),
                pl_sample_filter_params(.filter = *cf, .lut = &kr->sampler_lut))) {
            pl_dispatch_abort(kr->dp, &cx); return false;
        }
        if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader=&cx, .target=kr->chroma_tmp)))
            return false;
        // Y-Pass: ch->mh (Shift sy), X 1:1 (volle chroma_tmp-Breite mw) -> chroma_tex (mw×mh)
        pl_shader cy = pl_dispatch_begin(kr->dp);
        if (!pl_shader_sample_ortho2(cy, pl_sample_src(
                .tex = kr->chroma_tmp, .components = 2, .new_w = mw, .new_h = mh,
                .rect = { 0, (0-sy)*ry, mw, (h-sy)*ry }),
                pl_sample_filter_params(.filter = *cf, .lut = &kr->sampler_lut))) {
            pl_dispatch_abort(kr->dp, &cy); return false;
        }
        if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader=&cy, .target=kr->chroma_tex)))
            return false;
    }
    kk_lap(kr->gpu, &g_acc_chroma); // = Chroma-Upscale auf Luma-Res

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
              .binding = { .object = luma_src,     .sample_mode = PL_TEX_SAMPLE_NEAREST } },
            { .desc = { .name = "kk_chroma", .type = PL_DESC_SAMPLED_TEX },
              .binding = { .object = kr->chroma_tex, .sample_mode = PL_TEX_SAMPLE_NEAREST } },
        },
    });

    // decode_repr = normalisiertes repr falls deband (bits angepasst), sonst raw.
    struct pl_color_repr repr = decode_repr;
    pl_shader_decode_color(sh, &repr, NULL);

    // RGB/MAIN-Stage-Hooks (CAS, Anime4K-Restore) — VOR Linearize, wie Stock
    // (PL_HOOK_RGB:1959, vor pass_scale_main). Aktualisiert sh + repr/col.
    struct pl_color_space col = image->color;
    if (!kk_apply_hooks(kr, &sh, mw, mh, PL_HOOK_RGB, image, &repr, &col, params))
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
    if (!kk_apply_hooks(kr, &sh, mw, mh, PL_HOOK_PRE_KERNEL, image, &repr, &col, params))
        return false;

    if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(
            .shader = &sh, .target = kr->rgb_tex)))
        return false;
    kk_lap(kr->gpu, &g_acc_rgb); // = Merge + decode_color + RGB/PRE_KERNEL-Hooks (CAS/Anime4K)

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
                .shader = &dsh, .width = mw, .height = mh)))
            return false;
    } else {
        // Keine Peak-Detection -> einen evtl. noch im State stehenden Peak (von
        // einem früheren HDR-Frame / Capture) löschen, sonst biast er das SDR-
        // color_map (-> zu dunkel). Stock: hdr_update_peak cleanup-Label.
        if (kr->tone_map_state)
            pl_reset_detected_peak(kr->tone_map_state);
    }
    return true;
}

// Skaliert src_tex (voll, src_w×src_h) -> dst_w×dst_h mit filter. 1:1=direct,
// polar=1 Pass, separabel=ortho Ping-Pong (Y in scale_tmp, X fused). Gibt die
// finale sh zurück (Aufrufer hängt unsigmoid/colormap/dither/output an), NULL=Fehler.
// no_compute am finalen Pass (er finished auf evtl. nicht-storable Target).
static pl_shader kk_scale(struct kk_renderer *kr, pl_tex src_tex,
                          int src_w, int src_h, int dst_w, int dst_h,
                          const struct pl_filter_config *filter, pl_shader_obj *lut,
                          float antiring)
{
    if (src_w == dst_w && src_h == dst_h) {
        pl_shader sh = pl_dispatch_begin(kr->dp);
        pl_shader_sample_direct(sh, pl_sample_src( .tex = src_tex, .new_w = dst_w, .new_h = dst_h ));
        return sh;
    }
    if (filter && filter->polar) {
        pl_shader sh = pl_dispatch_begin(kr->dp);
        if (!pl_shader_sample_polar(sh, pl_sample_src( .tex = src_tex, .new_w = dst_w, .new_h = dst_h ),
                pl_sample_filter_params(.filter = *filter, .lut = lut, .no_compute = true))) {
            pl_dispatch_abort(kr->dp, &sh); return NULL;
        }
        return sh;
    }
    // separabel: Y (src_h->dst_h) in scale_tmp, dann X (src_w->dst_w) fused
    if (!kk_get_fbo(kr, &kr->scale_tmp, src_w, dst_h, true)) return NULL;
    pl_shader ysh = pl_dispatch_begin(kr->dp);
    if (!pl_shader_sample_ortho2(ysh, pl_sample_src(
            .tex = src_tex, .components = 4, .rect = { 0, 0, src_w, src_h },
            .new_w = src_w, .new_h = dst_h ),
            pl_sample_filter_params(.filter = *filter, .antiring = antiring, .lut = lut))) {
        pl_dispatch_abort(kr->dp, &ysh); return NULL;
    }
    if (!pl_dispatch_finish(kr->dp, pl_dispatch_params(.shader = &ysh, .target = kr->scale_tmp)))
        return NULL;
    pl_shader sh = pl_dispatch_begin(kr->dp);
    if (!pl_shader_sample_ortho2(sh, pl_sample_src(
            .tex = kr->scale_tmp, .components = 4, .rect = { 0, 0, src_w, dst_h },
            .new_w = dst_w, .new_h = dst_h ),
            pl_sample_filter_params(.filter = *filter, .antiring = antiring, .lut = lut, .no_compute = true))) {
        pl_dispatch_abort(kr->dp, &sh); return NULL;
    }
    return sh;
}

bool kk_render_image(struct kk_renderer *kr,
                     const struct pl_frame *image,
                     const struct pl_frame *target,
                     const struct pl_render_params *params)
{
    if (!kr || !kk_can_handle(image, target, params))
        return false; // -> Host fällt auf pl_render_image zurück

    kr->hook_fbo_used = 0; // Hook-Scratch-Pool pro Frame zurücksetzen

    if (g_timing < 0)
        g_timing = getenv("KUCKUCK_PASS_TIMING") ? 1 : 0;
    if (g_timing == 1 && ++g_lap_n >= 120) { // alle 120 Frames mitteln + reset
        const char *h = getenv("HOME");
        if (h) { char p[1024]; snprintf(p, sizeof p, "%s/Documents/kk-passtiming.log", h);
            FILE *f = fopen(p, "a"); if (f) {
                fprintf(f, "luma-cnn=%.2f chroma=%.2f rgb+merge+hooks=%.2f scale+colormap=%.2f ms (avg/%d)\n",
                        g_acc_luma/1e6/g_lap_n, g_acc_chroma/1e6/g_lap_n,
                        g_acc_rgb/1e6/g_lap_n, g_acc_scale/1e6/g_lap_n, g_lap_n); fclose(f); } }
        g_acc_luma = g_acc_chroma = g_acc_rgb = g_acc_scale = 0; g_lap_n = 0;
    }

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

    // --- Stufe 1+2: Decode (Luma+Chroma -> RGB), ggf. linearisiert. Liefert die
    // ECHTE rgb_tex-Res zurück (rgb_w×rgb_h = resized Luma-Res bei aktivem 2x-CNN
    // wie ArtCNN; sonst = src) + use_linear/use_sigmoid (gegen die echte Res bestimmt,
    // weil ein Resize die Scale-Richtung kippen kann). HDR: Sigmoid aus (pass_scale
    // _main:2038); use_linear bleibt bei Downscale (rgba16f float, :2040).
    int rgb_w = src_w, rgb_h = src_h;
    bool use_linear = false, use_sigmoid = false;
    if (!kk_build_rgb(kr, image, params, dst_w, dst_h, is_hdr,
                      &rgb_w, &rgb_h, &use_linear, &use_sigmoid))
        return false; // -> Fallback auf Stock
    bool prelinearized = use_linear || use_sigmoid;

    // --- Stufe 2b: Scale (rgb_tex[rgb_w×rgb_h] -> dst). Richtung gegen die echte
    // rgb-Res (nach evtl. CNN-Resize). Up=upscaler, Down=downscaler; je polar/separabel.
    bool up   = dst_w > rgb_w || dst_h > rgb_h;
    bool down = dst_w < rgb_w || dst_h < rgb_h;
    const struct pl_filter_config *sf = up ? params->upscaler
                                            : (down ? params->downscaler : NULL);
    pl_shader_obj *slut = up ? &kr->main_lut : &kr->downscaler_lut;
    pl_shader sh = kk_scale(kr, kr->rgb_tex, rgb_w, rgb_h, dst_w, dst_h,
                            sf, slut, params->antiringing_strength);
    if (!sh)
        return false;
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
    bool ok = pl_dispatch_finish(kr->dp, pl_dispatch_params(
        .shader       = &sh,
        .target       = target_tex,
        .rect         = rc,
        .blend_params = params->blend_params,
    ));
    kk_lap(kr->gpu, &g_acc_scale); // = Main-Scale + color_map + dither + output
    return ok;
}

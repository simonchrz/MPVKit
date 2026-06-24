// Standalone Hybrid-Render-Entry (kuckuck_hybrid_*), extrahiert aus libmpvs
// video/out/render_pl.c. Rendert einen decodierten CVPixelBuffer (von AVPlayers
// AVPlayerItemVideoOutput) via libplacebo in ein Metal-Target-Texture — der
// Render-Pfad des AVPlayer-Hybrid-Players. Keine mpv-/FFmpeg-Abhängigkeit:
// nur libplacebo + CoreVideo + Metal (talloc → malloc/calloc/strdup ersetzt).
// Wird in Libplacebo.xcframework gebaut → die App braucht libmpv NICHT mehr.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <CoreVideo/CoreVideo.h>

#include <libplacebo/renderer.h>
#include <libplacebo/metal.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/cache.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/dither.h>

#include "render_mtl.h"   // öffentliche kuckuck_hybrid_*-Deklarationen
#include "kk_renderer.h"  // eigener Render-Pfad (Strangler), env-gated

struct hybrid_priv {
    pl_log pllog;
    pl_metal metal;
    pl_gpu gpu;
    pl_renderer rr;
    pl_cache cache;
    struct pl_render_params rparams;
    const struct pl_hook *user_hook;
    char *user_shader_path;
    char *deband_mode;
    struct kk_renderer *kk; // NULL bis aktiviert; eigener schmaler Pass-Graph
};

static void hybrid_log_cb(void *priv, enum pl_log_level level, const char *msg)
{
    (void) priv; (void) level;
    fprintf(stderr, "[hybrid/pl] %s\n", msg);
}

// Persistenter GLSL->MSL-Transpile-Cache (HOME/Library/Caches), shared mit dem
// (jetzt entfernten) mpv-Pfad — selber Dateiname → warmer erster Frame.
static bool kk_cache_file_path(char *buf, size_t size)
{
    const char *home = getenv("HOME");
    if (!home)
        return false;
    int n = snprintf(buf, size, "%s/Library/Caches/render_pl_msl.cache", home);
    return n > 0 && (size_t) n < size;
}

// CoreVideo-Colour-Attachments -> libplacebo-Enums. Fehlende Attachments = BT.709
// SDR (Mediathek-Live-Default). CVBufferGetAttachment = unretained (kein Release),
// warning-free am iOS-14-Deployment-Target.
static enum pl_color_system hybrid_matrix(CVPixelBufferRef pb)
{
    CFTypeRef m = CVBufferGetAttachment(pb, kCVImageBufferYCbCrMatrixKey, NULL);
    if (!m) return PL_COLOR_SYSTEM_BT_709;
    if (CFEqual(m, kCVImageBufferYCbCrMatrix_ITU_R_601_4))
        return PL_COLOR_SYSTEM_BT_601;
    if (CFEqual(m, kCVImageBufferYCbCrMatrix_SMPTE_240M_1995))
        return PL_COLOR_SYSTEM_SMPTE_240M;
    if (CFEqual(m, kCVImageBufferYCbCrMatrix_ITU_R_2020))
        return PL_COLOR_SYSTEM_BT_2020_NC;
    return PL_COLOR_SYSTEM_BT_709;
}

static enum pl_color_primaries hybrid_prim(CVPixelBufferRef pb)
{
    CFTypeRef pr = CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, NULL);
    if (!pr) return PL_COLOR_PRIM_BT_709;
    if (CFEqual(pr, kCVImageBufferColorPrimaries_EBU_3213))
        return PL_COLOR_PRIM_BT_601_625;
    if (CFEqual(pr, kCVImageBufferColorPrimaries_SMPTE_C))
        return PL_COLOR_PRIM_BT_601_525;
    if (CFEqual(pr, kCVImageBufferColorPrimaries_ITU_R_2020))
        return PL_COLOR_PRIM_BT_2020;
    return PL_COLOR_PRIM_BT_709;
}

static enum pl_color_transfer hybrid_trc(CVPixelBufferRef pb)
{
    CFTypeRef tf = CVBufferGetAttachment(pb, kCVImageBufferTransferFunctionKey, NULL);
    if (!tf) return PL_COLOR_TRC_BT_1886;
    if (CFEqual(tf, kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ))
        return PL_COLOR_TRC_PQ;
    if (CFEqual(tf, kCVImageBufferTransferFunction_ITU_R_2100_HLG))
        return PL_COLOR_TRC_HLG;
    return PL_COLOR_TRC_BT_1886;
}

// User-Shader (Anime4K/CAS/ArtCNN) aus env KUCKUCK_GLSL_SHADER hot-swappen.
// strcmp-Guard → fopen/parse nur bei echtem Pfadwechsel. user_hook MUSS eine
// stabile Adresse haben (Feld im owning struct).
static void kk_apply_user_shader(pl_gpu gpu, struct pl_render_params *rparams,
                                 const struct pl_hook **user_hook, char **shader_path)
{
    const char *path = getenv("KUCKUCK_GLSL_SHADER");
    if (!path)
        path = "";
    const char *cur = *shader_path ? *shader_path : "";
    if (strcmp(path, cur) == 0)
        return;

    if (*user_hook)
        pl_mpv_user_shader_destroy(user_hook);
    rparams->hooks = NULL;
    rparams->num_hooks = 0;
    free(*shader_path);
    *shader_path = strdup(path);

    if (!path[0])
        return;

    FILE *sf = fopen(path, "rb");
    if (!sf)
        return;
    fseek(sf, 0, SEEK_END);
    long ssz = ftell(sf);
    fseek(sf, 0, SEEK_SET);
    if (ssz > 0 && ssz < 8 * 1024 * 1024) {
        char *sbuf = malloc(ssz + 1);
        if (sbuf) {
            size_t srd = fread(sbuf, 1, ssz, sf);
            sbuf[srd] = 0;
            *user_hook = pl_mpv_user_shader_parse(gpu, sbuf, srd);
            free(sbuf);
        }
    }
    fclose(sf);
    if (*user_hook) {
        rparams->hooks = user_hook;
        rparams->num_hooks = 1;
    }
}

// Per-Content-Deband-Presets. Grain bewusst 1.0 (IQ-Harness: Grain addiert nur
// Rauschen auf Cartoon-Flächen). HDR → off (Grain verschiebt HDR-Helligkeit).
static const struct pl_deband_params deband_mild = {
    .iterations = 1, .threshold = 3.0f, .radius = 16.0f, .grain = 1.0f,
};
static const struct pl_deband_params deband_strong = {
    .iterations = 2, .threshold = 4.0f, .radius = 16.0f, .grain = 1.0f,
};

static void kk_apply_deband(struct pl_render_params *rparams, char **deband_mode)
{
    const char *mode = getenv("KUCKUCK_DEBAND");
    if (!mode)
        mode = "";
    const char *cur = *deband_mode ? *deband_mode : "";
    if (strcmp(mode, cur) == 0)
        return;
    free(*deband_mode);
    *deband_mode = strdup(mode);

    if (strcmp(mode, "off") == 0)
        rparams->deband_params = NULL;
    else if (strcmp(mode, "mild") == 0)
        rparams->deband_params = &deband_mild;
    else if (strcmp(mode, "strong") == 0)
        rparams->deband_params = &deband_strong;
    else
        rparams->deband_params = &pl_deband_default_params;
}

// CVPixelBuffer-IOSurface-Planes als pl_frame wrappen (zero-copy). NV12 -> r8/rg8,
// P010 -> r16/rg16.
static bool hybrid_wrap_pixbuf(struct hybrid_priv *p, CVPixelBufferRef pb,
                               struct pl_frame *frame, pl_tex tex_out[2])
{
    IOSurfaceRef surf = pb ? CVPixelBufferGetIOSurface(pb) : NULL;
    if (!surf)
        return false;

    OSType pf = CVPixelBufferGetPixelFormatType(pb);
    const char *lname, *cname;
    struct pl_bit_encoding bits;
    enum pl_color_levels levels;
    switch (pf) {
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
        lname = "r8"; cname = "rg8"; levels = PL_COLOR_LEVELS_LIMITED;
        bits = (struct pl_bit_encoding) { .sample_depth = 8, .color_depth = 8 };
        break;
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
        lname = "r8"; cname = "rg8"; levels = PL_COLOR_LEVELS_FULL;
        bits = (struct pl_bit_encoding) { .sample_depth = 8, .color_depth = 8 };
        break;
    case kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange:
        lname = "r16"; cname = "rg16"; levels = PL_COLOR_LEVELS_LIMITED;
        bits = (struct pl_bit_encoding) { .sample_depth = 16, .color_depth = 10,
                                          .bit_shift = 6 };
        break;
    case kCVPixelFormatType_420YpCbCr10BiPlanarFullRange:
        lname = "r16"; cname = "rg16"; levels = PL_COLOR_LEVELS_FULL;
        bits = (struct pl_bit_encoding) { .sample_depth = 16, .color_depth = 10,
                                          .bit_shift = 6 };
        break;
    default:
        return false;
    }

    int w  = (int) CVPixelBufferGetWidthOfPlane(pb, 0);
    int h  = (int) CVPixelBufferGetHeightOfPlane(pb, 0);
    int cw = (int) CVPixelBufferGetWidthOfPlane(pb, 1);
    int ch = (int) CVPixelBufferGetHeightOfPlane(pb, 1);
    pl_fmt fl = pl_find_named_fmt(p->gpu, lname);
    pl_fmt fc = pl_find_named_fmt(p->gpu, cname);
    pl_tex luma   = fl ? pl_metal_wrap_iosurface(p->metal, surf, 0, w, h, fl) : NULL;
    pl_tex chroma = fc ? pl_metal_wrap_iosurface(p->metal, surf, 1, cw, ch, fc) : NULL;
    if (!luma || !chroma) {
        pl_tex_destroy(p->gpu, &luma);
        pl_tex_destroy(p->gpu, &chroma);
        return false;
    }

    *frame = (struct pl_frame) {
        .num_planes = 2,
        .planes = {
            { .texture = luma, .components = 1,
              .component_mapping = { PL_CHANNEL_Y, PL_CHANNEL_NONE,
                                     PL_CHANNEL_NONE, PL_CHANNEL_NONE } },
            { .texture = chroma, .components = 2,
              .component_mapping = { PL_CHANNEL_CB, PL_CHANNEL_CR,
                                     PL_CHANNEL_NONE, PL_CHANNEL_NONE } },
        },
        .repr = {
            .sys    = hybrid_matrix(pb),
            .levels = levels,
            .bits   = bits,
        },
        .color = {
            .primaries = hybrid_prim(pb),
            .transfer  = hybrid_trc(pb),
        },
        .crop = { .x0 = 0, .y0 = 0, .x1 = w, .y1 = h },
    };
    pl_frame_set_chroma_location(frame, PL_CHROMA_LEFT);
    tex_out[0] = luma;
    tex_out[1] = chroma;
    return true;
}

// TEMP (AOT-Schritt 4): füllt den MSL-Cache device-nativ über das volle kk-Combo-
// Kreuzprodukt (prozeduraler Frame; Pixelinhalt egal — nur die Param-Combos
// bestimmen die GLSL). Speichert danach den Cache. Logt die Shader-Zahl.
static void kk_cap_planes(pl_gpu gpu, int w, int h, int hdr, struct pl_frame *out, pl_tex tx[2])
{
    int cw=w/2, ch=h/2;
    if (hdr) {
        uint16_t *yp=malloc((size_t)w*h*2), *cp=malloc((size_t)cw*ch*2*2);
        for (int i=0;i<w*h;i++) yp[i]=(uint16_t)(((i*977)&1023)<<6);
        for (int i=0;i<cw*ch;i++){ cp[i*2]=(uint16_t)(((i*131)&1023)<<6); cp[i*2+1]=(uint16_t)(((i*419)&1023)<<6); }
        pl_tex ty=pl_tex_create(gpu,pl_tex_params(.w=w,.h=h,.format=pl_find_named_fmt(gpu,"r16"),.sampleable=true,.initial_data=yp));
        pl_tex tc=pl_tex_create(gpu,pl_tex_params(.w=cw,.h=ch,.format=pl_find_named_fmt(gpu,"rg16"),.sampleable=true,.initial_data=cp));
        free(yp);free(cp);
        *out=(struct pl_frame){.num_planes=2,
            .planes[0]={.texture=ty,.components=1,.component_mapping={PL_CHANNEL_Y}},
            .planes[1]={.texture=tc,.components=2,.component_mapping={PL_CHANNEL_CB,PL_CHANNEL_CR}},
            .repr={.sys=PL_COLOR_SYSTEM_BT_2020_NC,.levels=PL_COLOR_LEVELS_LIMITED,.bits={16,10,6}},
            .color={.primaries=PL_COLOR_PRIM_BT_2020,.transfer=PL_COLOR_TRC_PQ},.crop={0,0,w,h}};
        tx[0]=ty;tx[1]=tc;
    } else {
        unsigned char *yp=malloc((size_t)w*h), *cp=malloc((size_t)cw*ch*2);
        for (int i=0;i<w*h;i++) yp[i]=(unsigned char)(i*7);
        for (int i=0;i<cw*ch;i++){ cp[i*2]=(unsigned char)(i*3); cp[i*2+1]=(unsigned char)(i*5); }
        pl_tex ty=pl_tex_create(gpu,pl_tex_params(.w=w,.h=h,.format=pl_find_named_fmt(gpu,"r8"),.sampleable=true,.initial_data=yp));
        pl_tex tc=pl_tex_create(gpu,pl_tex_params(.w=cw,.h=ch,.format=pl_find_named_fmt(gpu,"rg8"),.sampleable=true,.initial_data=cp));
        free(yp);free(cp);
        *out=(struct pl_frame){.num_planes=2,
            .planes[0]={.texture=ty,.components=1,.component_mapping={PL_CHANNEL_Y}},
            .planes[1]={.texture=tc,.components=2,.component_mapping={PL_CHANNEL_CB,PL_CHANNEL_CR}},
            .repr={.sys=PL_COLOR_SYSTEM_BT_709,.levels=PL_COLOR_LEVELS_LIMITED},
            .color={.primaries=PL_COLOR_PRIM_BT_709,.transfer=PL_COLOR_TRC_BT_1886},.crop={0,0,w,h}};
        tx[0]=ty;tx[1]=tc;
    }
    pl_frame_set_chroma_location(out, PL_CHROMA_LEFT);
}

static void kk_capture_all(struct hybrid_priv *p)
{
    const char *shdir = getenv("KUCKUCK_KK_CAPTURE_SHADERS");
    if (!shdir) { fprintf(stderr, "[kk-capture] KUCKUCK_KK_CAPTURE_SHADERS fehlt\n"); return; }
    const char *shaders[]={ NULL,"cas.glsl","anime4k_mode_a_mobile.glsl","anime4k_mode_a_m.glsl",
                            "ArtCNN_C4F16.glsl","artcnn_ds_cas.glsl","fsrcnnx_x2_8.glsl" };
    // DICHTER Downscale-Sweep (0.30..1.0 in 0.02-Schritten): der Polar-/Ortho-
    // Downscaler unrollt die Tap-Schleife je Ratio (ceil(radius/scale)) -> jeder
    // Tap-Bound = eigener Shader; grobe Sweeps lassen Lücken (-> KK_AOT-Schwarz).
    // + Upscale-Punkte >1.3 aktivieren die WHEN-geguardeten LUMA-Hook-CNNs.
    double scales[64]; int ns=0;
    for (double s=0.30; s<=1.001; s+=0.02) scales[ns++]=s;
    double up[]={1.1,1.2,1.3,1.5,1.8,2.0,2.3,2.6,3.0,3.5,4.0};
    for (int i=0;i<(int)(sizeof(up)/sizeof(*up));i++) scales[ns++]=up[i];
    static const struct pl_deband_params dbp={.iterations=2,.threshold=4,.radius=16,.grain=0};
    int W=640,H=360, renders=0;
    pl_fmt rgba8=pl_find_named_fmt(p->gpu,"rgba8"), rgba16f=pl_find_named_fmt(p->gpu,"rgba16f");
    for (int hdr=0;hdr<2;hdr++)
    for (int si=0; si<(int)(sizeof(shaders)/sizeof(*shaders)); si++)
    for (int oi=0; oi<2; oi++)
    for (int di=0; di<2; di++)
    for (int sc=0; sc<ns; sc++) {
        struct pl_frame img; pl_tex tx[2]; kk_cap_planes(p->gpu, W, H, hdr, &img, tx);
        int ow=(int)(W*scales[sc]), oh=(int)(H*scales[sc]);
        pl_tex out=pl_tex_create(p->gpu,pl_tex_params(.w=ow,.h=oh,.format=hdr?rgba16f:rgba8,.renderable=true,.storable=true));
        struct pl_frame target={.num_planes=1,.planes[0]={.texture=out,.components=4,.component_mapping={0,1,2,3}},
            .repr=pl_color_repr_rgb,.color=hdr?pl_color_space_hdr10:pl_color_space_srgb,.crop={0,0,ow,oh}};
        struct pl_render_params rp=pl_render_high_quality_params;
        rp.error_diffusion=&pl_error_diffusion_sierra_lite;
        rp.deband_params = di ? &dbp : NULL;
        rp.peak_detect_params = hdr ? &pl_peak_detect_high_quality_params : NULL;
        if (oi) rp.upscaler=&pl_filter_lanczos;
        const struct pl_hook *hook=NULL;
        if (shaders[si]) { char path[1024]; snprintf(path,sizeof path,"%s/%s",shdir,shaders[si]);
            FILE *f=fopen(path,"rb"); if(f){fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
                char *b=malloc(n+1);size_t r=fread(b,1,n,f);b[r]=0;fclose(f);
                hook=pl_mpv_user_shader_parse(p->gpu,b,r);free(b); if(hook){rp.hooks=&hook;rp.num_hooks=1;}}}
        if (!(p->kk && kk_render_image(p->kk,&img,&target,&rp)))
            pl_render_image(p->rr,&img,&target,&rp);
        pl_gpu_finish(p->gpu);
        renders++;
        if (hook) pl_mpv_user_shader_destroy(&hook);
        pl_tex_destroy(p->gpu,&out); pl_tex_destroy(p->gpu,&tx[0]); pl_tex_destroy(p->gpu,&tx[1]);
    }
    if (p->cache) { char path[PATH_MAX]; if (kk_cache_file_path(path,sizeof path)) {
        FILE *f=fopen(path,"wb"); if(f){pl_cache_save_file(p->cache,f);fclose(f);} } }
    fprintf(stderr, "[kk-capture] %d renders fertig, Cache gespeichert\n", renders);
}

void *kuckuck_hybrid_create(void *mtl_device)
{
    if (!mtl_device)
        return NULL;
    struct hybrid_priv *p = calloc(1, sizeof(struct hybrid_priv));
    if (!p)
        return NULL;

    p->pllog = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = hybrid_log_cb,
        .log_level = PL_LOG_WARN,
    ));
    if (!p->pllog)
        goto fail;

    p->metal = pl_metal_create(p->pllog, pl_metal_params(
        .device = mtl_device,
    ));
    if (!p->metal) {
        fprintf(stderr, "[hybrid/pl] pl_metal_create failed\n");
        goto fail;
    }
    p->gpu = p->metal->gpu;

    p->cache = pl_cache_create(pl_cache_params(
        .log            = p->pllog,
        .max_total_size = 16 << 20,
    ));
    if (p->cache) {
        char path[PATH_MAX];
        // 1) Geshippter Bundle-Cache (eviction-immun) als Basis. PFLICHT unter
        //    KK_AOT: Library/Caches kann von iOS geräumt werden -> ohne Bundle-
        //    Cache + ohne Compiler = schwarz. Bundle-Pfad via App-Env (resourcePath).
        const char *res = getenv("KUCKUCK_KK_CAPTURE_SHADERS");
        if (res) {
            char bp[PATH_MAX];
            if (snprintf(bp, sizeof(bp), "%s/render_pl_msl.cache", res) < (int) sizeof(bp)) {
                FILE *bf = fopen(bp, "rb");
                if (bf) { pl_cache_load_file(p->cache, bf); fclose(bf); }
            }
        }
        // 2) Akkumulierter Caches-Cache obendrauf (merge; nicht-AOT-Warm-Launch).
        if (kk_cache_file_path(path, sizeof(path))) {
            FILE *f = fopen(path, "rb");
            if (f) { pl_cache_load_file(p->cache, f); fclose(f); }
        }
        pl_gpu_set_cache(p->gpu, p->cache);
    }

    p->rr = pl_renderer_create(p->pllog, p->gpu);
    if (!p->rr)
        goto fail;

    p->rparams = pl_render_high_quality_params;
    p->rparams.error_diffusion = &pl_error_diffusion_sierra_lite;
    p->rparams.peak_detect_params = NULL;

    // Eigener Render-Pfad (Strangler). Nur aktiv bei KUCKUCK_KK_RENDER=1; sonst
    // bleibt p->kk NULL und es läuft ausschließlich pl_render_image (Default).
    {
        const char *kkenv = getenv("KUCKUCK_KK_RENDER");
        if (kkenv && kkenv[0] == '1')
            p->kk = kk_renderer_create(p->pllog, p->gpu);
    }

    // TEMP (AOT-Schritt 4): On-Device-Shader-Capture. KUCKUCK_KK_CAPTURE=1 +
    // KUCKUCK_KK_CAPTURE_SHADERS=<Resources-Dir> -> rendert das volle Combo-
    // Kreuzprodukt durch kk, füllt den (persistenten) MSL-Cache device-nativ.
    if (p->kk && getenv("KUCKUCK_KK_CAPTURE"))
        kk_capture_all(p);

    return p;

fail:
    kuckuck_hybrid_destroy(p);
    return NULL;
}

int kuckuck_hybrid_render(void *ctx, void *cv_pixbuf, void *target_texture)
{
    struct hybrid_priv *p = ctx;
    if (!p || !cv_pixbuf || !target_texture)
        return -1;
    CVPixelBufferRef pb = (CVPixelBufferRef) cv_pixbuf;

    kk_apply_user_shader(p->gpu, &p->rparams, &p->user_hook, &p->user_shader_path);
    kk_apply_deband(&p->rparams, &p->deband_mode);

    struct pl_frame img;
    pl_tex src_tex[2] = { NULL, NULL };
    if (!hybrid_wrap_pixbuf(p, pb, &img, src_tex))
        return -2;

    pl_tex target_tex = pl_metal_wrap_tex(p->metal, target_texture);
    if (!target_tex) {
        pl_tex_destroy(p->gpu, &src_tex[0]);
        pl_tex_destroy(p->gpu, &src_tex[1]);
        return -3;
    }

    const char *flipenv = getenv("KUCKUCK_HYBRID_FLIP");
    bool flip = flipenv && flipenv[0] == '1';
    bool hdr_target = target_tex->params.format &&
                      target_tex->params.format->type == PL_FMT_FLOAT;
    p->rparams.peak_detect_params = hdr_target ? &pl_peak_detect_high_quality_params : NULL;
    struct pl_color_space hdr_csp = pl_color_space_hdr10;
    {
        const char *hmode = getenv("KUCKUCK_HDR_TARGET");
        const char *hnits = getenv("KUCKUCK_HDR_TARGET_NITS");
        if (!(hmode && strcmp(hmode, "hdr10") == 0))
            hdr_csp.hdr.max_luma = hnits ? (float) atof(hnits) : 1000.0f;
    }
    struct pl_swapchain_frame sw_frame = {
        .fbo         = target_tex,
        .flipped     = flip,
        .color_repr  = pl_color_repr_rgb,
        .color_space = hdr_target ? hdr_csp : pl_color_space_srgb,
    };
    struct pl_frame target;
    pl_frame_from_swapchain(&target, &sw_frame);

    float tw = target_tex->params.w, th = target_tex->params.h;
    float sw = img.crop.x1 - img.crop.x0, sh = img.crop.y1 - img.crop.y0;
    if (sw > 0 && sh > 0 && tw > 0 && th > 0) {
        float src_ar = sw / sh, dst_ar = tw / th;
        if (src_ar > dst_ar) {
            float fh = tw / src_ar, y0 = (th - fh) / 2.0f;
            target.crop = (struct pl_rect2df) { .x0 = 0, .y0 = y0, .x1 = tw, .y1 = y0 + fh };
        } else {
            float fw = th * src_ar, x0 = (tw - fw) / 2.0f;
            target.crop = (struct pl_rect2df) { .x0 = x0, .y0 = 0, .x1 = x0 + fw, .y1 = th };
        }
    }

    // HD-Light-Render für ≥1080p: separabler Scaler (pl_filter_lanczos) statt des
    // teuren 2D-ewa_lanczossharp + Dither/Deband aus → Render ~16ms statt ~22ms,
    // bleibt unter dem 50fps-Budget (20ms) auch bei thermisch gedrosselter GPU
    // (sonst Frame-Drops/maxGap-Spikes bei HD-50p-Broadcast, [[project_hd_pacing_render_bound]]).
    // Der CAS-User-Shader bleibt (Schärfe). Bei ≤1080p (SD/720p) bleibt der volle
    // HQ-Pfad. Env KUCKUCK_HD_LIGHT: "1"=force on, "0"=force off, unset=auto (src_h>=1080).
    struct pl_render_params rp = p->rparams;
    {
        const char *hl = getenv("KUCKUCK_HD_LIGHT");
        bool light = hl ? (hl[0] == '1') : (sh >= 1080.0f);
        if (light) {
            rp.upscaler        = &pl_filter_lanczos;
            rp.downscaler      = &pl_filter_lanczos;
            rp.error_diffusion = NULL;
            rp.dither_params   = NULL;
            rp.deband_params   = NULL;
        }
    }
    // kk-Pfad übernimmt den Frame, wenn aktiv UND der Fall migriert ist;
    // sonst Fallback auf den bewährten Stock-Renderer (kein Regressionsrisiko).
    if (!(p->kk && kk_render_image(p->kk, &img, &target, &rp)))
        pl_render_image(p->rr, &img, &target, &rp);
    pl_gpu_finish(p->gpu);

    pl_tex_destroy(p->gpu, &target_tex);
    pl_tex_destroy(p->gpu, &src_tex[0]);
    pl_tex_destroy(p->gpu, &src_tex[1]);
    return 0;
}

void kuckuck_hybrid_destroy(void *ctx)
{
    struct hybrid_priv *p = ctx;
    if (!p)
        return;
    if (p->kk)
        kk_renderer_destroy(&p->kk);
    if (p->user_hook)
        pl_mpv_user_shader_destroy(&p->user_hook);
    if (p->rr)
        pl_renderer_destroy(&p->rr);
    if (p->cache) {
        char path[PATH_MAX];
        if (kk_cache_file_path(path, sizeof(path))) {
            FILE *f = fopen(path, "wb");
            if (f) { pl_cache_save_file(p->cache, f); fclose(f); }
        }
        if (p->gpu)
            pl_gpu_set_cache(p->gpu, NULL);
        pl_cache_destroy(&p->cache);
    }
    if (p->metal)
        pl_metal_destroy(&p->metal);
    if (p->pllog)
        pl_log_destroy(&p->pllog);
    free(p->user_shader_path);
    free(p->deband_mode);
    free(p);
}

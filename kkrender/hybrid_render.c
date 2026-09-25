// Standalone Hybrid-Render-Entry (kuckuck_hybrid_*) — rendert einen decodierten
// CVPixelBuffer (von AVPlayers AVPlayerItemVideoOutput) via den EIGENEN Metal-Renderer
// (kk_gpu) in ein Metal-Target-Texture. KEINE libplacebo-/mpv-/FFmpeg-Abhängigkeit mehr
// (libplacebo-Drop abgeschlossen): nur kk_gpu + CoreVideo + Metal.
//
// kk_gpu deckt die einzigen AVPlayer-VideoToolbox-Formate: NV12 (8-bit SDR) + P010
// (10-bit HDR). Andere Formate -> nichts gerendert (return -2). KEIN Fallback mehr.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <pthread.h>
#include <CoreVideo/CoreVideo.h>

#include "render_mtl.h"   // öffentliche kuckuck_hybrid_*-Deklarationen
#include "kk_gpu.h"       // kk_hdr_params + native Generatoren

// kk_gpu-Render-Entries (in kk_gpu_render.c).
extern bool kk_gpu_render(void *metal_device, void *cv_pixbuf, void *target_texture,
                          const float *yuv2rgb, const float *prim2disp,
                          void (*done)(void*), void *done_ud);
extern bool kk_gpu_render_hdr(void *metal_device, void *cv_pixbuf, void *target_texture,
                              const kk_hdr_params *hp,
                              void (*done)(void*), void *done_ud);
// Native Color-Parameter-Generatoren (kk_gpu_genparams.c).
extern void kk_sdr_decode_matrix(int sys, int full, float out[12]);
extern void kk_primaries_to709(int prim, float out[9]);
extern void kk_hdr_tone(float src_max_nits, float dst_max_nits, float min_nits,
                        float *in_min, float *in_max, float *out_min, float *out_max, float lut[256]);
extern const float KK_IPT_RGB2LMS_2020[9], KK_IPT_LMS2RGB_2020[9], KK_IPT_LMS2IPT[9], KK_IPT_IPT2LMS[9];

// kk_gpu ist PROZESSWEIT: ein `g_kk` (ein Command-Buffer/Encoder) und alle Zwischen-
// texturen sind global. Mehrere Kontexte gibt es aber doch — beim Übergang zur nächsten
// Folge rendert der neue Player schon, während der alte noch abgebaut wird (belegt:
// `teardown` des alten nach `create ctx` des neuen im hybrid.log). Bis 2026-09-25 gab
// `destroy` dann die Caches ALLER frei, während der neue sie benutzte (use-after-free),
// und zwei renderQueues teilten sich unsynchronisiert Encoder und PSO-Cache.
// Darum: jeder Einstieg in den Render-Kontext unter `g_render_lock`, Freigabe erst beim
// LETZTEN Kontext. Deblock hat einen eigenen kk_gpu (`g_dbl`) → eigene Sperre, damit
// Stufe A (srQueue) und Stufe B (renderQueue) desselben Players parallel bleiben.
static pthread_mutex_t g_render_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_dbl_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_kontexte = 0;

struct hybrid_priv {
    float smoothPeak;   // geglätteter HDR-Bildpeak — PRO Kontext (vorher static: neues
    unsigned hdrFrame;  // Video startete mit dem Peak des vorigen)
    void *device;   // app-MTLDevice (id<MTLTexture>-Quelle); an kk_gpu durchgereicht
};

// CoreVideo-YCbCr-Matrix-Attachment -> kk_sdr_decode_matrix-Index (0=601,1=709,2=240M,3=2020NC).
// Über den H.273-CODEPUNKT, nicht per Vergleich mit den benannten Konstanten: SD-
// Aufnahmen (H.264, matrix_coefficients=5 = BT.470BG) tragen „YCbCrMatrix#5", wofür
// CoreVideo keine Konstante hat — der Vergleich fiel auf 709 durch (gemessen
// 2026-09-25, VOX-Aufnahme). Fehlt das Etikett ganz: 709 (Mediathek-Live-Default).
int hybrid_sysidx_fuer(CFTypeRef m)
{
    if (!m || CFGetTypeID(m) != CFStringGetTypeID()) return 1;
    switch (CVYCbCrMatrixGetIntegerCodePointForString((CFStringRef) m)) {
        case 5: case 6: return 0;   // BT.470BG (625) / SMPTE 170M (525) = BT.601
        case 7:         return 2;   // SMPTE 240M
        case 9: case 10: return 3;  // BT.2020 NCL (CL näherungsweise wie NCL)
        default:        return 1;   // 1 = BT.709, unbekannt → 709
    }
}
static int hybrid_sysidx(CVPixelBufferRef pb)
{
    return hybrid_sysidx_fuer(CVBufferGetAttachment(pb, kCVImageBufferYCbCrMatrixKey, NULL));
}

// CoreVideo-Primaries-Attachment -> kk_primaries_to709-Index (0=601-525,1=601-625,2=2020; -1=709/identity).
// Ebenfalls über den Codepunkt (5 = BT.470BG/EBU, 6/7 = SMPTE 170M/240M = SMPTE-C).
int hybrid_primidx_fuer(CFTypeRef pr)
{
    if (!pr || CFGetTypeID(pr) != CFStringGetTypeID()) return -1;
    switch (CVColorPrimariesGetIntegerCodePointForString((CFStringRef) pr)) {
        case 5:         return 1;   // EBU 3213 (601-625)
        case 6: case 7: return 0;   // SMPTE-C (601-525)
        case 9:         return 2;   // BT.2020
        default:        return -1;  // 1 = BT.709 / unbekannt → Identität
    }
}
static int hybrid_primidx(CVPixelBufferRef pb)
{
    return hybrid_primidx_fuer(CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, NULL));
}

// Quell-Peak (nits) aus den HDR-Metadaten: Mastering-Display-Max (ST 2086) bevorzugt,
// sonst MaxCLL (CEA-861.3), sonst 1000 (statischer Fallback). Range-Guard [1,10000]
// fängt fehlende/fehl-interpretierte Blobs ab.
static float hybrid_src_peak(CVPixelBufferRef pb)
{
    CFTypeRef md = CVBufferGetAttachment(pb, kCVImageBufferMasteringDisplayColorVolumeKey, NULL);
    if (md && CFGetTypeID(md) == CFDataGetTypeID() && CFDataGetLength((CFDataRef) md) >= 24) {
        const uint8_t *b = CFDataGetBytePtr((CFDataRef) md);
        uint32_t maxlum = ((uint32_t)b[16]<<24)|((uint32_t)b[17]<<16)|((uint32_t)b[18]<<8)|b[19]; // 0.0001 cd/m²
        float n = maxlum / 10000.0f;
        if (n >= 1.0f && n <= 10000.0f) return n;
    }
    CFTypeRef cll = CVBufferGetAttachment(pb, kCVImageBufferContentLightLevelInfoKey, NULL);
    if (cll && CFGetTypeID(cll) == CFDataGetTypeID() && CFDataGetLength((CFDataRef) cll) >= 2) {
        const uint8_t *b = CFDataGetBytePtr((CFDataRef) cll);
        uint32_t maxcll = ((uint32_t)b[0]<<8)|b[1];   // MaxCLL (nits, big-endian)
        if (maxcll >= 1 && maxcll <= 10000) return (float) maxcll;
    }
    return 1000.0f;
}

// Dynamischer Frame-Peak (nits): per CPU-Raster (96×96) über Luma UND Chroma der P010-
// Planes → R'G'B' (BT.2020-NCL) → max(R',G',B') → PQ-EOTF → nits.
// Bis 2026-09-25 nur max-Luma auf 48×48: ein gesättigtes Rot mit 1000 nit hat Y' ≈ 0,2
// (≈ 6 nit) — die Schätzung lag systematisch zu tief, und das Tonemapping schnitt die
// Lichter bei der geschätzten Spitze hart ab (Sweep kk_gpu). -1 bei Fehler. Läuft auf
// der Render-Queue (off-main), VOR dem GPU-Render.
float hybrid_frame_peak_nits(CVPixelBufferRef pb)
{
    if (CVPixelBufferGetPlaneCount(pb) < 2) return -1;
    if (CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return -1;
    int w = (int) CVPixelBufferGetWidthOfPlane(pb, 0), h = (int) CVPixelBufferGetHeightOfPlane(pb, 0);
    int cw = (int) CVPixelBufferGetWidthOfPlane(pb, 1), chh = (int) CVPixelBufferGetHeightOfPlane(pb, 1);
    size_t bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0), cbpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
    const uint8_t *base = (const uint8_t *) CVPixelBufferGetBaseAddressOfPlane(pb, 0);
    const uint8_t *cbase = (const uint8_t *) CVPixelBufferGetBaseAddressOfPlane(pb, 1);
    float mx = -1.0f;   // max R'G'B' (PQ-kodiert, 0..1)
    if (base && cbase && w > 0 && h > 0 && cw > 0 && chh > 0) {
        int sy = h > 96 ? h / 96 : 1, sx = w > 96 ? w / 96 : 1;
        for (int y = 0; y < h; y += sy) {
            const uint16_t *row = (const uint16_t *)(base + (size_t) y * bpr);
            int cy = y * chh / h; if (cy >= chh) cy = chh - 1;
            const uint16_t *crow = (const uint16_t *)(cbase + (size_t) cy * cbpr);
            for (int x = 0; x < w; x += sx) {
                int cx = x * cw / w; if (cx >= cw) cx = cw - 1;
                // P010: Werte in den oberen 10 Bit; limited range (Y 64..940, C 64..960).
                float Y = ((float)(row[x] >> 6) - 64.0f) / 876.0f;
                float Cb = ((float)(crow[2*cx] >> 6) - 512.0f) / 896.0f;
                float Cr = ((float)(crow[2*cx+1] >> 6) - 512.0f) / 896.0f;
                float R = Y + 1.4746f * Cr, G = Y - 0.16455f * Cb - 0.57135f * Cr, B = Y + 1.8814f * Cb;
                float m = R > G ? R : G; if (B > m) m = B;
                if (m > mx) mx = m;
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    if (mx <= 0) return -1;
    if (mx > 1) mx = 1;
    const float m1=0.1593017578125f, m2=78.84375f, c1=0.8359375f, c2=18.8515625f, c3=18.6875f;
    float ep = powf(mx, 1.0f/m2); float num = ep - c1; if (num < 0) num = 0; float den = c2 - c3*ep;
    return powf(num/den, 1.0f/m1) * 10000.0f;             // PQ-EOTF -> nits
}

void *kuckuck_hybrid_create(void *mtl_device)
{
    if (!mtl_device)
        return NULL;
    struct hybrid_priv *p = calloc(1, sizeof(struct hybrid_priv));
    if (!p)
        return NULL;
    p->device = mtl_device;
    p->smoothPeak = -1.0f;
    pthread_mutex_lock(&g_render_lock);
    g_kontexte++;
    pthread_mutex_unlock(&g_render_lock);
    return p;
}

// Async-Variante: Encode läuft synchron (Quell-Pointer-Zugriff bleibt im Caller-Scope),
// Commit OHNE Warten — done(ud) feuert auf Metals Completion-Thread, wenn der Frame
// fertig gerendert ist (Quell-/Target-Wraps leben intern bis dahin). Rückgabe 0 =
// angenommen (done kommt GENAU EINMAL, auch bei leerem CB), negativ = nichts encodet
// (done kommt NICHT).
static int hybrid_render_locked(struct hybrid_priv *p, void *cv_pixbuf, void *target_texture,
                                void (*done)(void *ud), void *ud);

int kuckuck_hybrid_render_async(void *ctx, void *cv_pixbuf, void *target_texture,
                                void (*done)(void *ud), void *ud)
{
    // Encode synchron unter der Sperre; `done` feuert später auf Metals Thread (ohne Sperre).
    pthread_mutex_lock(&g_render_lock);
    int rc = hybrid_render_locked(ctx, cv_pixbuf, target_texture, done, ud);
    pthread_mutex_unlock(&g_render_lock);
    return rc;
}

static int hybrid_render_locked(struct hybrid_priv *p, void *cv_pixbuf, void *target_texture,
                                void (*done)(void *ud), void *ud)
{
    if (!p || !cv_pixbuf || !target_texture)
        return -1;
    CVPixelBufferRef pb = (CVPixelBufferRef) cv_pixbuf;
    OSType pfmt = CVPixelBufferGetPixelFormatType(pb);

    // HDR (P010): IPT-Tonemap zum EDR-Peak -> PQ/2020. LUT-Gen gecacht (teuer).
    if (pfmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
        pfmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange) {
        static kk_hdr_params hp; static float cached_dst = -1.0f, cached_src = -1.0f;
        const char *nits = getenv("KUCKUCK_HDR_TARGET_NITS");
        float dst_peak = nits ? (float) atof(nits) : 203.0f;
        // `!(x >= 1 && x <= 10000)` fängt auch NaN/inf ab („nan" kam per atof durch → NaN-LUT).
        if (!(dst_peak >= 1.0f && dst_peak <= 10000.0f)) dst_peak = 203.0f;
        float mastering = hybrid_src_peak(pb);   // Mastering/MaxCLL = Obergrenze (Master nicht überschreitbar)
        // Dynamischer Peak: jeden 4. Frame den echten Frame-Peak messen, EMA-glätten,
        // durch Mastering deckeln → per-Szene-Tonemapping (dunkle Szenen nutzen EDR besser).
        if ((p->hdrFrame++ & 3) == 0) {
            float fp = hybrid_frame_peak_nits(pb);
            if (fp > 0) {
                if (fp > mastering) fp = mastering;
                // Asymmetrisch: heller schnell übernehmen (sonst ~1,2 s abgeschnittene
                // Lichter nach einem Schnitt ins Helle), dunkler langsam (kein Pumpen).
                float k = (fp > p->smoothPeak) ? 0.5f : 0.1f;
                p->smoothPeak = (p->smoothPeak < 0) ? fp : p->smoothPeak + k * (fp - p->smoothPeak);
            }
        }
        float src_peak = (p->smoothPeak > 0) ? p->smoothPeak : mastering;
        if (src_peak < 100.0f) src_peak = 100.0f;                 // Floor (nicht über-abdunkeln)
        src_peak = roundf(src_peak / 25.0f) * 25.0f;              // 25-nit-quantisiert → LUT-Regen nur bei echter Änderung
        if (dst_peak != cached_dst || src_peak != cached_src) {
            memcpy(hp.rgb2lms, KK_IPT_RGB2LMS_2020, 9 * sizeof(float));
            memcpy(hp.lms2rgb, KK_IPT_LMS2RGB_2020, 9 * sizeof(float));
            memcpy(hp.lms2ipt, KK_IPT_LMS2IPT, 9 * sizeof(float));
            memcpy(hp.ipt2lms, KK_IPT_IPT2LMS, 9 * sizeof(float));
            kk_hdr_tone(src_peak, dst_peak, 0.005f, &hp.in_min, &hp.in_max, &hp.out_min, &hp.out_max, hp.tone_lut);
            cached_dst = dst_peak; cached_src = src_peak;
        }
        return kk_gpu_render_hdr(p->device, cv_pixbuf, target_texture, &hp, done, ud) ? 0 : -2;
    }

    // SDR-NV12: echte YUV->RGB-Matrix + Primaries->709, beide aus eingebackenen Tabellen.
    if (pfmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        pfmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        bool full = (pfmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        float yuv2rgb[12]; kk_sdr_decode_matrix(hybrid_sysidx(pb), full ? 1 : 0, yuv2rgb);
        float prim2disp[9]; kk_primaries_to709(hybrid_primidx(pb), prim2disp);
        return kk_gpu_render(p->device, cv_pixbuf, target_texture, yuv2rgb, prim2disp, done, ud) ? 0 : -2;
    }

    return -2;   // unbekanntes Format (kommt von AVPlayer-VT nicht vor)
}

int kuckuck_hybrid_render(void *ctx, void *cv_pixbuf, void *target_texture)
{
    return kuckuck_hybrid_render_async(ctx, cv_pixbuf, target_texture, NULL, NULL);
}

void kuckuck_hybrid_prewarm(void *ctx)
{
    extern void kk_gpu_prewarm(void *metal_device);
    struct hybrid_priv *p = ctx;
    if (!p) return;
    pthread_mutex_lock(&g_render_lock);   // PSO-Cache (NSMutableDictionary) ist nicht threadsicher
    kk_gpu_prewarm(p->device);
    pthread_mutex_unlock(&g_render_lock);
}

int kuckuck_hybrid_deblock_nv12(void *ctx, void *src_pixbuf, void *dst_pixbuf)
{
    extern bool kk_gpu_deblock_nv12(void *metal_device, void *src_pb, void *dst_pb);
    struct hybrid_priv *p = ctx;
    if (!p) return -1;
    pthread_mutex_lock(&g_dbl_lock);
    bool ok = kk_gpu_deblock_nv12(p->device, src_pixbuf, dst_pixbuf);
    pthread_mutex_unlock(&g_dbl_lock);
    return ok ? 0 : -2;
}

void kuckuck_hybrid_destroy(void *ctx)
{
    extern void kk_gpu_release_all(void);
    if (!ctx) return;
    pthread_mutex_lock(&g_render_lock);
    // Caches erst freigeben, wenn KEIN Kontext mehr lebt (s. g_render_lock oben).
    if (--g_kontexte <= 0) { g_kontexte = 0; kk_gpu_release_all(); }
    pthread_mutex_unlock(&g_render_lock);
    free(ctx);
}

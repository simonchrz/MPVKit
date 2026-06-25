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

#include <CoreVideo/CoreVideo.h>

#include "render_mtl.h"   // öffentliche kuckuck_hybrid_*-Deklarationen
#include "kk_gpu.h"       // kk_hdr_params + native Generatoren

// kk_gpu-Render-Entries (in kk_gpu_render.c).
extern bool kk_gpu_render(void *metal_device, void *cv_pixbuf, void *target_texture,
                          const float *yuv2rgb, const float *prim2disp);
extern bool kk_gpu_render_hdr(void *metal_device, void *cv_pixbuf, void *target_texture,
                              const kk_hdr_params *hp);
// Native Color-Parameter-Generatoren (kk_gpu_genparams.c).
extern void kk_sdr_decode_matrix(int sys, int full, float out[12]);
extern void kk_primaries_to709(int prim, float out[9]);
extern void kk_hdr_tone(float src_max_nits, float dst_max_nits, float min_nits,
                        float *in_min, float *in_max, float *out_min, float *out_max, float lut[256]);
extern const float KK_IPT_RGB2LMS_2020[9], KK_IPT_LMS2RGB_2020[9], KK_IPT_LMS2IPT[9], KK_IPT_IPT2LMS[9];

struct hybrid_priv {
    void *device;   // app-MTLDevice (id<MTLTexture>-Quelle); an kk_gpu durchgereicht
};

// CoreVideo-YCbCr-Matrix-Attachment -> kk_sdr_decode_matrix-Index (0=601,1=709,2=240M,3=2020NC).
// Fehlend = BT.709 (Mediathek-Live-Default). CVBufferGetAttachment = unretained.
static int hybrid_sysidx(CVPixelBufferRef pb)
{
    CFTypeRef m = CVBufferGetAttachment(pb, kCVImageBufferYCbCrMatrixKey, NULL);
    if (!m) return 1;
    if (CFEqual(m, kCVImageBufferYCbCrMatrix_ITU_R_601_4))      return 0;
    if (CFEqual(m, kCVImageBufferYCbCrMatrix_SMPTE_240M_1995))  return 2;
    if (CFEqual(m, kCVImageBufferYCbCrMatrix_ITU_R_2020))       return 3;
    return 1;
}

// CoreVideo-Primaries-Attachment -> kk_primaries_to709-Index (0=601-525,1=601-625,2=2020; -1=709/identity).
static int hybrid_primidx(CVPixelBufferRef pb)
{
    CFTypeRef pr = CVBufferGetAttachment(pb, kCVImageBufferColorPrimariesKey, NULL);
    if (!pr) return -1;
    if (CFEqual(pr, kCVImageBufferColorPrimaries_EBU_3213)) return 1;  // 601-625
    if (CFEqual(pr, kCVImageBufferColorPrimaries_SMPTE_C))  return 0;  // 601-525
    if (CFEqual(pr, kCVImageBufferColorPrimaries_ITU_R_2020)) return 2;
    return -1;
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

// Dynamischer Frame-Peak (nits): max-Luma per CPU-Grid-Sample der P010-Luma-Plane
// (Plane 0, 16-bit, PQ-limited) -> PQ-EOTF -> nits. Grob (~48×48-Grid) aber + EMA-Glättung
// im Aufrufer = per-Szene-Peak. -1 bei Fehler. Läuft auf der Render-Queue (off-main),
// VOR dem GPU-Render (kein Concurrency mit dem IOSurface-GPU-Read).
static float hybrid_frame_peak_nits(CVPixelBufferRef pb)
{
    if (CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return -1;
    int w = (int) CVPixelBufferGetWidthOfPlane(pb, 0), h = (int) CVPixelBufferGetHeightOfPlane(pb, 0);
    size_t bpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
    const uint8_t *base = (const uint8_t *) CVPixelBufferGetBaseAddressOfPlane(pb, 0);
    uint16_t mx = 0;
    if (base && w > 0 && h > 0) {
        int sy = h > 48 ? h / 48 : 1, sx = w > 48 ? w / 48 : 1;
        for (int y = 0; y < h; y += sy) {
            const uint16_t *row = (const uint16_t *)(base + (size_t) y * bpr);
            for (int x = 0; x < w; x += sx) { uint16_t v = row[x]; if (v > mx) mx = v; }
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    if (mx == 0) return -1;
    float y10 = (float)(mx >> 6);                         // 16-bit P010 -> 10-bit
    float ypq = (y10 - 64.0f) / 876.0f;                   // BT.2020-limited (wie MKPQ)
    if (ypq < 0) ypq = 0; if (ypq > 1) ypq = 1;
    const float m1=0.1593017578125f, m2=78.84375f, c1=0.8359375f, c2=18.8515625f, c3=18.6875f;
    float ep = powf(ypq, 1.0f/m2); float num = ep - c1; if (num < 0) num = 0; float den = c2 - c3*ep;
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
    return p;
}

int kuckuck_hybrid_render(void *ctx, void *cv_pixbuf, void *target_texture)
{
    struct hybrid_priv *p = ctx;
    if (!p || !cv_pixbuf || !target_texture)
        return -1;
    CVPixelBufferRef pb = (CVPixelBufferRef) cv_pixbuf;
    OSType pfmt = CVPixelBufferGetPixelFormatType(pb);

    // HDR (P010): IPT-Tonemap zum EDR-Peak -> PQ/2020. LUT-Gen gecacht (teuer).
    if (pfmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange ||
        pfmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange) {
        static kk_hdr_params hp; static float cached_dst = -1.0f, cached_src = -1.0f;
        static float smoothPeak = -1.0f; static unsigned hdrFrame = 0;
        const char *nits = getenv("KUCKUCK_HDR_TARGET_NITS");
        float dst_peak = nits ? (float) atof(nits) : 203.0f; if (dst_peak < 1.0f) dst_peak = 203.0f;
        float mastering = hybrid_src_peak(pb);   // Mastering/MaxCLL = Obergrenze (Master nicht überschreitbar)
        // Dynamischer Peak: jeden 4. Frame den echten Frame-Peak messen, EMA-glätten,
        // durch Mastering deckeln → per-Szene-Tonemapping (dunkle Szenen nutzen EDR besser).
        if ((hdrFrame++ & 3) == 0) {
            float fp = hybrid_frame_peak_nits(pb);
            if (fp > 0) {
                if (fp > mastering) fp = mastering;
                smoothPeak = (smoothPeak < 0) ? fp : smoothPeak + 0.15f * (fp - smoothPeak);
            }
        }
        float src_peak = (smoothPeak > 0) ? smoothPeak : mastering;
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
        return kk_gpu_render_hdr(p->device, cv_pixbuf, target_texture, &hp) ? 0 : -2;
    }

    // SDR-NV12: echte YUV->RGB-Matrix + Primaries->709, beide aus eingebackenen Tabellen.
    if (pfmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
        pfmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        bool full = (pfmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange);
        float yuv2rgb[12]; kk_sdr_decode_matrix(hybrid_sysidx(pb), full ? 1 : 0, yuv2rgb);
        float prim2disp[9]; kk_primaries_to709(hybrid_primidx(pb), prim2disp);
        return kk_gpu_render(p->device, cv_pixbuf, target_texture, yuv2rgb, prim2disp) ? 0 : -2;
    }

    return -2;   // unbekanntes Format (kommt von AVPlayer-VT nicht vor)
}

void kuckuck_hybrid_destroy(void *ctx)
{
    free(ctx);
}

// kk_gpu.h — schlanke Metal-GPU-Abstraktion (libplacebo-Ablösung, Etappe 1).
//
// Ersetzt schrittweise pl_gpu/pl_dispatch: nur das, was der kk-Render-Pfad
// braucht (Texturen, Compute-/Raster-Pass aus MSL, PSO-Cache, Download). Reines
// Metal, kein Vulkan/SPIR-V. C-API (von kk_renderer.c nutzbar), Impl in kk_gpu.m.
//
// Methodik: libplacebo bleibt parallel als Referenz, bis jeder Pass hier nativ
// läuft + per IQ-Harness pixelgleich gegen den pl_*-Pfad verifiziert ist.
#ifndef KK_GPU_H
#define KK_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct kk_gpu kk_gpu;
typedef struct kk_tex kk_tex;
typedef struct kk_pass kk_pass;

// Pixelformat (nur was der Render-Pfad braucht).
typedef enum {
    KK_FMT_R8,        // Luma 8-bit
    KK_FMT_RG8,       // Chroma 8-bit
    KK_FMT_R16,       // Luma 10/16-bit
    KK_FMT_RG16,      // Chroma 10/16-bit
    KK_FMT_RGBA8,     // SDR-Output / Zwischentextur
    KK_FMT_BGRA8,     // Display-IOSurface-Output (AVSampleBufferDisplayLayer ist BGRA)
    KK_FMT_RGBA16F,   // HDR/Linearlicht-Zwischentextur
} kk_fmt;

typedef enum {
    KK_TEX_SAMPLE    = 1 << 0,  // als Eingang sampelbar
    KK_TEX_STORAGE   = 1 << 1,  // Compute-write (image2D)
    KK_TEX_RENDER    = 1 << 2,  // Raster-Target
    KK_TEX_DOWNLOAD  = 1 << 3,  // host-readable (Harness/Diff)
} kk_tex_usage;

// --- Kontext --------------------------------------------------------------
// mtl_device: id<MTLDevice> (void* aus Swift/App); NULL = System-Default.
kk_gpu *kk_gpu_create(void *mtl_device);
void    kk_gpu_destroy(kk_gpu **pgpu);
void   *kk_gpu_mtl_device(kk_gpu *gpu);   // id<MTLDevice> für Interop
void    kk_gpu_finish(kk_gpu *gpu);       // blockt bis GPU fertig (Bench/Download)

// --- Texturen -------------------------------------------------------------
kk_tex *kk_tex_create(kk_gpu *gpu, int w, int h, kk_fmt fmt, uint32_t usage,
                      const void *initial_data); // initial_data optional (Upload)
kk_tex *kk_tex_create_3d(kk_gpu *gpu, int w, int h, int d, kk_fmt fmt,
                         const void *initial_data); // 3D-LUT (z.B. Gamut-Map), texture3d
kk_tex *kk_tex_wrap_iosurface(kk_gpu *gpu, void *iosurface, int plane,
                              kk_fmt fmt, uint32_t usage); // zero-copy In(SAMPLE)/Out(STORAGE)
kk_tex *kk_tex_wrap_mtltexture(kk_gpu *gpu, void *mtltexture); // existierende MTLTexture (Display-Target, geteiltes Device)
void    kk_tex_destroy(kk_gpu *gpu, kk_tex **ptex);
int     kk_tex_w(const kk_tex *t);
int     kk_tex_h(const kk_tex *t);
bool    kk_tex_download(kk_gpu *gpu, kk_tex *t, void *dst); // nach kk_gpu_finish

// --- Compute-Pass ---------------------------------------------------------
// msl_source: vollständiger MSL-Kernel-Quelltext; entry: Kernel-Funktionsname.
// Die PSO wird nach (source-hash, entry) gecacht -> Compile nur einmal.
// Bindings: samp_tex[] an buffer/texture-Index 0..n-1 (sampelbar), out_tex als
// storage (image), uniforms als constant-Buffer. dispatch über das Output-Grid.
typedef struct {
    kk_tex      *out;                 // Storage-Target (write)
    kk_tex      *in[8]; int n_in;     // sampelbare Eingänge
    const void  *uniforms; size_t uniforms_size; // constant-Buffer (push-const-Äquiv.)
    bool         linear[8];           // pro Eingang: linear (true) / nearest sampling
} kk_compute_args;

bool kk_gpu_compute(kk_gpu *gpu, const char *msl_source, const char *entry,
                    const kk_compute_args *args);

#endif // KK_GPU_H

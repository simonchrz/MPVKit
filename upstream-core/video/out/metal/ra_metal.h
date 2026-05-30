/* Copyright (C) 2026 the mpv developers
 *
 * Internal header for the Apple Metal backend of mpv's render abstraction.
 * Parallel to video/out/opengl/ra_gl.h.
 */

#ifndef MP_RA_METAL_H
#define MP_RA_METAL_H

#import <Metal/Metal.h>
#include "video/out/gpu/ra.h"
#include "video/out/gpu/context.h"

/* Public API used by libmpv_metal.m -- all are implemented in ra_metal.m */

/**
 * Initialize a `ra` instance backed by Metal.
 * `device` must be a valid `id<MTLDevice>` (non-NULL).
 * `queue` may be NULL — caller-supplied command queue is preferred; if NULL,
 * a new one is allocated from `device`.
 * Fills `ra_ctx->ra` on success. Returns true on success.
 */
bool ra_init_metal(struct ra_ctx *ctx, id<MTLDevice> device,
                   id<MTLCommandQueue> queue);

/**
 * Wrap a caller-provided `id<MTLTexture>` (typically a CAMetalDrawable's
 * texture) as a transient `ra_tex` for the current frame. Returned tex is
 * valid until the next call to ra_metal_commit_frame().
 */
struct ra_tex *ra_metal_wrap_drawable(struct ra_ctx *ctx,
                                      id<MTLTexture> texture,
                                      int w, int h,
                                      id<MTLCommandBuffer> command_buffer);

/**
 * Wrap an externally-owned MTLTexture (= z.B. CVMetalTextureCache-erzeugt
 * von VideoToolbox hwdec) als sample-able ra_tex. Caller retains ownership
 * der underlying texture; ra_metal kopiert nur den Pointer.
 */
struct ra_tex *ra_metal_wrap_external_texture(struct ra_ctx *ctx,
                                              id<MTLTexture> texture,
                                              const struct ra_format *fmt,
                                              int w, int h);

/**
 * Commit the current frame's command buffer. If `display_synced` is true,
 * mpv requested display synchronization (currently honoured as
 * `addScheduledHandler:` for timing; presentation is the host app's job).
 */
void ra_metal_commit_frame(struct ra_ctx *ctx, bool display_synced);

/**
 * Tear down the Metal ra. Safe to call multiple times.
 */
void ra_metal_destroy(struct ra_ctx *ctx);

/* Internal types -- exposed here so chunks of the implementation can share */

struct ra_metal {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    /* Per-frame command buffer. Allocated lazily on first encode call.
     * Released by ra_metal_commit_frame(). */
    id<MTLCommandBuffer> cmd_buf;
    /* Sampler state cache keyed on (linear, repeat) — 4 combos. */
    id<MTLSamplerState> samplers[4];
    /* Currently-open render encoder, if any. Tracked so debug markers can
     * route insertDebugSignpost: calls to the live encoder. nil between
     * encodes. */
    id<MTLRenderCommandEncoder> active_render_enc;
    /* mpv_global stashed at init time so renderpass_create can look up
     * spirv_conf via mp_get_config_group(). struct ra itself has no global
     * pointer; we mirror what ra_d3d11 does and cache it on priv. */
    struct mpv_global *global;
    /* ra_ctx pointer for accessing ctx->spirv (the actual initialized
     * SPIR-V compiler) in renderpass_create. spirv_compiler_init must be
     * called once before any pass creation; we do it in ra_init_metal. */
    struct ra_ctx *ctx;
    /* Tiny LRU of MTLBuffer staging buffers for tex_upload. Without this
     * we allocate a fresh ~2 MB buffer per YUV plane per frame, which
     * jetsam's our process at ~40s of playback. 8 entries covers the
     * common case (3 planes + headroom) and we evict largest-LRU.
     *
     * void* + CF-bridging because this struct is talloc-allocated (raw C
     * memory) and ARC won't manage NSMutableArray * fields in C structs:
     * the autoreleased array gets freed at function exit → NSException
     * on first [pool count] in tex_upload. Explicit __bridge_retained
     * keeps it alive; CFBridgingRelease in destroy frees. */
    void *staging_pool;
    /* Pipeline cache: NSMutableDictionary<NSString*, MetalPassCacheEntry*>
     * keyed by hash of (vertex_glsl + frag_glsl + target_format) or
     * (compute_glsl). Cache hit short-circuits the full shaderc + SPIRV-Cross
     * + MTLLibrary + MTLRenderPipelineState compile chain — saves ~150-300ms
     * per renderpass_create. Same CF-bridge trick as staging_pool. */
    void *pipeline_cache;
    /* MTLBinaryArchive für Cross-Session-PSO-Cache. pipeline_cache oben ist
     * per-Session; nach App-Kill rekompiliert mpv die 10-20 PSOs neu
     * (50-300ms pro Pass = ~1-3s "Schwarz" auf erstem Video-Load post-
     * Cold-Start). Archive wird beim init aus NSCachesDirectory geladen
     * (oder leer erstellt wenn Datei fehlt/inkompatibel) und beim destroy
     * via serializeToURL: zurückgeschrieben. Cache-Invalidation passiert
     * automatisch: load schlägt fehl wenn GPU-Family wechselt oder mpv-
     * Binary anders ist, dann fängt das archive leer an. CF-bridged wie
     * staging_pool/pipeline_cache. */
    void *binary_archive;
    /* PSO-Sidecar für Background-Warmup. binary_archive (oben) cacht den
     * final MTLPipelineState-Build (~50-300ms/pass), aber die shaderc +
     * SPIRV-Cross + MSL-Compile-Chain (~100-200ms/pass) muss eh laufen
     * weil newRenderPipelineStateWithDescriptor: die MTLFunction-Objekte
     * im Descriptor braucht. Mit einem persistierten descriptor-list
     * können wir die compile-chain im Hintergrund beim App-Launch
     * vorab ausführen, sodass die in-memory pipeline_cache schon warm
     * ist wenn der User das erste Video tappt. Sidecar = NSMutableArray
     * von NSDictionary-Entries (serialisierte ra_renderpass_params),
     * disk-persisted als plist in NSCachesDirectory. */
    /* Synchronisation für PSO-warmup-thread. dispatch_async im
     * metal_kick_pso_warmup captured nur einen raw ra-pointer; wenn
     * destroy fired bevor warmup fertig ist, hat der background-thread
     * dangling pointers auf alle ra_metal-fields → use-after-free
     * crashes mit zufällig-aliased MTL/CA-objects an der freigegebenen
     * heap-stelle. warmup_group trackt die laufende background-work,
     * warmup_cancel signalisiert dem worker früh aufzuhören. Destroy
     * setzt cancel, wartet auf group, dann erst Cleanup. */
    /* Lazily-built shader+PSO for format-converting blit (used by
     * metal_blit when src/dst formats differ — pause path goes through
     * this when mpv blits its OSD-composited frame to the drawable in a
     * different format). Library kept alive so the PSO stays valid.
     * Keyed by destination MTLPixelFormat (NSNumber → MetalBlitEntry,
     * declared in ra_metal_textures.m). void* + CF-bridging matches the
     * pipeline_cache pattern. */
    void *blit_cache;
    /* Per-frame vertex ring buffer. Allocated once at init (2 MiB), reset
     * each frame in ra_metal_commit_frame. Avoids the newBufferWithBytes
     * per-draw alloc in renderpass_run when vertex data exceeds the inline
     * setVertexBytes limit (kMetalInlineBytesLimit=4 KiB). Mpv's render
     * graph emits ~10-30 draws per frame with vertex blocks up to ~64 KiB
     * (OSD glyphs, full-screen quads). Fallback to one-shot alloc if the
     * draw doesn't fit the remaining ring capacity. */
    id<MTLBuffer> vertex_ring;
    size_t vertex_ring_capacity;
    size_t vertex_ring_offset;

    /* Pass-Graph-Tracking für Phase-1 tile-shading-feasibility. Pro frame
     * trackt das jedes Pass mit (is_compute, target_tex_ptr). Bei
     * commit_frame walken wir das ring + zählen same-target-adjacent-
     * pairs (= raster-passes mit identischem target = direkte
     * tile-shading-merge-Kandidaten). Ring-size 32 deckt selbst worst-
     * case (HDR + error-diffusion = ~11 passes) komfortabel. Falls
     * überschritten → einfach beim 32. abbrechen, der Win wäre eh marginal. */
    /* __unsafe_unretained: ARC darf hier nicht retain'en weil's ein C-array
     * in einer talloc-allocated C-struct ist. Wir halten den pointer nur
     * für die Dauer eines frames; lifetime managed vom caller (mpv core
     * + ra_tex). */
};

struct ra_tex_metal {
    id<MTLTexture> texture;
    /* Set when this tex wraps a drawable (no ownership of texture lifecycle). */
    bool is_wrapped;
};

struct ra_buf_metal {
    id<MTLBuffer> buffer;
    /* Tracks the command buffer that last used the buf for buf_poll. */
    id<MTLCommandBuffer> last_used_cb;
};

struct ra_renderpass_metal {
    id<MTLRenderPipelineState> render_pso;
    id<MTLComputePipelineState> compute_pso;
    id<MTLLibrary> library;
    /* Map from ra_renderpass_input index → Metal binding slot. */
    int *binding_indices;
    int num_inputs;
    /* For compute passes only: threads-per-threadgroup as declared by the
     * shader's `layout(local_size_x = ...)` qualifier. Populated by
     * metal_renderpass_create from SPIRV-Cross reflection. */
    MTLSize compute_threads_per_group;
    /* MSL buffer slot for the push_constant block, or -1 if the pass has no
     * push constants. Determined at renderpass_create via
     * spvc_compiler_msl_get_automatic_resource_binding for the push_const
     * SPIR-V resource. renderpass_run uses setVertexBytes/setFragmentBytes/
     * setBytes atIndex:push_constant_slot if push_constants_size > 0. */
    int push_constant_slot;
};

/* Helpers that may be shared between chunks. */

/* Returns the `ra_metal` priv from a `ra*`. */
static inline struct ra_metal *ra_metal_get(struct ra *ra)
{
    return (struct ra_metal *)ra->priv;
}

/* Returns or lazily allocates the per-frame command buffer.
 * Implemented in ra_metal_init.m. */
id<MTLCommandBuffer> ra_metal_cmd_buf(struct ra *ra);

/* Returns or lazily creates a cached sampler state. Implemented in init.m. */
id<MTLSamplerState> ra_metal_sampler(struct ra *ra, bool linear, bool repeat);

/* Pixel format mapping. Returns MTLPixelFormatInvalid on no match.
 * Implemented in init.m alongside the formats table. */
MTLPixelFormat ra_metal_pixel_format(const struct ra_format *fmt);

#endif

/* Copyright (C) 2026 the mpv developers
 *
 * Metal RA backend — textures, buffers, clear, blit.
 *
 * Parallel chunk to video/out/opengl/ra_gl.c (lines ~266-718), the
 * texture/buffer/clear/blit slice of the GL backend. The Metal port is
 * structurally the same: ra_tex / ra_buf wrappers around the native
 * MTLTexture / MTLBuffer, with the GL FBO+PBO machinery replaced by
 * MTLBlitCommandEncoder transfers and MTLRenderPassDescriptor loadActions.
 *
 * All functions here are NON-STATIC: they are referenced by name from
 * the master ra_fns vtable assembled in ra_metal.m. Keep the prototypes
 * matching struct ra_fns in video/out/gpu/ra.h exactly.
 *
 * Phase-1 status: the common path (whole-texture clear, same-size /
 * same-format blit, simple uploads/downloads) is real. Anything that
 * would need a transient render pass (scaled blit, scissored clear,
 * format conversion) is gated on phase 2 and grep-marked TODO(phase-2).
 */

#import "ra_metal.h"
#include "mpv_talloc.h"
#include "common/msg.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdatomic.h>


/* Cache entry for format-converting blit (one PSO per dst-pixelformat,
 * library kept alive so the PSO stays valid). Stored in
 * ra_metal->blit_cache (NSMutableDictionary keyed by NSNumber-wrapped
 * MTLPixelFormat). */
@interface MetalBlitEntry : NSObject
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTLRenderPipelineState> pso;
@end
@implementation MetalBlitEntry @end

/* MSL source for the blit shader. Fullscreen-triangle via vertex_id
 * (no vertex buffer needed). The vertex shader maps the triangle's
 * normalised 0..1 space into the src_rc-within-src normalised UV range
 * passed via a uniform — required because mpv blits sub-regions of the
 * source texture (e.g. video letterboxed inside a full-screen FBO),
 * NOT the whole texture. Without the transform the entire src gets
 * stretched into the dst-viewport, producing a squished image.
 *
 * Y is flipped in the texcoord because the triangle's NDC-Y up maps to
 * Metal's framebuffer-Y down, so sampling needs the inverse to keep
 * the image right-side up. */
static NSString *const kMetalBlitMSL =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct BlitParams { float2 uv_offset; float2 uv_scale; };\n"
    @"struct VOut { float4 pos [[position]]; float2 uv; };\n"
    @"vertex VOut blit_v(uint vid [[vertex_id]],\n"
    @"                   constant BlitParams& bp [[buffer(0)]]) {\n"
    @"  float2 t = float2((vid << 1u) & 2u, vid & 2u);\n"
    @"  VOut o;\n"
    @"  o.pos = float4(t * 2.0 - 1.0, 0.0, 1.0);\n"
    @"  o.uv = bp.uv_offset + float2(t.x, 1.0 - t.y) * bp.uv_scale;\n"
    @"  return o;\n"
    @"}\n"
    @"fragment float4 blit_f(VOut v [[stage_in]],\n"
    @"                       texture2d<float> tex [[texture(0)]],\n"
    @"                       sampler s [[sampler(0)]]) {\n"
    @"  return tex.sample(s, v.uv);\n"
    @"}\n";

/* Build (or fetch from cache) the blit PSO for the given destination
 * pixel format. Returns nil on compile/link failure (caller logs). */
static MetalBlitEntry *metal_blit_pso(struct ra *ra, MTLPixelFormat dst_fmt)
{
    struct ra_metal *m = ra_metal_get(ra);
    NSMutableDictionary<NSNumber *, MetalBlitEntry *> *cache =
        (__bridge NSMutableDictionary *)m->blit_cache;
    if (!cache) {
        cache = [NSMutableDictionary new];
        m->blit_cache = (__bridge_retained void *)cache;
    }
    NSNumber *key = @((NSUInteger)dst_fmt);
    MetalBlitEntry *entry = cache[key];
    if (entry) return entry;

    NSError *err = nil;
    MTLCompileOptions *copts = [MTLCompileOptions new];
    [copts setLanguageVersion:MTLLanguageVersion3_2];
    id<MTLLibrary> lib =
        [m->device newLibraryWithSource:kMetalBlitMSL options:copts error:&err];
    if (!lib) {
        MP_ERR(ra, "metal: blit library compile failed: %s\n",
               err.localizedDescription.UTF8String);
        return nil;
    }

    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction   = [lib newFunctionWithName:@"blit_v"];
    pd.fragmentFunction = [lib newFunctionWithName:@"blit_f"];
    pd.colorAttachments[0].pixelFormat = dst_fmt;
    id<MTLRenderPipelineState> pso =
        [m->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!pso) {
        MP_ERR(ra, "metal: blit PSO failed: %s\n",
               err.localizedDescription.UTF8String);
        return nil;
    }

    entry = [MetalBlitEntry new];
    entry.library = lib;
    entry.pso = pso;
    cache[key] = entry;
    return entry;
}

/* ------------------------------------------------------------------ */
/* Internal helpers (file-local, not part of the vtable)              */
/* ------------------------------------------------------------------ */

/*
 * Translate (dimensions, params) → MTLTextureType. Metal exposes 1D, 2D
 * and 3D as distinct types (plus array / cube / multisample variants we
 * don't use). This is the metal analogue of the GL switch in
 * gl_tex_create_blank (ra_gl.c:295).
 */
static MTLTextureType metal_texture_type(const struct ra_tex_params *params)
{
    switch (params->dimensions) {
    case 1: return MTLTextureType1D;
    case 2: return MTLTextureType2D;
    case 3: return MTLTextureType3D;
    default: return MTLTextureType2D; /* caller validated; defensive */
    }
}

/*
 * Compute MTLTextureUsage from ra_tex_params. ShaderRead covers
 * render_src (sampling in a fragment/compute shader); RenderTarget
 * covers anything that needs an MTLRenderPassDescriptor attachment
 * (render_dst, blit_src/blit_dst — Metal's blit encoder doesn't strictly
 * require RenderTarget, but mpv's BLIT path may fall back through a
 * render pass for scaled blits in phase 2, so request it up-front);
 * ShaderWrite covers RA_VARTYPE_IMG_W (storage_dst).
 */
static MTLTextureUsage metal_texture_usage(const struct ra_tex_params *params)
{
    MTLTextureUsage usage = 0;
    if (params->render_src)
        usage |= MTLTextureUsageShaderRead;
    if (params->render_dst || params->blit_src || params->blit_dst ||
        params->downloadable)
    {
        usage |= MTLTextureUsageRenderTarget;
    }
    if (params->storage_dst)
        usage |= MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    if (usage == 0)
        usage = MTLTextureUsageShaderRead; /* sensible default */
    return usage;
}

/*
 * Pick a storage mode. Private = GPU-only (fastest, the default).
 * Shared = CPU+GPU coherent — required when the caller wants to read
 * the texture back via tex_download(). host_mutable textures stay
 * Private; the upload path stages through a Shared MTLBuffer.
 */
static MTLStorageMode metal_texture_storage(const struct ra_tex_params *params)
{
    if (params->downloadable)
        return MTLStorageModeShared;
    /* 1D textures get uploaded via [MTLTexture replaceRegion:...] which
     * requires CPU access. Some Apple GPUs (A19 Pro/iPhone 17 Pro) tolerate
     * the call on Private storage via an internal copy, others (M1-class
     * iPad Air) SIGABRT in mpv_render_context_render with `_validate-
     * ReplaceRegion: CPU access for textures with MTLResourceStorageMode-
     * Private storage mode is disallowed`. 1D textures in mpv are tiny
     * LUTs — host-coherent storage has negligible cost and avoids the
     * iPad-side crash. */
    if (params->dimensions == 1)
        return MTLStorageModeShared;
    /* 2D textures with one dimension == 1 are LUT-like (e.g., scaler kernel
     * tables 1×256 rgba32f). These get uploaded via initial_data and Apple's
     * blit copyFromBuffer: silently drops the data for 1-pixel-wide rows on
     * iOS Private storage (verified via .60c readback diag). Use Shared so
     * replaceRegion:withBytes: works (same path as 1D textures). */
    if (params->dimensions == 2 && (params->w == 1 || params->h == 1))
        return MTLStorageModeShared;
    return MTLStorageModePrivate;
}

/*
 * Stage `data` (CPU memory) into `texture` via a transient Shared
 * MTLBuffer + blit encoder. Shared inside ra_metal_textures.m: used by
 * both initial_data (tex_create) and the buffer-less upload path
 * (tex_upload with params->src). The blit is issued on the per-frame
 * command buffer so it's serialized with everything else in the frame.
 */
static bool metal_stage_to_texture(struct ra *ra, id<MTLTexture> texture,
                                   const void *data, size_t bytes_per_row,
                                   MTLRegion region, int dimensions)
{
    struct ra_metal *p = ra_metal_get(ra);

    /* Apple iOS: copyFromBuffer:sourceBytesPerRow: silently fails when
     * sourceBytesPerRow is smaller than minimumLinearTextureAlignmentForPixelFormat.
     * For rgba32f (= scaler LUT) at width=1, sourceBytesPerRow=16 violates
     * the 256-byte alignment minimum → blit lands all-zero → scaler weights
     * are zero → U+V output is zero → solid green.
     *
     * Fix: query the per-pixel-format minimum alignment, pad sourceBytesPerRow
     * to that and stride-copy the source data into the padded staging buffer.
     * (Texture row width <= padded stride; blit picks up the right pixels.)
     */
    NSUInteger min_align = 1;
    if (@available(iOS 13.0, *)) {
        min_align = [p->device minimumLinearTextureAlignmentForPixelFormat:texture.pixelFormat];
        if (min_align == 0) min_align = 1;
    }
    size_t padded_bpr = (bytes_per_row + min_align - 1) & ~(min_align - 1);
    bool pad_rows = (padded_bpr != bytes_per_row) && (region.size.height > 1 || dimensions == 3);
    size_t height_for_total = (dimensions == 3 ? region.size.depth : region.size.height);
    size_t total = (pad_rows ? padded_bpr : bytes_per_row) * height_for_total;
    if (dimensions == 3 && pad_rows)
        total = padded_bpr * region.size.height * region.size.depth;

    /* Try to reuse an MTLBuffer from the pool that's at least `total` big.
     * Without pooling, every YUV-plane upload (3x per frame on 1080p
     * playback) allocates a fresh ~2 MB Shared MTLBuffer, which iOS
     * jetsam's the process after ~40s. */
    NSMutableArray *pool = (__bridge NSMutableArray *)p->staging_pool;
    id<MTLBuffer> staging = nil;
    NSUInteger pick_idx = NSNotFound;
    @synchronized (pool) {
        for (NSUInteger i = 0; i < pool.count; i++) {
            id<MTLBuffer> b = pool[i];
            if (b.length >= total) {
                staging = b;
                pick_idx = i;
                break;
            }
        }
        if (staging)
            [pool removeObjectAtIndex:pick_idx];
    }
    if (!staging) {
        staging = [p->device newBufferWithLength:total
                                        options:MTLResourceStorageModeShared];
        if (!staging) {
            MP_ERR(ra, "metal: failed to allocate %zu-byte staging buffer\n", total);
            return false;
        }
    }
    /* Copy source data into staging — if we're padding, stride-copy row by row. */
    if (pad_rows) {
        uint8_t *dst = (uint8_t *)staging.contents;
        const uint8_t *src = (const uint8_t *)data;
        size_t rows = (dimensions == 3) ? (region.size.height * region.size.depth)
                                        : region.size.height;
        for (size_t r = 0; r < rows; r++)
            memcpy(dst + r * padded_bpr, src + r * bytes_per_row, bytes_per_row);
    } else {
        memcpy(staging.contents, data,
               bytes_per_row * region.size.height *
               (dimensions == 3 ? region.size.depth : 1));
    }
    size_t blit_bpr = pad_rows ? padded_bpr : bytes_per_row;

    id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromBuffer:staging
           sourceOffset:0
      sourceBytesPerRow:blit_bpr
    sourceBytesPerImage:(dimensions == 3 ? blit_bpr * region.size.height : 0)
             sourceSize:region.size
              toTexture:texture
       destinationSlice:0
       destinationLevel:0
      destinationOrigin:region.origin];
    [enc endEncoding];


    /* Return to pool once the GPU is done reading. Capture `pool` and
     * `staging` into the block (keeps them alive across the GPU work). Cap
     * the pool at 8 entries to bound memory. */
    [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buf) {
        @synchronized (pool) {
            if (pool.count < 8)
                [pool addObject:staging];
        }
    }];
    return true;
}

/* ------------------------------------------------------------------ */
/* tex_destroy / tex_create                                           */
/* ------------------------------------------------------------------ */

void metal_tex_destroy(struct ra *ra, struct ra_tex *tex)
{
    (void)ra;
    if (!tex)
        return;
    struct ra_tex_metal *tex_metal = tex->priv;
    if (tex_metal) {
        /*
         * Wrapped textures (CAMetalDrawable, hwdec surfaces) are owned
         * by the caller — drop the strong ref but don't otherwise touch
         * the underlying object. For owned textures, nilling the strong
         * ref under ARC is what actually triggers the release.
         */
        if (!tex_metal->is_wrapped)
            tex_metal->texture = nil;
        else
            tex_metal->texture = nil; /* drop our ref either way */
    }
    talloc_free(tex_metal);
    talloc_free(tex);
}

struct ra_tex *metal_tex_create(struct ra *ra,
                                const struct ra_tex_params *params)
{
    struct ra_metal *p = ra_metal_get(ra);
    mp_assert(!params->format->dummy_format);

    /*
     * Edge case rejection mirroring gl_tex_create_blank's check at
     * ra_gl.c:310: downloadable requires a 2D renderable texture.
     * Additionally reject 1D textures used as compute storage targets —
     * Metal supports it in principle but our binding layout assumes 2D
     * for image-writable slots.
     */
    if (params->downloadable && !(params->dimensions == 2 &&
                                  params->format->renderable))
    {
        MP_ERR(ra, "metal: downloadable texture must be 2D + renderable\n");
        return NULL;
    }
    if (params->dimensions == 1 && params->storage_dst &&
        params->format->ctype != RA_CTYPE_UNORM &&
        params->format->ctype != RA_CTYPE_FLOAT)
    {
        MP_ERR(ra, "metal: 1D storage textures must be RGB-like\n");
        return NULL;
    }

    MTLPixelFormat pf = ra_metal_pixel_format(params->format);
    if (pf == MTLPixelFormatInvalid) {
        MP_ERR(ra, "metal: no MTLPixelFormat for ra_format '%s'\n",
               params->format->name ? params->format->name : "?");
        return NULL;
    }

    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = metal_texture_type(params);
    desc.pixelFormat = pf;
    desc.width       = params->w;
    desc.height      = (params->dimensions >= 2) ? params->h : 1;
    desc.depth       = (params->dimensions == 3) ? params->d : 1;
    desc.mipmapLevelCount = 1;
    desc.sampleCount      = 1;
    desc.arrayLength      = 1;
    desc.usage       = metal_texture_usage(params);
    desc.storageMode = metal_texture_storage(params);

    id<MTLTexture> mtl_tex = [p->device newTextureWithDescriptor:desc];
    if (!mtl_tex) {
        MP_ERR(ra,
               "metal: newTextureWithDescriptor failed "
               "(%dx%dx%d fmt=%s usage=0x%lx)\n",
               (int)desc.width, (int)desc.height, (int)desc.depth,
               params->format->name ? params->format->name : "?",
               (unsigned long)desc.usage);
        return NULL;
    }

    struct ra_tex *tex = talloc_zero(NULL, struct ra_tex);
    tex->params = *params;
    tex->params.initial_data = NULL; /* don't expose stale pointer */

    struct ra_tex_metal *tex_metal = talloc_zero(NULL, struct ra_tex_metal);
    tex_metal->texture    = mtl_tex;
    tex_metal->is_wrapped = false;
    tex->priv = tex_metal;

    if (params->initial_data) {
        MTLRegion region = MTLRegionMake3D(0, 0, 0,
                                           params->w,
                                           (params->dimensions >= 2) ? params->h : 1,
                                           (params->dimensions == 3) ? params->d : 1);
        size_t bpr = (size_t)params->format->pixel_size * (size_t)params->w;
        /* For Shared-storage textures (1D + LUT-shaped 2D), use replaceRegion
         * directly — bypass the blit path which silently drops upload data
         * on iOS for 1-pixel-wide rgba32f Private textures. */
        bool use_replace = (params->dimensions == 1) ||
                           (params->dimensions == 2 &&
                            mtl_tex.storageMode == MTLStorageModeShared);
        if (use_replace) {
            [mtl_tex replaceRegion:region
                       mipmapLevel:0
                         withBytes:params->initial_data
                       bytesPerRow:bpr];
        } else if (!metal_stage_to_texture(ra, mtl_tex, params->initial_data,
                                           bpr, region, params->dimensions))
        {
            metal_tex_destroy(ra, tex);
            return NULL;
        }
    }

    return tex;
}

/* ------------------------------------------------------------------ */
/* tex_upload / tex_download                                          */
/* ------------------------------------------------------------------ */

bool metal_tex_upload(struct ra *ra, const struct ra_tex_upload_params *params)
{
    struct ra_tex *tex = params->tex;
    struct ra_tex_metal *tex_metal = tex->priv;
    mp_assert(tex->params.host_mutable);
    mp_assert(!params->buf || !params->src);


    /* TODO(phase-2): honour params->invalidate via
     *   MTLRenderPassDescriptor loadAction=DontCare on the next pass.
     *   The Metal blit encoder has no equivalent of glInvalidateTexImage,
     *   so we'd have to defer the hint to the next render pass that
     *   targets `tex`. For now we always preserve existing contents. */

    struct mp_rect rc = {0, 0, tex->params.w, tex->params.h};
    if (params->rc && tex->params.dimensions == 2)
        rc = *params->rc;
    int rc_w = (tex->params.dimensions >= 1) ? (rc.x1 - rc.x0) : 0;
    int rc_h = (tex->params.dimensions >= 2) ? (rc.y1 - rc.y0) : 1;
    int rc_d = (tex->params.dimensions == 3) ? tex->params.d : 1;
    MTLRegion region = MTLRegionMake3D(rc.x0,
                                       (tex->params.dimensions >= 2) ? rc.y0 : 0,
                                       0,
                                       rc_w, rc_h, rc_d);
    size_t bpr = params->stride ?
                 (size_t)params->stride :
                 (size_t)tex->params.format->pixel_size * (size_t)rc_w;

    /* 1D fast path: Metal's replaceRegion works on host-visible storage,
     * but it also works on Private storage via an internal copy. We
     * always go through it — 1D textures are tiny (LUTs, primarily) and
     * the kernel-side staging is cheaper than spinning up a blit pass. */
    if (tex->params.dimensions == 1) {
        const void *src = NULL;
        if (params->buf) {
            struct ra_buf_metal *buf_metal = params->buf->priv;
            src = (const uint8_t *)[buf_metal->buffer contents] +
                  params->buf_offset;
        } else {
            src = params->src;
        }
        [tex_metal->texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:src
                              bytesPerRow:bpr];
        return true;
    }

    /* Buffer-sourced upload: blit-copy from the caller's MTLBuffer. */
    if (params->buf) {
        struct ra_buf_metal *buf_metal = params->buf->priv;
        id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc copyFromBuffer:buf_metal->buffer
               sourceOffset:params->buf_offset
          sourceBytesPerRow:bpr
        sourceBytesPerImage:(tex->params.dimensions == 3 ? bpr * rc_h : 0)
                 sourceSize:region.size
                  toTexture:tex_metal->texture
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:region.origin];
        [enc endEncoding];

        /* Pin the buffer to this command buffer so buf_poll reports it
         * in-use until the GPU is done reading from it. */
        buf_metal->last_used_cb = cb;
        return true;
    }

    /* Direct upload from CPU memory: stage through a transient Shared
     * MTLBuffer. Same code path used for initial_data. */
    return metal_stage_to_texture(ra, tex_metal->texture, params->src,
                                  bpr, region, tex->params.dimensions);
}

bool metal_tex_download(struct ra *ra, struct ra_tex_download_params *params)
{
    struct ra_metal *p = ra_metal_get(ra);
    struct ra_tex *tex = params->tex;
    struct ra_tex_metal *tex_metal = tex->priv;

    if (tex->params.dimensions != 2) {
        MP_ERR(ra, "metal: tex_download only supports 2D textures\n");
        return false;
    }
    if (!tex->params.render_dst) {
        MP_ERR(ra, "metal: tex_download requires render_dst=true\n");
        return false;
    }

    size_t bpr_packed = (size_t)tex->params.format->pixel_size *
                        (size_t)tex->params.w;
    size_t total = bpr_packed * (size_t)tex->params.h;

    id<MTLBuffer> staging =
        [p->device newBufferWithLength:total
                               options:MTLResourceStorageModeShared];
    if (!staging) {
        MP_ERR(ra, "metal: tex_download failed to allocate %zu-byte staging\n",
               total);
        return false;
    }

    id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromTexture:tex_metal->texture
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(tex->params.w, tex->params.h, 1)
                toBuffer:staging
       destinationOffset:0
  destinationBytesPerRow:bpr_packed
destinationBytesPerImage:total];
    [enc endEncoding];

    /*
     * tex_download is synchronous from the caller's POV — commit the
     * frame's command buffer and block until the GPU has resolved.
     * ra_metal_commit_frame() will re-allocate the cmd_buf on next use.
     */
    [cb commit];
    [cb waitUntilCompleted];
    p->cmd_buf = nil;

    const uint8_t *src = (const uint8_t *)[staging contents];
    uint8_t *dst = params->dst;
    size_t dst_stride = params->stride ? (size_t)params->stride : bpr_packed;
    for (int y = 0; y < tex->params.h; y++) {
        memcpy(dst + (size_t)y * dst_stride,
               src + (size_t)y * bpr_packed,
               bpr_packed);
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* buf_destroy / buf_create / buf_update / buf_poll                   */
/* ------------------------------------------------------------------ */

void metal_buf_destroy(struct ra *ra, struct ra_buf *buf)
{
    (void)ra;
    if (!buf)
        return;
    struct ra_buf_metal *buf_metal = buf->priv;
    if (buf_metal) {
        /* ARC: nil the strong refs to release. */
        buf_metal->buffer       = nil;
        buf_metal->last_used_cb = nil;
    }
    talloc_free(buf_metal);
    talloc_free(buf);
}

struct ra_buf *metal_buf_create(struct ra *ra,
                                const struct ra_buf_params *params)
{
    struct ra_metal *p = ra_metal_get(ra);

    /*
     * Pick storage mode. The GL backend uses GL_STREAM_DRAW / STATIC_DRAW
     * hints (ra_gl.c:631) which the driver maps to either host-visible
     * or device-local memory. Metal makes us choose explicitly:
     *
     *   - host_mapped       → Shared (caller wants buf->data alive)
     *   - UNIFORM, TEX_UPLOAD → Shared (small, written every frame)
     *   - SHADER_STORAGE    → Private (GPU-resident SSBO, unless mapped)
     */
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    bool private_storage = false;
    if (params->type == RA_BUF_TYPE_SHADER_STORAGE && !params->host_mapped) {
        opts = MTLResourceStorageModePrivate;
        private_storage = true;
    }

    id<MTLBuffer> mtl_buf;
    if (params->initial_data && !private_storage) {
        mtl_buf = [p->device newBufferWithBytes:params->initial_data
                                         length:params->size
                                        options:opts];
    } else {
        mtl_buf = [p->device newBufferWithLength:params->size options:opts];
    }
    if (!mtl_buf) {
        MP_ERR(ra, "metal: newBuffer failed (size=%zu)\n", params->size);
        return NULL;
    }

    struct ra_buf *buf = talloc_zero(NULL, struct ra_buf);
    buf->params = *params;
    buf->params.initial_data = NULL;

    struct ra_buf_metal *buf_metal = talloc_zero(NULL, struct ra_buf_metal);
    buf_metal->buffer = mtl_buf;
    buf->priv = buf_metal;

    if (params->host_mapped)
        buf->data = [mtl_buf contents];

    /* Private + initial_data: stage through a Shared buffer and blit. */
    if (params->initial_data && private_storage) {
        id<MTLBuffer> staging =
            [p->device newBufferWithBytes:params->initial_data
                                   length:params->size
                                  options:MTLResourceStorageModeShared];
        if (!staging) {
            MP_ERR(ra, "metal: failed to stage initial_data into private buf\n");
            metal_buf_destroy(ra, buf);
            return NULL;
        }
        id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc copyFromBuffer:staging
               sourceOffset:0
                   toBuffer:mtl_buf
          destinationOffset:0
                       size:params->size];
        [enc endEncoding];
        buf_metal->last_used_cb = cb;
    }

    return buf;
}

void metal_buf_update(struct ra *ra, struct ra_buf *buf, ptrdiff_t offset,
                      const void *data, size_t size)
{
    struct ra_metal *p = ra_metal_get(ra);
    struct ra_buf_metal *buf_metal = buf->priv;
    mp_assert(buf->params.host_mutable);
    mp_assert((offset & 3) == 0); /* spec: offset is a multiple of 4 */

    /*
     * Shared storage: direct memcpy. Metal's Shared mode is coherent on
     * Apple silicon, so the GPU sees the write the next time we encode
     * against the buffer. Equivalent to glBufferSubData for the
     * persistently-mapped case.
     */
    if (buf_metal->buffer.storageMode == MTLStorageModeShared) {
        memcpy((uint8_t *)[buf_metal->buffer contents] + offset, data, size);
        return;
    }

    /*
     * Private storage: build a transient Shared staging buffer and
     * blit-copy onto the device-local target. Same shape as the
     * initial_data path in metal_buf_create but parameterised by offset.
     */
    id<MTLBuffer> staging =
        [p->device newBufferWithBytes:data
                               length:size
                              options:MTLResourceStorageModeShared];
    if (!staging) {
        MP_ERR(ra, "metal: buf_update failed to allocate staging\n");
        return;
    }
    id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromBuffer:staging
           sourceOffset:0
               toBuffer:buf_metal->buffer
      destinationOffset:offset
                   size:size];
    [enc endEncoding];
    buf_metal->last_used_cb = cb;
}

bool metal_buf_poll(struct ra *ra, struct ra_buf *buf)
{
    (void)ra;
    struct ra_buf_metal *buf_metal = buf->priv;

    /* Never been touched by the GPU → trivially reusable. Matches the
     * gl_buf_poll early-out for un-fenced buffers (ra_gl.c:661). */
    if (!buf_metal->last_used_cb)
        return true;

    MTLCommandBufferStatus status = buf_metal->last_used_cb.status;
    if (status >= MTLCommandBufferStatusCompleted) {
        /* Drop the ref so subsequent polls hit the fast path above. */
        buf_metal->last_used_cb = nil;
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* clear / blit                                                       */
/* ------------------------------------------------------------------ */

void metal_clear(struct ra *ra, struct ra_tex *dst, float color[4],
                 struct mp_rect *scissor)
{
    struct ra_metal *p = ra_metal_get(ra);
    mp_assert(dst->params.render_dst);
    struct ra_tex_metal *tex_metal = dst->priv;

    bool full_rect = !scissor ||
                     (scissor->x0 == 0 && scissor->y0 == 0 &&
                      scissor->x1 == dst->params.w &&
                      scissor->y1 == dst->params.h);

    if (!full_rect) {
        /* TODO(phase-2): scissored clear. Metal's loadAction=.clear
         * always covers the full attachment; a sub-rect clear requires
         * either a tiny render pass with a colored quad + scissor rect,
         * or a compute kernel that writes the color over the region.
         * The GL backend uses GL_SCISSOR_TEST around glClear
         * (ra_gl.c:687-695); the Metal equivalent needs the render-pass
         * code that's gated on phase-2 shader work. */
        MP_FATAL(ra,
            "metal: scissored clear not implemented "
            "(TODO(phase-2): needs render-pass quad + scissor)\n");
        return;
    }

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = tex_metal->texture;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor  =
        MTLClearColorMake(color[0], color[1], color[2], color[3]);

    id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
    id<MTLRenderCommandEncoder> enc =
        [cb renderCommandEncoderWithDescriptor:rpd];
    /* No draw calls — endEncoding commits the loadAction, which is the
     * clear. Metal optimises an "empty" render pass into a pure clear. */
    [enc endEncoding];
    (void)p;
}

void metal_blit(struct ra *ra, struct ra_tex *dst, struct ra_tex *src,
                struct mp_rect *dst_rc, struct mp_rect *src_rc)
{
    mp_assert(src->params.blit_src);
    mp_assert(dst->params.blit_dst);

    struct ra_tex_metal *src_metal = src->priv;
    struct ra_tex_metal *dst_metal = dst->priv;

    int dst_w = dst_rc->x1 - dst_rc->x0;
    int dst_h = dst_rc->y1 - dst_rc->y0;
    int src_w = src_rc->x1 - src_rc->x0;
    int src_h = src_rc->y1 - src_rc->y0;

    /* Detect flipped blit (negative dst dimensions). mpv's pause-redraw
     * path in gl_video.c swaps dst.y0/y1 when fbo->flip is true (which
     * we set via MPV_RENDER_PARAM_FLIP_Y=1 in Kuckuck) — that arrives
     * here as dst_h < 0. Normalize to positive rect + remember flip
     * flags; the render-pass branch below uses them to mirror via the
     * uv_scale uniform. Pre-flip-fix path FATAL'd on the "scaled blit
     * not implemented" check below because |dst_h| == src_h passed the
     * abs check but the signed comparison didn't. */
    bool flip_x = dst_w < 0;
    bool flip_y = dst_h < 0;
    if (flip_x) dst_w = -dst_w;
    if (flip_y) dst_h = -dst_h;
    int dst_x0 = flip_x ? dst_rc->x1 : dst_rc->x0;
    int dst_y0 = flip_y ? dst_rc->y1 : dst_rc->y0;

    bool same_size   = (dst_w == src_w) && (dst_h == src_h);
    bool same_format = (src->params.format == dst->params.format) ||
                       (ra_metal_pixel_format(src->params.format) ==
                        ra_metal_pixel_format(dst->params.format));
    bool flipped = flip_x || flip_y;

    if (!same_size) {
        /* TODO: scaled blit (different rect sizes). Would need
         * uv-scaling uniforms in the blit shader. Not hit by mpv's
         * pause-redraw path so deferred. */
        MP_FATAL(ra,
            "metal: scaled blit not implemented "
            "(src=%dx%d dst=%dx%d)\n",
            src_w, src_h, dst_w, dst_h);
        return;
    }

    if (!same_format || flipped) {
        /* Format-converting blit (same size, different pixel format).
         * mpv hits this on every pause-redraw: the OSD-composited frame
         * lives in an internal RGBA16F-ish FBO, the drawable is
         * BGRA8Unorm — copyFromTexture can't bridge that. Drive a
         * one-pass render that samples src → dst via a fullscreen
         * triangle. PSO is built lazily per dst-pixelformat. */
        MTLPixelFormat dst_fmt = ra_metal_pixel_format(dst->params.format);
        MetalBlitEntry *entry = metal_blit_pso(ra, dst_fmt);
        if (!entry) return;  /* error already logged */

        /* Convert src_rc (pixel coords within the src texture) into the
         * normalized 0..1 UV range that the shader expects. mpv blits
         * sub-regions (e.g. video at y=972..1650 inside a 2622-tall
         * FBO); without this transform we'd sample the whole src and
         * stretch it into the dst-viewport — squished image. */
        struct {
            float uv_offset[2];
            float uv_scale[2];
        } bp = {
            .uv_offset = {
                (float)src_rc->x0 / (float)src->params.w,
                (float)src_rc->y0 / (float)src->params.h,
            },
            .uv_scale = {
                (float)src_w / (float)src->params.w,
                (float)src_h / (float)src->params.h,
            },
        };
        /* Flip: shift uv-origin to the far edge then negate the span.
         * In the shader: uv = uv_offset + frag_uv * uv_scale. Plugging in
         * fragment.uv ∈ {0..1}, we want fragment.uv=0 to sample src.high
         * and fragment.uv=1 to sample src.low when flipped. */
        if (flip_x) {
            bp.uv_offset[0] += bp.uv_scale[0];
            bp.uv_scale[0]   = -bp.uv_scale[0];
        }
        if (flip_y) {
            bp.uv_offset[1] += bp.uv_scale[1];
            bp.uv_scale[1]   = -bp.uv_scale[1];
        }

        id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
        MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor new];
        rpd.colorAttachments[0].texture = dst_metal->texture;
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc =
            [cb renderCommandEncoderWithDescriptor:rpd];
        [enc setRenderPipelineState:entry.pso];
        MTLViewport vp = {
            .originX = dst_x0,
            .originY = dst_y0,
            .width   = dst_w,
            .height  = dst_h,
            .znear = 0, .zfar = 1,
        };
        [enc setViewport:vp];
        [enc setVertexBytes:&bp length:sizeof(bp) atIndex:0];
        [enc setFragmentTexture:src_metal->texture atIndex:0];
        [enc setFragmentSamplerState:
                ra_metal_sampler(ra, src->params.src_linear, false)
                             atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0 vertexCount:3];
        [enc endEncoding];
        return;
    }

    /* Fast path: same size, same format → MTLBlitCommandEncoder
     * copyFromTexture. Negative widths/heights (the GL backend's flip
     * trick) aren't representable in the Metal blit encoder; if a flip
     * is needed it has to go through the phase-2 render-pass blit too. */
    if (dst_w < 0 || dst_h < 0 || src_w < 0 || src_h < 0) {
        MP_FATAL(ra,
            "metal: flipping blit not implemented "
            "(TODO(phase-2): needs render-pass blit shader)\n");
        return;
    }

    id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromTexture:src_metal->texture
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(src_rc->x0, src_rc->y0, 0)
              sourceSize:MTLSizeMake(src_w, src_h, 1)
               toTexture:dst_metal->texture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:MTLOriginMake(dst_rc->x0, dst_rc->y0, 0)];
    [enc endEncoding];
}

/* Copyright (C) 2026 the mpv developers
 *
 * Metal RA backend -- init, lifecycle, and pixel formats.
 *
 * Parallel chunk to video/out/opengl/ra_gl.c lines 1-310 (init + format
 * table) and the ra_fns_gl table near line 1186. This file owns:
 *
 *   - the static metal_formats[] table and ra_metal_pixel_format()
 *   - ra_init_metal() / ra_metal_destroy() lifecycle entry points
 *   - the per-frame command-buffer accessor ra_metal_cmd_buf()
 *   - the 4-entry sampler-state cache ra_metal_sampler()
 *   - ra_metal_wrap_drawable() / ra_metal_commit_frame() bridge helpers
 *   - the ra_fns "destroy" slot (ra_metal_destroy_fns) consumed by the
 *     master ra_fns_metal table in ra_metal.m.
 *
 * Helper functions (formats table, sampler descriptors) are file-local;
 * everything declared in ra_metal.h is exported with external linkage.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import "ra_metal.h"
#include "mpv_talloc.h"
#include "common/msg.h"
#include "video/out/gpu/spirv.h"
#include <stdatomic.h>

/* MTLBinaryArchive Cross-Session-Cache für kompilierte PSOs. Pfad in
 * NSCachesDirectory damit's iOS-side automatisch purgable ist unter
 * disk-pressure, aber normalerweise stabil zwischen App-Launches.
 * Filename hat keine Version — Cache-Invalidation kommt automatisch
 * über MTLBinaryArchive's Load-Validation: load schlägt fehl wenn
 * GPU-Family oder mpv-binary anders ist, dann fängt empty an. */
NSString *metal_binary_archive_path(void)
{
    NSArray *paths = NSSearchPathForDirectoriesInDomains(
        NSCachesDirectory, NSUserDomainMask, YES);
    if (!paths.count) return nil;
    return [paths[0] stringByAppendingPathComponent:@"MPVKit-PSO.bin"];
}

id<MTLBinaryArchive> metal_load_or_create_binary_archive(id<MTLDevice> device)
{
    if (@available(iOS 14.0, macOS 11.0, tvOS 14.0, *)) {
        MTLBinaryArchiveDescriptor *desc = [MTLBinaryArchiveDescriptor new];
        NSString *path = metal_binary_archive_path();
        if (path && [[NSFileManager defaultManager] fileExistsAtPath:path]) {
            desc.url = [NSURL fileURLWithPath:path];
        }
        NSError *err = nil;
        id<MTLBinaryArchive> a = [device newBinaryArchiveWithDescriptor:desc
                                                                  error:&err];
        /* Load fehlgeschlagen (z.B. anderes Device, mpv-bump): leer starten. */
        if (!a && desc.url) {
            desc.url = nil;
            a = [device newBinaryArchiveWithDescriptor:desc error:nil];
        }
        return a;
    }
    return nil;
}

void metal_serialize_binary_archive(id<MTLBinaryArchive> archive)
{
    if (!archive) return;
    if (@available(iOS 14.0, macOS 11.0, tvOS 14.0, *)) {
        NSString *path = metal_binary_archive_path();
        if (!path) return;
        NSURL *url = [NSURL fileURLWithPath:path];
        NSError *err = nil;
        [archive serializeToURL:url error:&err];
        /* best-effort — falls write fehlschlägt (disk full etc) rebuilden
         * wir beim nächsten Start halt die PSOs neu, kein Disaster. */
    }
}

/* PSO-Sidecar Cross-Session Persistenz. Speichert serialisierte
 * ra_renderpass_params zu jedem cache-Entry damit ein Background-Thread
 * beim nächsten App-Launch die compile-chain (shaderc + SPIRV-Cross +
 * Metal lib+pso compile) vorab ausführen kann.
 *
 * Architektur: NSMutableArray<NSDictionary*> in ra_metal->pso_sidecar.
 * Bei jedem PSO-Cache-Insert in renderpass_create wird ein NSDictionary
 * mit den params angehängt. Bei ra_metal_destroy wird das Array via
 * NSPropertyListSerialization als binary plist gespeichert. Beim
 * nächsten Launch lädt ra_init_metal die plist und kickt einen
 * Background-Thread an der pro Entry renderpass_create aufruft (das
 * füllt pipeline_cache und MTLBinaryArchive). */



/* Forward-declared in ra_metal.h: das warmup-replay ruft renderpass_create
 * (definiert in ra_metal_renderpass.m) für jeden sidecar-entry auf. Wir
 * können die struct ra_renderpass_params hier nicht direkt einbinden ohne
 * gpu/ra.h erneut zu inkludieren — die Logik sitzt komplett im
 * renderpass-file via metal_warmup_replay_one(). */


/* Defined by the master assembled file ra_metal.m. Declared here so
 * ra_init_metal() can wire `ra->fns = &ra_fns_metal` without pulling in
 * the renderpass/tex chunks at compile time. */
extern const struct ra_fns ra_fns_metal;

/* The ra_fns destroy slot. Distinct from ra_metal_destroy(), which is
 * the public lifecycle teardown that takes a `ra_ctx *`. Defined below. */
void ra_metal_destroy_fns(struct ra *ra);

/* ------------------------------------------------------------------ */
/* Pixel format table                                                 */
/* ------------------------------------------------------------------ */

/*
 * Mirrors gl_formats[] in video/out/opengl/formats.c. Each row is the
 * mpv-side `ra_format` description plus the Apple-side `MTLPixelFormat`
 * that backs it. Order matters: ra_find_unorm_format() and friends pick
 * the first match, so the preferred upload paths come first per type
 * width (this also matches ra_gl.c's ordering choices -- rg8 ahead of
 * any luminance-alpha fallback, etc.).
 *
 * `priv` on the resulting ra_format points at the table entry so
 * downstream (tex_create, renderpass_create) can recover the
 * MTLPixelFormat in O(1) without re-walking the table.
 */
struct metal_format {
    const char       *name;
    MTLPixelFormat    mtl_format;
    enum ra_ctype     ctype;
    int               num_components;
    int               component_size;   /* bits per component as stored */
    int               component_depth;  /* bits actually in use (== size unless padded) */
    int               pixel_size;       /* bytes */
    bool              linear_filter;
    bool              renderable;
};

static const struct metal_format metal_formats[] = {
    /* 8-bit UNORM */
    { "r8",       MTLPixelFormatR8Unorm,       RA_CTYPE_UNORM, 1,  8,  8,  1, true,  true  },
    { "rg8",      MTLPixelFormatRG8Unorm,      RA_CTYPE_UNORM, 2,  8,  8,  2, true,  true  },
    { "rgba8",    MTLPixelFormatRGBA8Unorm,    RA_CTYPE_UNORM, 4,  8,  8,  4, true,  true  },
    { "bgra8",    MTLPixelFormatBGRA8Unorm,    RA_CTYPE_UNORM, 4,  8,  8,  4, true,  true  },

    /* 16-bit UNORM. Note Apple-silicon GPUs require macOS 11+/iOS 14+
     * for these; older Intel Macs lack rgba16-unorm filtering, but the
     * filtering capability is a runtime check on MTLDevice and not
     * worth gating at table-build time. */
    { "r16",      MTLPixelFormatR16Unorm,      RA_CTYPE_UNORM, 1, 16, 16,  2, true,  true  },
    { "rg16",     MTLPixelFormatRG16Unorm,     RA_CTYPE_UNORM, 2, 16, 16,  4, true,  true  },
    { "rgba16",   MTLPixelFormatRGBA16Unorm,   RA_CTYPE_UNORM, 4, 16, 16,  8, true,  true  },

    /* 16-bit float */
    { "r16f",     MTLPixelFormatR16Float,      RA_CTYPE_FLOAT, 1, 16, 16,  2, true,  true  },
    { "rg16f",    MTLPixelFormatRG16Float,     RA_CTYPE_FLOAT, 2, 16, 16,  4, true,  true  },
    { "rgba16f",  MTLPixelFormatRGBA16Float,   RA_CTYPE_FLOAT, 4, 16, 16,  8, true,  true  },

    /* 32-bit float. Per Apple docs, only rgba32float is renderable on
     * every GPU family; r/rg32float are renderable on macOS, iOS-A11+. */
    { "r32f",     MTLPixelFormatR32Float,      RA_CTYPE_FLOAT, 1, 32, 32,  4, true,  true  },
    { "rg32f",    MTLPixelFormatRG32Float,     RA_CTYPE_FLOAT, 2, 32, 32,  8, true,  true  },
    { "rgba32f",  MTLPixelFormatRGBA32Float,   RA_CTYPE_FLOAT, 4, 32, 32, 16, true,  true  },

    /* Packed UNORM. Metal has no MTLPixelFormatRGB10A2Unorm with the
     * memory layout mpv expects (R in LSB); BGR10A2Unorm matches the
     * "rgb10_a2" interpretation in the shader-side swizzle path. */
    { "rgb10_a2", MTLPixelFormatBGR10A2Unorm,  RA_CTYPE_UNORM, 4, 10, 10,  4, true,  true  },
    { "rgb565",   MTLPixelFormatB5G6R5Unorm,   RA_CTYPE_UNORM, 3,  6,  5,  2, true,  true  },

    /* 8-bit unsigned integer (non-filterable -- linear_filter=false). */
    { "r8ui",     MTLPixelFormatR8Uint,        RA_CTYPE_UINT,  1,  8,  8,  1, false, true  },
    { "rg8ui",    MTLPixelFormatRG8Uint,       RA_CTYPE_UINT,  2,  8,  8,  2, false, true  },
    { "rgba8ui",  MTLPixelFormatRGBA8Uint,     RA_CTYPE_UINT,  4,  8,  8,  4, false, true  },
};

#define METAL_NUM_FORMATS (sizeof(metal_formats) / sizeof(metal_formats[0]))

MTLPixelFormat ra_metal_pixel_format(const struct ra_format *fmt)
{
    if (!fmt || !fmt->priv)
        return MTLPixelFormatInvalid;
    const struct metal_format *mf = fmt->priv;
    return mf->mtl_format;
}

/* ------------------------------------------------------------------ */
/* Sampler cache                                                      */
/* ------------------------------------------------------------------ */

/*
 * mpv only ever needs four sampler configurations (linear/nearest x
 * clamp/repeat), so we cache them as a 2x2 grid keyed on (linear, repeat)
 * to avoid allocating a fresh MTLSamplerState on every render pass.
 * Index encoding: (linear << 1) | repeat -- stable so the renderpass
 * chunk can reproduce it.
 */
static inline int sampler_index(bool linear, bool repeat)
{
    return (linear ? 2 : 0) | (repeat ? 1 : 0);
}

id<MTLSamplerState> ra_metal_sampler(struct ra *ra, bool linear, bool repeat)
{
    struct ra_metal *p = ra_metal_get(ra);
    int idx = sampler_index(linear, repeat);
    if (p->samplers[idx])
        return p->samplers[idx];

    MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
    desc.minFilter    = linear ? MTLSamplerMinMagFilterLinear
                               : MTLSamplerMinMagFilterNearest;
    desc.magFilter    = linear ? MTLSamplerMinMagFilterLinear
                               : MTLSamplerMinMagFilterNearest;
    desc.mipFilter    = MTLSamplerMipFilterNotMipmapped;
    desc.sAddressMode = repeat ? MTLSamplerAddressModeRepeat
                               : MTLSamplerAddressModeClampToEdge;
    desc.tAddressMode = desc.sAddressMode;
    desc.rAddressMode = desc.sAddressMode;
    desc.normalizedCoordinates = YES;

    p->samplers[idx] = [p->device newSamplerStateWithDescriptor:desc];
    return p->samplers[idx];
}

/* ------------------------------------------------------------------ */
/* Per-frame command buffer                                           */
/* ------------------------------------------------------------------ */

/*
 * Lazy single-cmd-buf-per-frame strategy. The first encode call in a
 * frame allocates; commit_frame nils it out. This matches how ra_d3d11
 * keeps an ID3D11DeviceContext open across the frame. A future change
 * may want multiple in-flight buffers for parallel compute encoders --
 * the ra->caps RA_CAP_PARALLEL_COMPUTE bit is set, but the actual
 * fanout happens in renderpass_run (TODO(phase-2)).
 */
id<MTLCommandBuffer> ra_metal_cmd_buf(struct ra *ra)
{
    struct ra_metal *p = ra_metal_get(ra);
    if (!p->cmd_buf)
        p->cmd_buf = [p->queue commandBuffer];
    return p->cmd_buf;
}

/* ------------------------------------------------------------------ */
/* ra_init_metal                                                      */
/* ------------------------------------------------------------------ */

bool ra_init_metal(struct ra_ctx *ctx, id<MTLDevice> device,
                   id<MTLCommandQueue> queue)
{
    if (!ctx || !device) {
        if (ctx)
            MP_FATAL(ctx, "ra_init_metal: device must be non-NULL\n");
        return false;
    }

    struct ra *ra = talloc_zero(NULL, struct ra);
    ra->log    = ctx->log;
    /* ctx->global may be NULL in headless unit tests; that's fine, the
     * fields that consume it (spirv config lookup) are guarded. */
    ra->fns    = &ra_fns_metal;

    struct ra_metal *p = talloc_zero(ra, struct ra_metal);
    ra->priv   = p;
    p->device  = device;
    p->global  = ctx->global;
    p->ctx     = ctx;
    p->queue   = queue ?: [device newCommandQueue];
    p->staging_pool = (__bridge_retained void *)[[NSMutableArray alloc] initWithCapacity:8];
    /* NSCache statt NSMutableDictionary: thread-safe, evictet automatisch
     * bei iOS Memory-Pressure-Notifications und limitiert count, sodass
     * lange Sessions mit vielen Format-Wechseln nicht unbounded wachsen.
     * Typische Session hat 20-50 unique passes — 128 lässt komfortabel
     * Headroom für mehrere Videos in einer Sitzung. */
    NSCache *pcache = [[NSCache alloc] init];
    pcache.countLimit = 128;
    pcache.name = @"kuckuck.metal.pipeline_cache";
    p->pipeline_cache = (__bridge_retained void *)pcache;

    /* Vertex ring: 2 MiB persistent shared-storage buffer. Renderpass_run
     * sub-allocates from it for draws whose vertex data exceeds the inline
     * setVertexBytes limit (4 KiB). Reset to offset=0 per frame in
     * ra_metal_commit_frame. Saves one heap allocation per non-inline draw
     * (~10-30 per frame in mpv's render graph). Capacity sized for OSD
     * glyph bursts + occasional full-screen quads. */
    p->vertex_ring_capacity = 2 * 1024 * 1024;
    p->vertex_ring = [p->device newBufferWithLength:p->vertex_ring_capacity
                                            options:MTLResourceStorageModeShared];
    p->vertex_ring_offset = 0;

    /* SPIR-V compiler init. mpv's gl_sc generates GLSL; we compile that
     * to SPIR-V via shaderc, then translate to MSL via SPIRV-Cross. Without
     * spirv_compiler_init(), ctx->spirv stays NULL and every
     * renderpass_create fails silently — which manifests as blank/blue
     * output with audio working. d3d11/context.c:520 follows the same
     * pattern: spirv must be initialised once per ra_ctx before any pass.
     * Note: we share the ra_ctx with libmpv_metal.m's wrap_fbo path; that
     * path doesn't allocate ctx->spirv so we own it here. */
    if (!spirv_compiler_init(ctx)) {
        MP_FATAL(ra, "ra_init_metal: spirv_compiler_init failed\n");
        talloc_free(ra);
        return false;
    }

    /* Cross-Session-PSO-Cache via MTLBinaryArchive. Wenn die Datei vom
     * früheren Launch da ist, läd Metal die fertig-kompilierten PSOs (=
     * spart ~50-300ms pro Pass auf erstem Video-Tap nach Cold-Start).
     * Fall: erstes Launch ever / GPU-Family-Wechsel / mpv-Bump → load
     * schlägt fehl, archive startet leer und wird beim destroy frisch
     * geschrieben. Init muss nach spirv_compiler_init kommen — falls der
     * fehlschlägt wird ra mit talloc_free zerstört, dann führt
     * ra_metal_destroy_fns eine evtl. allozierte archive sauber zurück. */
    id<MTLBinaryArchive> barchive =
        metal_load_or_create_binary_archive(device);
    p->binary_archive = (__bridge_retained void *)barchive;

    /* PSO-Sidecar laden + Background-Warmup anstoßen. MTLBinaryArchive
     * (oben) cacht den final PSO-build, aber GLSL→SPIR-V→MSL→MTLLibrary
     * muss noch jedes Mal laufen weil newRenderPipelineState die
     * MTLFunction-Objekte braucht. Mit Sidecar-Replay im Background-
     * Thread läuft das während User durch's Menü navigiert, sodass
     * pipeline_cache (in-memory NSCache, oben) warm ist wenn der erste
     * Video-Tap kommt. */
    /* PSO-Sidecar wird geladen + beim destroy persistiert, aber das
     * Background-Warmup-Replay ist DISABLED:
     *
     * Symptom: nach jedem Video-Playback crasht der nächste nextDrawable
     * mit korrupteten CAMetalLayer cleanup-callbacks (function-pointer
     * slot enthält tagged-int statt callback-addr). Auch mit
     * dispatch_group_async + cancel-flag + wait-in-destroy reproduzibel
     * → der warmup-thread macht IM LAUF was, das CAMetalLayer-state
     * korrumpiert, nicht erst beim teardown. Vermutlich MTLLibrary/
     * MTLRenderPipelineState-creation auf background-thread interagiert
     * mit dem MTLDevice das main-thread für nextDrawable nutzt; Apple-
     * docs sagen MTLDevice ist thread-safe, in der Praxis offenbar nicht
     * unter dem Druck den der warmup macht.
     *
     * Sidecar bleibt drin (kost nix wenn empty), ggf. später re-enable
     * mit MTLDevice-Mirror oder anderem isolations-mechanismus. */
    /* metal_load_pso_sidecar() returnt [NSMutableArray array] (= autoreleased
     * +0). Diese Datei ist NICHT mit -fobjc-arc kompiliert (meson defaults für
     * mpv .m-files), also ist `__bridge_retained` hier ein no-op cast — kein
     * tatsächlicher retain. Beim nächsten autoreleasepool-drain wird das Array
     * dealloc't und m->pso_sidecar zeigt auf freigegebenen heap-slot, der
     * danach beliebig von anderen Apple-internen Objekten reused wird (gesehen:
     * BSStackFrameInfo, MTLRenderPassSampleBufferAttachmentDescriptorArrayInternal,
     * CAPresentationModifier). CRASH bei nächster Verwendung.
     *
     * Fix: explicit CFRetain damit +1 für die void*-Storage da ist; destroy
     * unten balanced das mit CFRelease. (binary_archive war zufällig safe weil
     * MTLDevice's newBinaryArchiveWithDescriptor: nach Cocoa-`new`-convention
     * +1 owned returnt, nicht autoreleased.) */
    if (!p->queue) {
        MP_FATAL(ra, "ra_init_metal: failed to allocate MTLCommandQueue\n");
        talloc_free(ra);
        return false;
    }

    /* Capability bitset. Metal natively supports all of these on every
     * GPU family mpv targets (Apple silicon + macOS Intel >= Skylake).
     * Granular fallbacks (e.g. compute on older iOS A8) are not worth
     * the maintenance burden -- the minimum deployment target is
     * iOS 13/macOS 10.15 for our shaderc dependency anyway. */
    /* Phase-3 (.25+): RA_CAP_COMPUTE re-enabled. Phase-1's NULL+4 deref
     * in metal_compile_stage was diagnosed before push_const,
     * spirv-compiler-acquisition, and MSL-binding-mapping fixes landed —
     * worth re-validating now that the raster path is stable. If it still
     * crashes, the console capture will show the current stack. */
    /* RA_CAP_PARALLEL_COMPUTE makes mpv's finish_pass_tex() prefer the
     * compute path over fragment for every storage_dst intermediate pass
     * (merge / chroma / scale), exactly as it does on Vulkan and D3D11.
     * Without it those passes fall back to the fragment raster path, where
     * the Apple Metal driver corrupts texture()-sampled chroma writes to
     * non-tile-aligned r8/rg8 targets (360x288). The fragment failure was
     * isolated across experiments A–H: gradient-only writes and compute
     * imageStore writes are both clean, only fragment texture() sampling
     * corrupts. The stale comment in ra_metal_cmd_buf claimed this bit was
     * already set — it never was. Setting it routes the chroma merge through
     * the verified-clean compute path. */
    ra->caps = RA_CAP_TEX_1D
             | RA_CAP_TEX_3D
             | RA_CAP_BLIT
             | RA_CAP_DIRECT_UPLOAD
             | RA_CAP_BUF_RO
             | RA_CAP_BUF_RW
             | RA_CAP_FRAGCOORD
             | RA_CAP_GLOBAL_UNIFORM
             | RA_CAP_GATHER
             | RA_CAP_COMPUTE
             | RA_CAP_PARALLEL_COMPUTE;

    /* GLSL 4.50 + Vulkan dialect: matches what SPIRV-Cross expects on
     * the input side. SPIRV-Cross will then emit MSL. ra->glsl_es stays
     * false -- we never feed Metal an ES dialect. */
    ra->glsl_version = 450;
    ra->glsl_vulkan  = true;
    ra->glsl_es      = false;

    /* Hard-wired limits. The real Metal device exposes
     * MTLDevice.maxBufferLength etc., but for the values mpv actually
     * consults the per-feature-set minimums are a safe lower bound and
     * avoid an init-time device query that would block in a unit test. */
    ra->max_texture_wh             = 16384;
    /* Phase-2: push_constant_layout=std430_layout + setBytes binding,
     * see ra_metal.m vtable + metal_renderpass_run. */
    ra->max_pushc_size             = 4096;
    ra->max_compute_group_threads  = 1024;
    ra->max_shmem                  = 32768; /* 32 KB threadgroup memory */

    /* Build the ra_format table. Each entry is a talloc child of `ra` so
     * the formats die with the ra (same lifetime model as ra_gl.c:149). */
    for (size_t i = 0; i < METAL_NUM_FORMATS; i++) {
        const struct metal_format *mf = &metal_formats[i];
        struct ra_format *fmt = talloc_zero(ra, struct ra_format);
        fmt->name           = mf->name;
        fmt->priv           = (void *)mf;
        fmt->ctype          = mf->ctype;
        fmt->ordered        = true;
        fmt->num_components = mf->num_components;
        fmt->pixel_size     = mf->pixel_size;
        fmt->linear_filter  = mf->linear_filter;
        fmt->renderable     = mf->renderable;
        fmt->storable       = true;
        for (int c = 0; c < mf->num_components; c++) {
            fmt->component_size[c]  = mf->component_size;
            fmt->component_depth[c] = mf->component_depth;
        }
        MP_TARRAY_APPEND(ra, ra->formats, ra->num_formats, fmt);
    }

    ctx->ra = ra;
    MP_VERBOSE(ra, "Metal RA initialised: device=\"%s\", %d formats\n",
               [[device name] UTF8String] ?: "(unknown)", ra->num_formats);
    return true;
}

/* ------------------------------------------------------------------ */
/* ra_metal_wrap_drawable                                             */
/* ------------------------------------------------------------------ */

/*
 * Synthetic dummy format used by the wrapped drawable, mirroring
 * fbo_dummy_format in ra_gl.c:420-429. The drawable's real
 * MTLPixelFormat could be probed via [texture pixelFormat] and matched
 * back into metal_formats[], but the renderer only ever consults
 * dummy_format=true / renderable=true / pixel_size on this path, so a
 * single shared entry is enough.
 */
static const struct metal_format drawable_dummy_metal_format = {
    .name           = "unknown_drawable",
    .mtl_format     = MTLPixelFormatBGRA8Unorm,
    .ctype          = RA_CTYPE_UNORM,
    .num_components = 4,
    .component_size = 8,
    .component_depth = 8,
    .pixel_size     = 4,
    .linear_filter  = true,
    .renderable     = true,
};

const struct ra_format drawable_dummy_ra_format = {
    .name           = "unknown_drawable",
    .priv           = (void *)&drawable_dummy_metal_format,
    .ctype          = RA_CTYPE_UNORM,
    .num_components = 4,
    .pixel_size     = 4,
    .renderable     = true,
    .dummy_format   = true,
};

struct ra_tex *ra_metal_wrap_drawable(struct ra_ctx *ctx,
                                      id<MTLTexture> texture,
                                      int w, int h,
                                      id<MTLCommandBuffer> command_buffer)
{
    if (!ctx || !ctx->ra || !texture)
        return NULL;

    struct ra_metal *p = ra_metal_get(ctx->ra);

    /* If the caller hands us a fresh command buffer mid-frame, commit
     * whatever we had pending before adopting the new one. Dropping it
     * silently would orphan any encoded work. */
    if (command_buffer) {
        if (p->cmd_buf && p->cmd_buf != command_buffer)
            [p->cmd_buf commit];
        p->cmd_buf = command_buffer;
    }

    struct ra_tex *tex = talloc_zero(ctx->ra, struct ra_tex);
    tex->params = (struct ra_tex_params){
        .dimensions = 2,
        .w = w, .h = h, .d = 1,
        .format     = &drawable_dummy_ra_format,
        .render_dst = true,
        .blit_src   = true,
        .blit_dst   = true,
    };

    struct ra_tex_metal *tm = talloc_zero(tex, struct ra_tex_metal);
    tm->texture    = texture;
    tm->is_wrapped = true;
    tex->priv      = tm;
    return tex;
}

/* Wrap externally-owned MTLTexture (= CVMetalTextureCache von hwdec_vt_metal)
 * als sampleable ra_tex. Im Unterschied zu ra_metal_wrap_drawable: dies hier
 * setzt render_src+blit_src statt render_dst, weil die Texture im Shader
 * gesampled wird (= YUV-plane), nicht als render-target dient. fmt kommt vom
 * mpv ra_format-system (= r8/rg8/r16/rg16 für YUV planes). */
struct ra_tex *ra_metal_wrap_external_texture(struct ra_ctx *ctx,
                                              id<MTLTexture> texture,
                                              const struct ra_format *fmt,
                                              int w, int h)
{
    if (!ctx || !ctx->ra || !texture || !fmt) return NULL;
    struct ra_tex *tex = talloc_zero(ctx->ra, struct ra_tex);
    tex->params = (struct ra_tex_params){
        .dimensions = 2,
        .w = w, .h = h, .d = 1,
        .format     = fmt,
        .render_src = true,
        .src_linear = true,
        .blit_src   = true,
    };
    struct ra_tex_metal *tm = talloc_zero(tex, struct ra_tex_metal);
    tm->texture    = texture;
    tm->is_wrapped = true;
    tex->priv      = tm;
    return tex;
}

/* ------------------------------------------------------------------ */
/* Frame commit                                                       */
/* ------------------------------------------------------------------ */

/* Phase-1 tile-shading-feasibility analysis. Walked das per-frame
 * pass-ring + zählt adjacent (N, N+1) wo:
 *   - beide pass->type == RASTER, UND
 *   - target_tex(N) == target_tex(N+1)
 * Das sind direkte tile-shading-merge-Kandidaten: man könnte beide in
 * dieselben MTLRenderCommandEncoder packen weil das output im tile-
 * memory bleiben kann (kein store/load-roundtrip via VRAM). */


void ra_metal_commit_frame(struct ra_ctx *ctx, bool display_synced)
{
    (void)display_synced;
    if (!ctx || !ctx->ra)
        return;
    struct ra_metal *p = ra_metal_get(ctx->ra);
    if (!p->cmd_buf)
        return;
    /* Presentation is intentionally not requested here -- the libmpv
     * embedder owns the CAMetalDrawable lifecycle and calls
     * presentDrawable on the host queue. We only commit. */
    [p->cmd_buf commit];
    p->cmd_buf = nil;
    /* Reset vertex ring offset so the next frame starts from the top.
     * MTLResourceStorageModeShared is CPU-coherent; the just-committed
     * command buffer above ensures GPU is done reading from this frame's
     * range. mpv submits frames synchronously w.r.t. its main loop so
     * single-frame in-flight is the worst case. */
    p->vertex_ring_offset = 0;
}

/* ------------------------------------------------------------------ */
/* Teardown                                                           */
/* ------------------------------------------------------------------ */

void ra_metal_destroy_fns(struct ra *ra)
{
    if (!ra)
        return;
    struct ra_metal *p = ra->priv;
    if (p) {
        /* Warmup-thread synchronisieren BEVOR irgendwas anderes cleanupt
         * wird — der background-thread hält raw ra-pointer und liest
         * p->pipeline_cache / p->pso_sidecar / p->binary_archive in seiner
         * renderpass_create-replay-loop. Cancel-flag damit mid-iteration
         * früh bailen, dann wait (mit timeout um destroy nicht endlos zu
         * blockieren falls warmup hängt). Non-ARC: dispatch_group_create
         * returnt +1, hier explicit dispatch_release zum balancen. */
        if (p->cmd_buf) {
            [p->cmd_buf commit];
            p->cmd_buf = nil;
        }
        for (int i = 0; i < 4; i++)
            p->samplers[i] = nil;
        p->queue  = nil;
        p->device = nil;
        if (p->staging_pool) {
            CFRelease(p->staging_pool);
            p->staging_pool = NULL;
        }
        if (p->binary_archive) {
            id<MTLBinaryArchive> archive =
                (__bridge_transfer id<MTLBinaryArchive>)p->binary_archive;
            p->binary_archive = NULL;
            metal_serialize_binary_archive(archive);
            archive = nil;
        }
        if (p->pipeline_cache) {
            CFRelease(p->pipeline_cache);
            p->pipeline_cache = NULL;
        }
        if (p->blit_cache) {
            CFRelease(p->blit_cache);
            p->blit_cache = NULL;
        }
        p->vertex_ring = nil;
        p->vertex_ring_capacity = 0;
        p->vertex_ring_offset = 0;
    }
    /* `p` is a talloc child of `ra`, so it dies with the parent. */
    talloc_free(ra);
}

void ra_metal_destroy(struct ra_ctx *ctx)
{
    if (!ctx || !ctx->ra)
        return;
    ra_metal_destroy_fns(ctx->ra);
    ctx->ra = NULL;
}

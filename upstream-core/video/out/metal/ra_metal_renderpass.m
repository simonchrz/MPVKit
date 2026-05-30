/* Copyright (C) 2026 the mpv developers
 *
 * Apple Metal backend for mpv's render abstraction (ra). Parallel to
 * video/out/opengl/ra_gl.c. This file is the assembled skeleton for
 * Phase 1 — most lifecycle/format/buffer/texture functions are real
 * implementations; renderpass + timer + scaled-blit + scissored-clear
 * are stubs marked `TODO(phase-2)` pointing at follow-up work.
 *
 * The file is built up from three logical chunks (init, textures,
 * renderpass) — kept inline for static-function locality, matching
 * ra_gl.c's monolithic structure.
 */

#import "ra_metal.h"
#import <Foundation/Foundation.h>
#include "mpv_talloc.h"
#include "common/msg.h"
#include "options/m_config.h"
#include "video/out/gpu/utils.h"
#include "video/out/gpu/spirv.h"
#include "misc/mp_assert.h"

#include <spirv_cross/spirv_cross_c.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <CommonCrypto/CommonDigest.h>


/* PSO-Cache-Stats Counter (Definition in ra_metal.m). */
extern _Atomic int64_t mpv_metal_pso_lookups;
extern _Atomic int64_t mpv_metal_pso_hits;

/* PSO-Sidecar resilient-save kick (Definition in ra_metal.m). */

/* Cache-Key-Logging (Definition in ra_metal.m). */
extern void metal_log_cache_key(const char *status, NSString *cache_key, const char *vsh);



/* Drawable's synthetic ra_format (Definition in ra_metal_init.m).
 * Wird im Warmup-Replay als target_format gesetzt wenn der Sidecar-Entry
 * "unknown_drawable" als format-Namen hat. */
extern const struct ra_format drawable_dummy_ra_format;

/* spirv_conf was used by an earlier (broken) acquisition path. We now use
 * ctx->spirv (populated by spirv_compiler_init in ra_init_metal). The decl
 * stays here only because removing it cascades through include changes. */


/* ============================================================ */

/* Pipeline cache value object. We retain the MTLLibrary because it's the
 * Metal-side container for the compiled GPU code that the PSO references
 * — releasing it can invalidate the PSO on some driver versions. */
@interface MetalPassCacheEntry : NSObject
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTLRenderPipelineState> renderPSO;
@property (nonatomic, strong) id<MTLComputePipelineState> computePSO;
@property (nonatomic, assign) MTLSize computeThreadsPerGroup;
@property (nonatomic, strong) NSData *bindingIndices;  /* int[numInputs] */
@property (nonatomic, assign) int numInputs;
@property (nonatomic, assign) int pushConstantSlot;   /* MSL buffer slot for push_constant block, -1 if none */
@end
@implementation MetalPassCacheEntry @end

/*
 * Vertex stream binding slot. SPIRV-Cross emits MSL that reads vertex
 * attributes from buffer(30); the layout entry on the pipeline state
 * descriptor must agree. Top buffer slots are reserved by Metal for this.
 */
#ifndef kMetalVertexBufferIndex
#define kMetalVertexBufferIndex 30
#endif

/*
 * Slot used for the implicit "loose uniforms" UBO that SPIRV-Cross packs
 * scalar/vector/matrix uniforms into. SPIRV-Cross's default MSL binding
 * for the global uniform block is 0; mirror it here.
 */
#ifndef kMetalGlobalUniformSlot
#define kMetalGlobalUniformSlot 0
#endif

/*
 * Per Metal spec, setVertexBytes:/setFragmentBytes:/setBytes: are only
 * legal up to 4 KB. Above that we have to round-trip through a real
 * MTLBuffer.
 */
#ifndef kMetalInlineBytesLimit
#define kMetalInlineBytesLimit 4096
#endif

/* MSL has one [[texture(N)]] slot space per shader stage, while Vulkan/GLSL
 * permits a sampled image and a storage (write) image to share (set,
 * binding) under different descriptor types. mpv's compute shaders do this
 * (e.g. `image2D out_image` and `sampler2D texture0` both at binding=0).
 * We move every STORAGE_IMAGE binding up by this offset on both the SPIR-V
 * side (via spvc_compiler_set_decoration) and the Metal-side bind call,
 * so the two never collide in the emitted [[texture(N)]] decoration. */
#ifndef kMetalStorageImageSlotOffset
#define kMetalStorageImageSlotOffset 16
#endif

// ============ INIT + FORMATS + LIFECYCLE ============

// ============ TEXTURES + BUFFERS + CLEAR + BLIT ============

// ============ RENDERPASS + TIMER + DEBUG_MARKER ============
/*
 * Metal RA backend: renderpass, timers, debug markers, uniform layout.
 *
 * Parallel chunk to video/out/opengl/ra_gl.c (lines ~720-1208) and
 * video/out/d3d11/ra_d3d11.c (the GLSL->SPIR-V->target-language pipeline
 * around lines 1301-1409). This file collects the ra_fns entries that
 * deal with shader compilation and submission.
 *
 * Phase-1 status: scaffolding only. Shader translation is gated on the
 * shaderc + SPIRV-Cross dependencies which are wired up in phase 2.
 * Every deferred path is marked with the literal token "TODO(phase-2)"
 * so it can be grepped uniformly across files. Don't drop or rename
 * the token without updating the other chunks.
 */



/*
 * mpv's std140 uniform layout helper lives in video/out/opengl/utils.c.
 * Both the GL and (eventually) the Metal backend use it: SPIRV-Cross
 * emits MSL with matching `packed_*` types and explicit padding when the
 * SPIR-V module declares std140-laid-out uniform blocks. Re-using the
 * shared helper guarantees mpv's CPU-side packing matches the GPU-side
 * struct layout.
 */
extern struct ra_layout std140_layout(struct ra_renderpass_input *inp);

/* ------------------------------------------------------------------ */
/* desc_namespace                                                     */
/* ------------------------------------------------------------------ */

/*
 * Same trick as gl_desc_namespace (ra_gl.c:720): each ra_vartype gets
 * its own binding-index namespace. In Metal terms, textures, samplers,
 * read-only buffers and read-write buffers each live in independent
 * binding-index spaces -- a texture at slot 0 does not conflict with a
 * buffer at slot 0. The shader compiler hands out indices per type, so
 * returning the vartype itself as the namespace key is correct.
 */
int metal_desc_namespace(struct ra *ra, enum ra_vartype type)
{
    return type;
}

/* ------------------------------------------------------------------ */
/* renderpass_create / destroy / run                                  */
/* ------------------------------------------------------------------ */

/*
 * Map an mpv vertex-attribute (vartype, vector-dim) pair to a Metal
 * vertex format. Mirrors the DXGI_FORMAT switch in
 * ra_d3d11.c:1668-1685 (renderpass_create_raster). mpv only ever uses
 * FLOAT (1..4 components) and BYTE_UNORM (1, 2, 4 components); other
 * combinations return MTLVertexFormatInvalid so the caller can fail
 * cleanly. The 3-component byte-unorm case has no canonical Metal
 * format -- mpv pads to 4 components on the host side just like d3d11.
 */
static MTLVertexFormat metal_vertex_format(enum ra_vartype type, int dim_v)
{
    switch (type) {
    case RA_VARTYPE_FLOAT:
        switch (dim_v) {
        case 1: return MTLVertexFormatFloat;
        case 2: return MTLVertexFormatFloat2;
        case 3: return MTLVertexFormatFloat3;
        case 4: return MTLVertexFormatFloat4;
        }
        break;
    case RA_VARTYPE_BYTE_UNORM:
        switch (dim_v) {
        case 1: return MTLVertexFormatUCharNormalized;
        case 2: return MTLVertexFormatUChar2Normalized;
        case 4: return MTLVertexFormatUChar4Normalized;
        }
        break;
    default:
        break;
    }
    return MTLVertexFormatInvalid;
}

/*
 * Map mpv blend factors to Metal blend factors. Mirrors map_ra_blend()
 * in ra_d3d11.c:1426-1435. Only the four factors mpv actually uses are
 * defined in ra.h; default to ZERO for safety so an unknown factor at
 * worst zeroes the channel instead of leaving it undefined.
 */
static MTLBlendFactor metal_blend_factor(enum ra_blend f)
{
    switch (f) {
    case RA_BLEND_ONE:                 return MTLBlendFactorOne;
    case RA_BLEND_SRC_ALPHA:           return MTLBlendFactorSourceAlpha;
    case RA_BLEND_ONE_MINUS_SRC_ALPHA: return MTLBlendFactorOneMinusSourceAlpha;
    case RA_BLEND_ZERO:
    default:                           return MTLBlendFactorZero;
    }
}

/*
 * SPIR-V module compiled from one GLSL shader plus the matching
 * SPIRV-Cross/MSL state. Bundled into a small helper struct so the
 * raster path (vert + frag) and the compute path can share the same
 * compile / reflect / cleanup code without duplicating goto-cleanup
 * book-keeping.
 */
struct metal_shader_stage {
    bstr spv;                  /* SPIR-V bytecode (talloc'd) */
    spvc_context sc_ctx;       /* owns parsed_ir + compiler */
    spvc_compiler sc_compiler; /* SPIRV-Cross MSL compiler */
    const char *msl;           /* MSL source -- lifetime tied to sc_ctx */
    id<MTLLibrary> library;    /* compiled MTLLibrary */
    id<MTLFunction> function;  /* "main0" entry point */
};

/*
 * Compile one GLSL stage all the way to a usable MTLFunction. Returns
 * true on success; on failure, stage may be partially populated and the
 * caller is responsible for releasing it via metal_shader_stage_uninit.
 *
 * Pipeline (matches ra_d3d11.c:1301-1409 compile_glsl, adapted for MSL):
 *   GLSL -> shaderc -> SPIR-V -> SPIRV-Cross -> MSL ->
 *     [device newLibraryWithSource:...] -> [lib newFunctionWithName:@"main0"]
 */
static bool metal_compile_stage(struct ra *ra,
                                id<MTLDevice> device,
                                struct spirv_compiler *spirv,
                                void *ta_ctx,
                                enum glsl_shader stage_type,
                                const char *glsl,
                                struct metal_shader_stage *stage)
{
    if (!glsl) {
        MP_ERR(ra, "metal: NULL shader source for stage %d\n", (int)stage_type);
        return false;
    }
    if (!stage) {
        MP_ERR(ra, "metal: NULL stage pointer\n");
        return false;
    }
    if (!spirv || !spirv->fns) {
        MP_ERR(ra, "metal: NULL spirv/fns\n");
        return false;
    }

    /* GLSL -> SPIR-V via shaderc. */
    if (!spirv->fns->compile_glsl(spirv, ta_ctx, stage_type, glsl, &stage->spv))
        return false;

    /* SPIR-V -> MSL via SPIRV-Cross. */
    if (spvc_context_create(&stage->sc_ctx) != SPVC_SUCCESS) {
        MP_ERR(ra, "metal: spvc_context_create failed\n");
        return false;
    }

    spvc_parsed_ir parsed_ir = NULL;
    if (spvc_context_parse_spirv(stage->sc_ctx,
                                 (const SpvId *)stage->spv.start,
                                 stage->spv.len / sizeof(SpvId),
                                 &parsed_ir) != SPVC_SUCCESS)
    {
        MP_ERR(ra, "metal: spvc_context_parse_spirv failed: %s\n",
               spvc_context_get_last_error_string(stage->sc_ctx));
        return false;
    }

    if (spvc_context_create_compiler(stage->sc_ctx, SPVC_BACKEND_MSL,
                                     parsed_ir,
                                     SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                     &stage->sc_compiler) != SPVC_SUCCESS)
    {
        MP_ERR(ra, "metal: spvc_context_create_compiler failed: %s\n",
               spvc_context_get_last_error_string(stage->sc_ctx));
        return false;
    }

    spvc_compiler_options opts = NULL;
    if (spvc_compiler_create_compiler_options(stage->sc_compiler, &opts)
            != SPVC_SUCCESS) {
        MP_ERR(ra, "metal: spvc_compiler_create_compiler_options failed\n");
        return false;
    }

    /* MSL 3.2 (iOS 18+ / macOS 15+). SPIRV-Cross encodes MSL version as
     * decimal `major*10000 + minor*100 + patch` (NOT bit-packed hex despite
     * what skeleton's history of 0x20004 suggested). At/above 3.2 it emits
     * `atomic_thread_fence(..., memory_order_seq_cst, thread_scope_threadgroup)`
     * which requires Apple's MSL compiler to also be set to 3.2 or higher
     * (the identifiers don't exist in earlier MSL). Matches the
     * [copts setLanguageVersion:MTLLanguageVersion3_2] below. */
    spvc_compiler_options_set_uint(opts,
        SPVC_COMPILER_OPTION_MSL_VERSION, 30200);

    /* SPIRV-Cross target platform. We always pick iOS here -- the
     * skeleton's Metal context is currently iOS-only (libmpv_metal.m
     * targets CAMetalLayer on iOS). When the macOS backend lands, swap
     * the constant under a #if TARGET_OS_OSX guard. */
    spvc_compiler_options_set_uint(opts,
        SPVC_COMPILER_OPTION_MSL_PLATFORM, SPVC_MSL_PLATFORM_IOS);

    /* TODO(phase-3): argument buffers are required when a single shader
     * references more than ~31 textures (the Metal default-binding limit).
     * mpv's renderer rarely exceeds 8, so leave them off for now and pay
     * the per-resource setFragmentTexture: cost. */
    spvc_compiler_options_set_bool(opts,
        SPVC_COMPILER_OPTION_MSL_ARGUMENT_BUFFERS, SPVC_FALSE);

    if (spvc_compiler_install_compiler_options(stage->sc_compiler, opts)
            != SPVC_SUCCESS) {
        MP_ERR(ra, "metal: spvc_compiler_install_compiler_options failed\n");
        return false;
    }

    /* Force MSL bindings to match the Vulkan-side SpvDecorationBinding so
     * our renderpass_run setFragmentTexture:atIndex: numbers line up with
     * what the emitted MSL shader actually reads from. Without this,
     * SPIRV-Cross auto-renumbers (e.g. layout(binding=7) → [[texture(0)]])
     * and we end up sampling the wrong texture (red checkerboard symptom
     * from sampling the dither LUT instead of the Y plane). We sweep every
     * reflected resource and pin its MSL slot == its Vulkan slot for both
     * stages compiled here. */
    spvc_resources res = NULL;
    if (spvc_compiler_create_shader_resources(stage->sc_compiler, &res) == SPVC_SUCCESS) {
        static const spvc_resource_type kTypes[] = {
            SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
            SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
            SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
            SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
        };
        for (size_t t = 0; t < MP_ARRAY_SIZE(kTypes); t++) {
            const spvc_reflected_resource *list = NULL;
            size_t count = 0;
            if (spvc_resources_get_resource_list_for_type(res, kTypes[t],
                                                          &list, &count)
                    != SPVC_SUCCESS) continue;
            for (size_t i = 0; i < count; i++) {
                unsigned set = spvc_compiler_get_decoration(stage->sc_compiler,
                                  list[i].id, SpvDecorationDescriptorSet);
                unsigned bind = spvc_compiler_get_decoration(stage->sc_compiler,
                                  list[i].id, SpvDecorationBinding);
                /* MSL has ONE [[texture(N)]] slot space. Vulkan/GLSL lets a
                 * write-only image and a sampled image share (set, binding)
                 * because they are different descriptor types. mpv's compute
                 * shaders rely on this — `image2D out_image` and
                 * `sampler2D texture0` both at binding=0. SPIRV-Cross's
                 * resource_bindings map is keyed only by (stage, set, binding)
                 * so a single pin can't separate them. Offset STORAGE_IMAGE
                 * by kMetalStorageImageSlotOffset both in the SPIR-V binding
                 * decoration AND in renderpass_run's setComputeTexture call,
                 * so SAMPLED stays at its natural slot and STORAGE moves up. */
                if (kTypes[t] == SPVC_RESOURCE_TYPE_STORAGE_IMAGE) {
                    bind += kMetalStorageImageSlotOffset;
                    spvc_compiler_set_decoration(stage->sc_compiler,
                                                 list[i].id,
                                                 SpvDecorationBinding,
                                                 bind);
                }
                spvc_msl_resource_binding rb = {0};
                rb.stage = (stage_type == GLSL_SHADER_VERTEX)
                         ? SpvExecutionModelVertex
                         : (stage_type == GLSL_SHADER_FRAGMENT)
                         ? SpvExecutionModelFragment
                         : SpvExecutionModelGLCompute;
                rb.desc_set      = set;
                rb.binding       = bind;
                rb.msl_buffer    = bind;
                rb.msl_texture   = bind;
                rb.msl_sampler   = bind;
                spvc_compiler_msl_add_resource_binding(stage->sc_compiler, &rb);
            }
        }
        /* Push-constant has no layout(binding=N) decoration; without an
         * explicit override SPIRV-Cross auto-assigns it MSL buffer(0),
         * which collides with the UBO we just pinned to buffer(0) — two
         * `constant Struct&` parameters at the same slot in the emitted
         * MSL → undefined behaviour, all-blue output. Pin push_const to
         * slot 28 (well below the vertex buffer at 30, well above any
         * normal UBO/SSBO mpv generates). */
        spvc_msl_resource_binding pcb = {0};
        pcb.stage = (stage_type == GLSL_SHADER_VERTEX)
                  ? SpvExecutionModelVertex
                  : (stage_type == GLSL_SHADER_FRAGMENT)
                  ? SpvExecutionModelFragment
                  : SpvExecutionModelGLCompute;
        pcb.desc_set     = 0xFFFFFFFFu;   /* ResourceBindingPushConstantDescriptorSet */
        pcb.binding      = 0;             /* ResourceBindingPushConstantBinding (binding=0, NOT ~0u) */
        pcb.msl_buffer   = 28;
        pcb.msl_texture  = 0;
        pcb.msl_sampler  = 0;
        spvc_compiler_msl_add_resource_binding(stage->sc_compiler, &pcb);
    }

    if (spvc_compiler_compile(stage->sc_compiler, &stage->msl) != SPVC_SUCCESS) {
        MP_ERR(ra, "metal: spvc_compiler_compile (MSL) failed: %s\n",
               spvc_context_get_last_error_string(stage->sc_ctx));
        return false;
    }

    /* Shader-dump diagnostic (ios-metal.50). */

    /* MSL -> MTLLibrary. */
    NSString *src = [NSString stringWithUTF8String:stage->msl];
    MTLCompileOptions *copts = [MTLCompileOptions new];
    [copts setLanguageVersion:MTLLanguageVersion3_2];

    NSError *err = nil;
    stage->library = [device newLibraryWithSource:src
                                          options:copts
                                            error:&err];
    if (!stage->library) {
        MP_ERR(ra, "metal: newLibraryWithSource failed: %s\n",
               err.localizedDescription
                   ? err.localizedDescription.UTF8String
                   : "unknown error");
        MP_ERR(ra, "metal: failing GLSL source was:\n%s\n", glsl);
        MP_ERR(ra, "metal: translated MSL was:\n%s\n", stage->msl);
        return false;
    }

    /* SPIRV-Cross always emits the entry point as "main0" for the MSL
     * backend (it renames `main` to dodge MSL's reserved-word rule). */
    stage->function = [stage->library newFunctionWithName:@"main0"];
    if (!stage->function) {
        MP_ERR(ra, "metal: library has no 'main0' entry point\n");
        return false;
    }

    return true;
}

/* Release every Metal/SPIRV-Cross resource held by a stage. Safe to call
 * on a zero-initialised or partially-initialised struct. */
static void metal_shader_stage_uninit(struct metal_shader_stage *stage)
{
    stage->function = nil;
    stage->library  = nil;
    /* sc_compiler is owned by sc_ctx (SPVC_CAPTURE_MODE_TAKE_OWNERSHIP). */
    if (stage->sc_ctx) {
        spvc_context_destroy(stage->sc_ctx);
        stage->sc_ctx      = NULL;
        stage->sc_compiler = NULL;
        stage->msl         = NULL;
    }
    /* stage->spv is a child of the surrounding talloc ta_ctx. */
}

/*
 * Map mpv ra_renderpass_input.binding (mpv-side slot number) to the
 * SPIRV-Cross-emitted Metal binding slot, by name. SPIRV-Cross numbers
 * its MSL bindings sequentially in declaration order, which usually --
 * but not always -- matches mpv's. Going through the reflection API is
 * the only way to be sure.
 *
 * Returns -1 if the input isn't found in the reflected resource list
 * (which is fine: scalar uniforms that ended up in the implicit global
 * UBO won't show up under any of the named resource types).
 */
static int metal_lookup_binding(spvc_compiler sc_compiler,
                                spvc_resources resources,
                                spvc_resource_type rtype,
                                const char *want_name)
{
    const spvc_reflected_resource *list = NULL;
    size_t count = 0;
    if (spvc_resources_get_resource_list_for_type(resources, rtype,
                                                  &list, &count)
            != SPVC_SUCCESS) {
        return -1;
    }
    for (size_t i = 0; i < count; i++) {
        if (list[i].name && want_name && strcmp(list[i].name, want_name) == 0) {
            /* SPIRV-Cross assigns its OWN automatic MSL binding indices
             * ([[texture(0)]], [[buffer(0)]] etc.) — these are NOT the same
             * as SpvDecorationBinding (the layout(binding=N) from the
             * original Vulkan-flavoured GLSL). Using the Vulkan binding
             * makes setFragmentTexture:atIndex: target the wrong slot and
             * the shader samples a random texture (e.g. dither LUT instead
             * of the YUV planes → solid colored checkerboard output).
             * msl_get_automatic_resource_binding returns the MSL index. */
            unsigned msl_idx =
                spvc_compiler_msl_get_automatic_resource_binding(sc_compiler,
                                                                 list[i].id);
            if (msl_idx == (unsigned)-1)
                return -1;
            return (int)msl_idx;
        }
    }
    return -1;
}

/*
 * Full implementation. Template is ra_d3d11.c:1301-1409 (the compile_glsl
 * helper) plus the D3D11 pipeline-state assembly in
 * renderpass_create_raster (ra_d3d11.c:1611-1741) and
 * renderpass_create_compute (ra_d3d11.c:1743+). The Metal pipeline is
 * GLSL -> shaderc -> SPIR-V -> SPIRV-Cross -> MSL -> MTLLibrary ->
 * MTLRenderPipelineState (raster) or MTLComputePipelineState (compute).
 */
struct ra_renderpass *metal_renderpass_create(struct ra *ra,
                                const struct ra_renderpass_params *params)
{
    struct ra_metal *m = ra_metal_get(ra);
    id<MTLDevice> device = m->device;

    void *ta_ctx = talloc_new(NULL);
    struct ra_renderpass *pass = NULL;
    struct ra_renderpass_metal *pass_metal = NULL;

    struct metal_shader_stage vert = {0};
    struct metal_shader_stage frag = {0};
    struct metal_shader_stage comp = {0};

    // === 0. Pipeline cache lookup ===
    /* Cache key from GLSL input (cheap) before the expensive shaderc +
     * SPIRV-Cross + Metal-compile chain. On hit we return a fresh
     * ra_renderpass that reuses the cached library/PSO — saves ~150-300ms
     * per pass and avoids the per-pass MTLLibrary alloc that contributes
     * to long-playback memory pressure. */
    NSCache *pipeline_cache = (__bridge NSCache *)m->pipeline_cache;
    NSMutableString *cache_key_buf = [NSMutableString new];
    if (params->vertex_shader)  [cache_key_buf appendFormat:@"V|%s|", params->vertex_shader];
    if (params->frag_shader)    [cache_key_buf appendFormat:@"F|%s|", params->frag_shader];
    if (params->compute_shader) [cache_key_buf appendFormat:@"C|%s|", params->compute_shader];
    if (params->target_format && params->target_format->name)
        [cache_key_buf appendFormat:@"T|%s|", params->target_format->name];
    [cache_key_buf appendFormat:@"B|%d|", params->enable_blend];
    NSString *cache_key = [cache_key_buf copy];

    MetalPassCacheEntry *cached = [pipeline_cache objectForKey:cache_key];

    if (cached) {
        pass = talloc_zero(NULL, struct ra_renderpass);
        pass_metal = talloc_zero(pass, struct ra_renderpass_metal);
        pass->priv = pass_metal;
        pass_metal->library = cached.library;
        pass_metal->render_pso = cached.renderPSO;
        pass_metal->compute_pso = cached.computePSO;
        pass_metal->compute_threads_per_group = cached.computeThreadsPerGroup;
        pass_metal->num_inputs = cached.numInputs;
        pass_metal->push_constant_slot = cached.pushConstantSlot;
        if (cached.numInputs > 0 && cached.bindingIndices) {
            pass_metal->binding_indices = talloc_array(pass_metal, int, cached.numInputs);
            memcpy(pass_metal->binding_indices, cached.bindingIndices.bytes,
                   sizeof(int) * cached.numInputs);
        }
        pass->params = *ra_renderpass_params_copy(pass, params);
        pass->params.cached_program = (struct bstr){0};
        talloc_free(ta_ctx);
        return pass;
    }

    // === 1. Acquire SPIR-V compiler ===
    /* Use the ra_ctx->spirv that was initialized by spirv_compiler_init()
     * during ra_init_metal. The old `mp_get_config_group(&spirv_conf)`
     * path returned a `struct spirv_opts` (the int-only options struct),
     * NOT a compiler — casting it produced fns=NULL → silent renderpass
     * failure → blank/blue output. */
    struct spirv_compiler *spirv = m->ctx ? m->ctx->spirv : NULL;
    if (!spirv || !spirv->fns) {
        MP_ERR(ra, "metal: no SPIR-V compiler available\n");
        goto fail;
    }

    // === 2. Compile GLSL -> SPIR-V (per stage) ===
    if (params->type == RA_RENDERPASS_TYPE_RASTER) {
        if (!metal_compile_stage(ra, device, spirv, ta_ctx,
                                 GLSL_SHADER_VERTEX,
                                 params->vertex_shader, &vert))
            goto fail;
        if (!metal_compile_stage(ra, device, spirv, ta_ctx,
                                 GLSL_SHADER_FRAGMENT,
                                 params->frag_shader, &frag))
            goto fail;
    } else if (params->type == RA_RENDERPASS_TYPE_COMPUTE) {
        if (!metal_compile_stage(ra, device, spirv, ta_ctx,
                                 GLSL_SHADER_COMPUTE,
                                 params->compute_shader, &comp))
            goto fail;
    } else {
        MP_ERR(ra, "metal: invalid renderpass type %d\n", (int)params->type);
        goto fail;
    }

    // === 3. Allocate ra_renderpass + ra_renderpass_metal priv ===
    pass = talloc_zero(NULL, struct ra_renderpass);
    pass_metal = talloc_zero(pass, struct ra_renderpass_metal);
    pass->priv = pass_metal;
    pass_metal->num_inputs = params->num_inputs;
    pass_metal->push_constant_slot = -1;  /* set in reflection step if pushc present */
    if (params->num_inputs > 0) {
        pass_metal->binding_indices =
            talloc_zero_array(pass_metal, int, params->num_inputs);
        for (int i = 0; i < params->num_inputs; i++)
            pass_metal->binding_indices[i] = params->inputs[i].binding;
    }

    if (params->type == RA_RENDERPASS_TYPE_RASTER) {
        // === 4. Vertex descriptor (raster only) ===
        MTLVertexDescriptor *vd = [MTLVertexDescriptor new];
        for (int i = 0; i < params->num_vertex_attribs; i++) {
            struct ra_renderpass_input *attr = &params->vertex_attribs[i];
            MTLVertexFormat vfmt = metal_vertex_format(attr->type, attr->dim_v);
            if (vfmt == MTLVertexFormatInvalid) {
                MP_ERR(ra, "metal: unsupported vertex attribute "
                           "(type=%d dim_v=%d)\n",
                       (int)attr->type, attr->dim_v);
                goto fail;
            }
            vd.attributes[i].format      = vfmt;
            vd.attributes[i].offset      = attr->offset;
            vd.attributes[i].bufferIndex = kMetalVertexBufferIndex;
        }
        vd.layouts[kMetalVertexBufferIndex].stride       = params->vertex_stride;
        vd.layouts[kMetalVertexBufferIndex].stepFunction =
            MTLVertexStepFunctionPerVertex;

        // === 5. Render pipeline descriptor ===
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction   = vert.function;
        pd.fragmentFunction = frag.function;
        pd.vertexDescriptor = vd;

        MTLPixelFormat tfmt = ra_metal_pixel_format(params->target_format);
        if (tfmt == MTLPixelFormatInvalid) {
            MP_ERR(ra, "metal: unsupported target format '%s'\n",
                   params->target_format
                       ? params->target_format->name : "(null)");
            goto fail;
        }
        /* TODO(phase-3): multi-attachment fragment outputs (mpv currently
         * always renders to a single color attachment, but the renderer
         * may grow MRT support for HDR tone-mapping pipelines). */
        pd.colorAttachments[0].pixelFormat = tfmt;

        if (params->enable_blend) {
            pd.colorAttachments[0].blendingEnabled = YES;
            pd.colorAttachments[0].sourceRGBBlendFactor =
                metal_blend_factor(params->blend_src_rgb);
            pd.colorAttachments[0].destinationRGBBlendFactor =
                metal_blend_factor(params->blend_dst_rgb);
            pd.colorAttachments[0].sourceAlphaBlendFactor =
                metal_blend_factor(params->blend_src_alpha);
            pd.colorAttachments[0].destinationAlphaBlendFactor =
                metal_blend_factor(params->blend_dst_alpha);
            pd.colorAttachments[0].rgbBlendOperation   = MTLBlendOperationAdd;
            pd.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        }

        /* MTLBinaryArchive für Cross-Session-PSO-Cache. Bei einem Cache-
         * Hit lädt Metal das fertig-kompilierte PSO aus dem archive (statt
         * ~50-300ms shaderc+SPIRV-Cross+Metal-compile). Der archive enthält
         * dieselbe MTLRenderPipelineDescriptor-Hash-Logik wie unser in-
         * memory pipeline_cache, nur persistent. */
        id<MTLBinaryArchive> barchive =
            (__bridge id<MTLBinaryArchive>)m->binary_archive;
        if (barchive)
            pd.binaryArchives = @[barchive];

        NSError *pso_err = nil;
        pass_metal->render_pso =
            [device newRenderPipelineStateWithDescriptor:pd error:&pso_err];
        if (!pass_metal->render_pso) {
            MP_ERR(ra, "metal: newRenderPipelineStateWithDescriptor failed: %s\n",
                   pso_err.localizedDescription
                       ? pso_err.localizedDescription.UTF8String
                       : "unknown error");
            goto fail;
        }
        /* PSO erfolgreich kompiliert → in archive aufnehmen damit nächste
         * Session den compile-skip kriegt. Fehler ignorieren — best-effort. */
        if (barchive)
            [barchive addRenderPipelineFunctionsWithDescriptor:pd error:nil];
        /* Keep the fragment library alive on the pass priv. Vertex library
         * is kept alive transitively via the pipeline state; we publish the
         * fragment one because debug-marker / reflection paths may need
         * to query it after creation. */
        pass_metal->library = frag.library;
    } else {
        // === 5'. Compute pipeline state ===
        id<MTLBinaryArchive> barchive =
            (__bridge id<MTLBinaryArchive>)m->binary_archive;
        MTLComputePipelineDescriptor *cd = [MTLComputePipelineDescriptor new];
        cd.computeFunction = comp.function;
        if (barchive)
            cd.binaryArchives = @[barchive];

        NSError *cpo_err = nil;
        pass_metal->compute_pso =
            [device newComputePipelineStateWithDescriptor:cd
                                                  options:MTLPipelineOptionNone
                                               reflection:nil
                                                    error:&cpo_err];
        if (!pass_metal->compute_pso) {
            MP_ERR(ra, "metal: newComputePipelineStateWithDescriptor failed: %s\n",
                   cpo_err.localizedDescription
                       ? cpo_err.localizedDescription.UTF8String
                       : "unknown error");
            goto fail;
        }
        if (barchive)
            [barchive addComputePipelineFunctionsWithDescriptor:cd error:nil];
        pass_metal->library = comp.library;

        /* Pull the GLSL `layout(local_size_x = N, local_size_y = M, ...) in`
         * declaration out of the SPIR-V so dispatch knows the threadgroup
         * size. SPIRV-Cross exposes it via the ExecutionMode argument API.
         * Falls back to (1,1,1) — legal, but starves the GPU. */
        unsigned lx = spvc_compiler_get_execution_mode_argument_by_index(
            comp.sc_compiler, SpvExecutionModeLocalSize, 0);
        unsigned ly = spvc_compiler_get_execution_mode_argument_by_index(
            comp.sc_compiler, SpvExecutionModeLocalSize, 1);
        unsigned lz = spvc_compiler_get_execution_mode_argument_by_index(
            comp.sc_compiler, SpvExecutionModeLocalSize, 2);
        if (lx == 0) lx = 1;
        if (ly == 0) ly = 1;
        if (lz == 0) lz = 1;
        pass_metal->compute_threads_per_group = MTLSizeMake(lx, ly, lz);
    }

    // === 6. Binding indices == mpv input.binding ===
    /* spvc_compiler_msl_add_resource_binding above pinned MSL slots to
     * match the Vulkan-side bindings, so the runtime mapping is identity
     * (no reflection-by-name lookup needed). */
    (void)metal_lookup_binding;  /* legacy helper, kept for reference */

    // === 6a. Push-constant MSL slot ===
    /* metal_compile_stage pinned push_const to MSL [[buffer(28)]] via
     * spvc_compiler_msl_add_resource_binding (desc_set=~0u, binding=0 — the
     * SPIRV-Cross magic for push-const). The Metal side delivers the bytes
     * via setVertex/FragmentBytes:length:atIndex:28 in renderpass_run. */
    if (params->push_constants_size > 0)
        pass_metal->push_constant_slot = 28;

    // === 7. Publish the params (copied so the caller can drop theirs) ===
    pass->params = *ra_renderpass_params_copy(pass, params);
    /* Metal doesn't reuse the SPIR-V text cache the way d3d11 does --
     * MSL is fast enough to (re)compile that the disk-cache round-trip
     * isn't worth the cross-build headaches yet. */
    pass->params.cached_program = (struct bstr){0};

    // === 8. Cache the freshly-built pipeline state ===
    /* Save to pipeline_cache so the next renderpass_create with the same
     * GLSL skips the full shaderc+SPIRV-Cross+Metal-compile chain.
     * cache_key was computed at the top from the same GLSL. */
    {
        MetalPassCacheEntry *entry = [MetalPassCacheEntry new];
        entry.library = pass_metal->library;
        entry.renderPSO = pass_metal->render_pso;
        entry.computePSO = pass_metal->compute_pso;
        entry.computeThreadsPerGroup = pass_metal->compute_threads_per_group;
        entry.numInputs = pass_metal->num_inputs;
        entry.pushConstantSlot = pass_metal->push_constant_slot;
        if (pass_metal->num_inputs > 0 && pass_metal->binding_indices) {
            entry.bindingIndices =
                [NSData dataWithBytes:pass_metal->binding_indices
                               length:sizeof(int) * pass_metal->num_inputs];
        }
        [pipeline_cache setObject:entry forKey:cache_key];

    }

    // === 9. Cleanup ===
    // NOTE: don't talloc_free(spirv) — it's owned by ra_ctx (set up by
    // spirv_compiler_init in ra_init_metal). Freeing it nulls ctx->spirv's
    // backing memory; the talloc allocator reuses that slot for the next
    // allocation → next renderpass_create reads fns=NULL → silent fail.
    metal_shader_stage_uninit(&vert);
    metal_shader_stage_uninit(&frag);
    metal_shader_stage_uninit(&comp);
    talloc_free(ta_ctx);
    return pass;

fail:
    metal_shader_stage_uninit(&vert);
    metal_shader_stage_uninit(&frag);
    metal_shader_stage_uninit(&comp);
    if (pass)
        talloc_free(pass);
    talloc_free(ta_ctx);
    return NULL;
}

void metal_renderpass_destroy(struct ra *ra, struct ra_renderpass *pass)
{
    (void)ra;
    if (!pass)
        return;

    struct ra_renderpass_metal *pass_metal = pass->priv;
    if (pass_metal) {
        /*
         * ARC owns the Metal objects -- nilling the strong refs is what
         * actually releases them. Be explicit so the lifetimes are
         * visible at the call site (and so a future move to a non-ARC
         * subset still does the right thing).
         */
        pass_metal->render_pso  = nil;
        pass_metal->compute_pso = nil;
        pass_metal->library     = nil;
        /* binding_indices is talloc'd as a child of pass_metal. */
    }
    talloc_free(pass);
}

/*
 * Dispatch one draw or compute pass. Mirrors gl_renderpass_run() in
 * video/out/opengl/ra_gl.c:1038-1098 and renderpass_run() in
 * video/out/d3d11/ra_d3d11.c:1906-1986: walk params->values, route each
 * input by ra_vartype to the appropriate Metal encoder setter, then issue
 * the draw / dispatchThreadgroups call.
 *
 * Notes on uniform handling: SPIRV-Cross collapses GLSL loose uniforms
 * (scalars, vectors, matrices) into a single std140-laid-out global UBO,
 * which we deliver via setFragmentBytes:/setBytes: at slot
 * kMetalGlobalUniformSlot. ra_metal_get_uniform_offset() walks
 * pass->params.inputs[].offset (populated by metal_renderpass_create from
 * std140_layout) to pack the contiguous block on the stack.
 *
 * Per-buffer last_used_cb tracking matches ra_metal_buf_poll's
 * expectations: a buffer is "in flight" until the cb that referenced it
 * has signalled completion.
 */

/* Size in bytes of a host-side uniform value (scalar/vector/matrix). */
static size_t metal_uniform_value_size(const struct ra_renderpass_input *in)
{
    /* dim_v * dim_m elements, 4 bytes each for FLOAT/INT (std140 column
     * stride is handled at pack-emit time by std140_layout; the host data
     * arrives already laid out, so we just need the contiguous byte
     * count). For mat3 GLSL hands us 9 floats; SPIRV-Cross will emit MSL
     * that reads them as packed_float3x3 with appropriate padding -- our
     * std140 packing on the host side puts that padding in for us. */
    return (size_t)in->dim_v * (size_t)in->dim_m * sizeof(float);
}

void metal_renderpass_run(struct ra *ra,
                                 const struct ra_renderpass_run_params *params)
{
    struct ra_metal *m = ra_metal_get(ra);
    struct ra_renderpass *pass = params->pass;
    struct ra_renderpass_metal *pm = pass->priv;

    id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);
    if (!cb) {
        MP_ERR(ra, "metal: renderpass_run: no command buffer\n");
        return;
    }


    /*
     * Collect loose-uniform inputs into a single std140 block. We sweep
     * params->values once before issuing the encoder, since the same
     * block is bound to both vertex and fragment (raster) or compute
     * (compute) at slot kMetalGlobalUniformSlot. Max size is bounded by
     * pass->params.push_constants_size when set, otherwise we
     * conservatively cap at 4 KB (Metal's setBytes: limit).
     */
    /* Stack-zero damit ungepflegte gaps in der UBO-Layout (= mpv updated nur
     * einige uniforms pro frame, andere stay-default) nicht garbage-bytes vom
     * caller-stack an die GPU senden. Manifesterte sich als wrong colormatrix
     * / falsche Texture-rotation für broadcast-streams (raw-TS): mpv updated
     * den colormatrix-mat3 (offset 0-47), aber texture_rot0/rot1 (offset 48+)
     * nicht jedes frame → stack garbage statt zero/identity. Bei HLS keine
     * artifacts weil dort alle UBO-fields jeden frame neu geschrieben werden. */
    uint8_t uniform_block[kMetalInlineBytesLimit] = {0};
    size_t uniform_block_size = 0;
    bool have_uniform_block = false;

    for (int i = 0; i < params->num_values; i++) {
        const struct ra_renderpass_input_val *val = &params->values[i];
        const struct ra_renderpass_input *in = &pass->params.inputs[val->index];
        if (in->type != RA_VARTYPE_FLOAT && in->type != RA_VARTYPE_INT)
            continue;
        size_t vsize = metal_uniform_value_size(in);
        size_t end = in->offset + vsize;
        if (end > sizeof(uniform_block)) {
            MP_ERR(ra, "metal: uniform block exceeds %d bytes "
                       "(needs %zu); TODO(phase-3): spill to ring buffer\n",
                   kMetalInlineBytesLimit, end);
            return;
        }
        memcpy(uniform_block + in->offset, val->data, vsize);
        if (end > uniform_block_size)
            uniform_block_size = end;
        have_uniform_block = true;
    }


    switch (pass->params.type) {
    case RA_RENDERPASS_TYPE_RASTER: {
        MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor new];
        id<MTLTexture> target_tex =
            ((struct ra_tex_metal *)params->target->priv)->texture;
        rpd.colorAttachments[0].texture = target_tex;

        /* Phase-1 tile-shading-feasibility tracking: record (target, raster)
         * im per-frame-ring. ra_metal_commit_frame walked das ring + zählt
         * same-target-adjacent-pairs. */
        /* mpv's ra_renderpass_run_params has no invalidate flag; preserve
         * target contents (MTLLoadActionLoad), mirroring the GL backend
         * which doesn't glClear before issuing draw calls. */
        rpd.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> enc =
            [cb renderCommandEncoderWithDescriptor:rpd];
        m->active_render_enc = enc;
        [enc setRenderPipelineState:pm->render_pso];

        /* Push constants — bind once on both vertex + fragment stages.
         * mpv populates params->push_constants buffer per dispatch with
         * std430-laid-out data matching what gl_sc declared. */
        if (pm->push_constant_slot >= 0
            && params->push_constants
            && pass->params.push_constants_size > 0)
        {
            [enc setVertexBytes:params->push_constants
                         length:pass->params.push_constants_size
                        atIndex:pm->push_constant_slot];
            [enc setFragmentBytes:params->push_constants
                           length:pass->params.push_constants_size
                          atIndex:pm->push_constant_slot];
        }

        MTLViewport vp = {
            .originX = params->viewport.x0,
            .originY = params->viewport.y0,
            .width   = params->viewport.x1 - params->viewport.x0,
            .height  = params->viewport.y1 - params->viewport.y0,
            .znear   = 0,
            .zfar    = 1,
        };
        [enc setViewport:vp];

        /* Metal asserts when the scissor rect leaves the attachment, so
         * clamp here. Matches the d3d11 backend's tolerance for the
         * caller passing rects with x1 > width. */
        NSUInteger tex_w = target_tex.width;
        NSUInteger tex_h = target_tex.height;
        NSUInteger sx = (NSUInteger)MPMAX(0, params->scissors.x0);
        NSUInteger sy = (NSUInteger)MPMAX(0, params->scissors.y0);
        NSUInteger sw = 0, sh = 0;
        if (sx < tex_w) {
            NSUInteger want = (NSUInteger)MPMAX(0, params->scissors.x1 - (int)sx);
            sw = MPMIN(want, tex_w - sx);
        }
        if (sy < tex_h) {
            NSUInteger want = (NSUInteger)MPMAX(0, params->scissors.y1 - (int)sy);
            sh = MPMIN(want, tex_h - sy);
        }
        MTLScissorRect sr = { .x = sx, .y = sy, .width = sw, .height = sh };
        [enc setScissorRect:sr];

        /* Bind inputs. */
        for (int i = 0; i < params->num_values; i++) {
            const struct ra_renderpass_input_val *val = &params->values[i];
            const struct ra_renderpass_input *in =
                &pass->params.inputs[val->index];
            int slot = pm->binding_indices
                       ? pm->binding_indices[val->index]
                       : in->binding;
            switch (in->type) {
            case RA_VARTYPE_TEX: {
                struct ra_tex *t = *(struct ra_tex **)val->data;
                struct ra_tex_metal *tm = t->priv;
                [enc setFragmentTexture:tm->texture atIndex:slot];
                [enc setFragmentSamplerState:
                        ra_metal_sampler(ra, t->params.src_linear,
                                             t->params.src_repeat)
                                     atIndex:slot];
                break;
            }
            case RA_VARTYPE_IMG_W: {
                struct ra_tex *t = *(struct ra_tex **)val->data;
                struct ra_tex_metal *tm = t->priv;
                /* Storage images live offset from sampled textures in the
                 * MSL [[texture(N)]] space to avoid collision when GLSL
                 * gives an image and a sampler the same binding number. */
                [enc setFragmentTexture:tm->texture
                                atIndex:slot + kMetalStorageImageSlotOffset];
                break;
            }
            case RA_VARTYPE_BUF_RO:
            case RA_VARTYPE_BUF_RW: {
                struct ra_buf *b = *(struct ra_buf **)val->data;
                struct ra_buf_metal *bm = b->priv;
                [enc setFragmentBuffer:bm->buffer offset:0 atIndex:slot];
                bm->last_used_cb = cb;
                break;
            }
            case RA_VARTYPE_FLOAT:
            case RA_VARTYPE_INT:
                /* Packed into uniform_block above. */
                break;
            default:
                MP_WARN(ra, "metal: unhandled input vartype %d\n",
                        (int)in->type);
                break;
            }
        }

        if (have_uniform_block) {
            [enc setVertexBytes:uniform_block
                         length:uniform_block_size
                        atIndex:kMetalGlobalUniformSlot];
            [enc setFragmentBytes:uniform_block
                           length:uniform_block_size
                          atIndex:kMetalGlobalUniformSlot];
        }

        /* Vertex stream. Inline if it fits in the 4 KiB setVertexBytes
         * fast-path; otherwise sub-allocate from the per-frame vertex
         * ring (2 MiB, reset in ra_metal_commit_frame). Ring offsets
         * are 16-byte aligned per Metal's vertex buffer requirements.
         * Fallback to one-shot newBufferWithBytes if the draw doesn't
         * fit the ring remainder. */
        size_t vbytes = (size_t)params->vertex_count *
                        (size_t)pass->params.vertex_stride;
        if (vbytes <= kMetalInlineBytesLimit) {
            [enc setVertexBytes:params->vertex_data
                         length:vbytes
                        atIndex:kMetalVertexBufferIndex];
        } else {
            size_t aligned_offset = (m->vertex_ring_offset + 15) & ~(size_t)15;
            if (m->vertex_ring && aligned_offset + vbytes <= m->vertex_ring_capacity) {
                memcpy((uint8_t *)m->vertex_ring.contents + aligned_offset,
                       params->vertex_data, vbytes);
                [enc setVertexBuffer:m->vertex_ring
                              offset:aligned_offset
                             atIndex:kMetalVertexBufferIndex];
                m->vertex_ring_offset = aligned_offset + vbytes;
            } else {
                id<MTLBuffer> vbuf = [m->device
                    newBufferWithBytes:params->vertex_data
                                length:vbytes
                               options:MTLResourceStorageModeShared];
                [enc setVertexBuffer:vbuf offset:0
                             atIndex:kMetalVertexBufferIndex];
            }
        }

        [enc drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:params->vertex_count];
        [enc endEncoding];
        m->active_render_enc = nil;

        break;
    }

    case RA_RENDERPASS_TYPE_COMPUTE: {
        /* Phase-1 tracking: compute hat keinen target_tex im pass-sense. */

        id<MTLComputeCommandEncoder> cenc = [cb computeCommandEncoder];
        [cenc setComputePipelineState:pm->compute_pso];

        /* Push constants for compute shader. */
        if (pm->push_constant_slot >= 0
            && params->push_constants
            && pass->params.push_constants_size > 0)
        {
            [cenc setBytes:params->push_constants
                    length:pass->params.push_constants_size
                   atIndex:pm->push_constant_slot];
        }

        for (int i = 0; i < params->num_values; i++) {
            const struct ra_renderpass_input_val *val = &params->values[i];
            const struct ra_renderpass_input *in =
                &pass->params.inputs[val->index];
            int slot = pm->binding_indices
                       ? pm->binding_indices[val->index]
                       : in->binding;
            switch (in->type) {
            case RA_VARTYPE_TEX: {
                struct ra_tex *t = *(struct ra_tex **)val->data;
                struct ra_tex_metal *tm = t->priv;
                [cenc setTexture:tm->texture atIndex:slot];
                [cenc setSamplerState:
                        ra_metal_sampler(ra, t->params.src_linear,
                                             t->params.src_repeat)
                              atIndex:slot];
                break;
            }
            case RA_VARTYPE_IMG_W: {
                struct ra_tex *t = *(struct ra_tex **)val->data;
                struct ra_tex_metal *tm = t->priv;
                [cenc setTexture:tm->texture
                         atIndex:slot + kMetalStorageImageSlotOffset];
                break;
            }
            case RA_VARTYPE_BUF_RO:
            case RA_VARTYPE_BUF_RW: {
                struct ra_buf *b = *(struct ra_buf **)val->data;
                struct ra_buf_metal *bm = b->priv;
                [cenc setBuffer:bm->buffer offset:0 atIndex:slot];
                bm->last_used_cb = cb;
                break;
            }
            case RA_VARTYPE_FLOAT:
            case RA_VARTYPE_INT:
                /* Packed into uniform_block above. */
                break;
            default:
                MP_WARN(ra, "metal: unhandled input vartype %d\n",
                        (int)in->type);
                break;
            }
        }

        if (have_uniform_block) {
            [cenc setBytes:uniform_block
                    length:uniform_block_size
                   atIndex:kMetalGlobalUniformSlot];
        }

        /* TODO(phase-3): if compute_threads_per_group is zero (i.e. not
         * yet wired in renderpass_create's SPIRV-Cross reflection), fall
         * back to a 1x1x1 group so the dispatch is still legal but the
         * shader will likely under-utilise the GPU. */
        MTLSize tpg = pm->compute_threads_per_group;
        if (tpg.width == 0)
            tpg = MTLSizeMake(1, 1, 1);
        MTLSize groups = MTLSizeMake((NSUInteger)params->compute_groups[0],
                                     (NSUInteger)params->compute_groups[1],
                                     (NSUInteger)params->compute_groups[2]);
        [cenc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
        [cenc endEncoding];
        break;
    }

    default:
        MP_ERR(ra, "metal: renderpass_run: invalid pass type %d\n",
               (int)pass->params.type);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Timers                                                             */
/* ------------------------------------------------------------------ */

/*
 * Metal GPU timing requires MTLCounterSampleBuffer (iOS 14.2 / macOS
 * 11+). The wiring is non-trivial: query device->counterSets for
 * MTLCommonCounterSetTimestamp, allocate a sample buffer per timer,
 * call sampleCountersInBuffer:atSampleIndex:withBarrier: at encoder
 * boundaries, then resolve at frame end. Deferred -- the rest of the
 * backend works fine without it (mpv uses timer data only for the
 * --vo-gpu-stats overlay).
 */
ra_timer *metal_timer_create(struct ra *ra)
{
    (void)ra;
    /* TODO(phase-2): MTLCounterSampleBuffer wiring. */
    return NULL;
}

void metal_timer_destroy(struct ra *ra, ra_timer *timer)
{
    (void)ra;
    (void)timer;
    /* TODO(phase-2): release counter sample buffer. */
}

void metal_timer_start(struct ra *ra, ra_timer *timer)
{
    (void)ra;
    (void)timer;
    /* TODO(phase-2): sampleCountersInBuffer at encoder start. */
}

uint64_t metal_timer_stop(struct ra *ra, ra_timer *timer)
{
    (void)ra;
    (void)timer;
    /* TODO(phase-2): resolve counter sample buffer, return ns. */
    return 0;
}

/* ------------------------------------------------------------------ */
/* Debug marker                                                       */
/* ------------------------------------------------------------------ */

/*
 * Forwards the breadcrumb to the currently-open render encoder via
 * insertDebugSignpost: when one is live (set by metal_renderpass_run
 * around its encoder block); otherwise logs to mpv's VO trace stream so
 * markers issued between passes still show up under
 * --msg-level=vo=v. Compute encoders are short-lived and we don't track
 * them on ra_metal, so markers issued during compute fall through to the
 * log path -- acceptable since Xcode's GPU capture still shows
 * compute-encoder boundaries on its own.
 */
void metal_debug_marker(struct ra *ra, const char *msg)
{
    struct ra_metal *m = ra_metal_get(ra);
    const char *s = msg ? msg : "";
    if (m && m->active_render_enc) {
        [m->active_render_enc
            insertDebugSignpost:[NSString stringWithUTF8String:s]];
        return;
    }
    MP_VERBOSE(ra, "[marker] %s\n", s);
}

/* ------------------------------------------------------------------ */
/* ra_fns slice                                                       */
/* ------------------------------------------------------------------ */

/*
 * Only the entries owned by this chunk are listed. The init chunk
 * assembles the full ra_fns_metal table by combining these with the
 * tex/buf/clear/blit/destroy entries it owns. Naming convention matches
 * ra_fns_gl in ra_gl.c:1186.
 */
const struct ra_fns ra_fns_metal_renderpass = {
    .uniform_layout     = std140_layout,
    .desc_namespace     = metal_desc_namespace,
    .renderpass_create  = metal_renderpass_create,
    .renderpass_destroy = metal_renderpass_destroy,
    .renderpass_run     = metal_renderpass_run,
    .timer_create       = metal_timer_create,
    .timer_destroy      = metal_timer_destroy,
    .timer_start        = metal_timer_start,
    .timer_stop         = metal_timer_stop,
    .debug_marker       = metal_debug_marker,
};


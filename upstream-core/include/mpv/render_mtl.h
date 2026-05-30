/* Copyright (C) 2026 the mpv developers
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef MPV_CLIENT_API_RENDER_MTL_H_
#define MPV_CLIENT_API_RENDER_MTL_H_

#include "render.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Metal backend
 * -------------
 *
 * This header defines the Apple Metal backend for the mpv render API
 * (render.h). It targets iOS / macOS and is intended as a replacement for
 * the deprecated OpenGL ES path on iOS.
 *
 * Unlike the OpenGL backend, there is no implicit per-thread context — Metal
 * is fully explicit. The API user provides an `id<MTLDevice>` at init time;
 * mpv allocates command queues, libraries, pipeline states and buffers from
 * that device.
 *
 * Shaders
 * -------
 *
 * mpv's renderer (vo_gpu / vo_gpu_next) emits Vulkan-flavoured GLSL. The
 * Metal backend compiles that to SPIR-V via shaderc (already a build
 * dependency for the Vulkan-only RA) and then translates SPIR-V to MSL via
 * SPIRV-Cross (already a build dependency for the D3D11 RA). Pipeline state
 * objects are cached per renderpass.
 *
 * API use
 * -------
 *
 * Use mpv_render_context_create() with:
 *   - MPV_RENDER_PARAM_API_TYPE = MPV_RENDER_API_TYPE_METAL
 *   - MPV_RENDER_PARAM_METAL_INIT_PARAMS = pointer to a populated
 *     `mpv_metal_init_params` describing the `id<MTLDevice>` to use.
 *
 * Call mpv_render_context_render() with MPV_RENDER_PARAM_METAL_DRAWABLE to
 * render the video frame into a Metal texture (typically the `texture`
 * property of a CAMetalDrawable obtained from a CAMetalLayer).
 *
 * Hardware decoding
 * -----------------
 *
 * VideoToolbox hardware-decoded surfaces (CVPixelBuffer) are imported into
 * Metal via CVMetalTextureCache. The backend handles this transparently when
 * `hwdec=videotoolbox` is selected.
 */

/**
 * For MPV_RENDER_PARAM_API_TYPE. The corresponding init-params struct is
 * mpv_metal_init_params.
 */
#define MPV_RENDER_API_TYPE_METAL "metal"

/**
 * For MPV_RENDER_PARAM_METAL_INIT_PARAMS.
 *
 * The `device` field is an `id<MTLDevice>` opaque pointer. The lifetime of
 * the device must exceed the lifetime of the mpv_render_context.
 */
typedef struct mpv_metal_init_params {
    /**
     * Opaque pointer to an `id<MTLDevice>`. Bridge-cast from Objective-C with
     * `(__bridge void *)device`, or `(__bridge_retained void *)device` if the
     * caller wants mpv to own a reference. Required.
     */
    void *device;

    /**
     * Optional opaque pointer to an `id<MTLCommandQueue>`. If NULL, mpv
     * creates its own queue from `device`. Provide one to share scheduling
     * with other Metal work in the host app.
     */
    void *command_queue;
} mpv_metal_init_params;

/**
 * For MPV_RENDER_PARAM_METAL_DRAWABLE. Describes the destination texture
 * for a single mpv_render_context_render() call.
 */
typedef struct mpv_metal_drawable {
    /**
     * Opaque pointer to an `id<MTLTexture>` that mpv will render into.
     * Typically the `.texture` property of a `CAMetalDrawable` acquired
     * from `[CAMetalLayer nextDrawable]`. The texture's `pixelFormat`
     * must be color-renderable; mpv supports BGRA8Unorm, RGBA8Unorm,
     * BGRA10A2Unorm and RGBA16Float.
     */
    void *texture;

    /**
     * Valid dimensions of the texture in pixels. Both must be > 0.
     */
    int w, h;

    /**
     * Optional `id<MTLCommandBuffer>` opaque pointer. If non-NULL, mpv
     * encodes its render commands into this buffer (so the host app can
     * batch present + other work on a single buffer + commit once). If
     * NULL, mpv allocates a fresh command buffer per render call.
     */
    void *command_buffer;
} mpv_metal_drawable;

#ifdef __cplusplus
}
#endif

#endif

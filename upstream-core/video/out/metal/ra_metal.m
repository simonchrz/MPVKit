/* Copyright (C) 2026 the mpv developers
 *
 * Master file for mpv's Apple Metal RA backend. Defines the public
 * `ra_fns_metal` vtable that aggregates the function-pointer slots
 * implemented across ra_metal_init.m, ra_metal_textures.m and
 * ra_metal_renderpass.m.
 */

#import "ra_metal.h"
#include "video/out/gpu/utils.h"

/* Prototypes for the non-static functions defined in the three chunks. */

/* From ra_metal_textures.m */
void           metal_tex_destroy(struct ra *ra, struct ra_tex *tex);
struct ra_tex *metal_tex_create(struct ra *ra, const struct ra_tex_params *params);
bool           metal_tex_upload(struct ra *ra, const struct ra_tex_upload_params *params);
bool           metal_tex_download(struct ra *ra, struct ra_tex_download_params *params);
void           metal_buf_destroy(struct ra *ra, struct ra_buf *buf);
struct ra_buf *metal_buf_create(struct ra *ra, const struct ra_buf_params *params);
void           metal_buf_update(struct ra *ra, struct ra_buf *buf, ptrdiff_t offset,
                                const void *data, size_t size);
bool           metal_buf_poll(struct ra *ra, struct ra_buf *buf);
void           metal_clear(struct ra *ra, struct ra_tex *dst, float color[4],
                           struct mp_rect *scissor);
void           metal_blit(struct ra *ra, struct ra_tex *dst, struct ra_tex *src,
                          struct mp_rect *dst_rc, struct mp_rect *src_rc);

/* From ra_metal_renderpass.m */
int                       metal_desc_namespace(struct ra *ra, enum ra_vartype type);
struct ra_renderpass     *metal_renderpass_create(struct ra *ra, const struct ra_renderpass_params *params);
void                      metal_renderpass_destroy(struct ra *ra, struct ra_renderpass *pass);
void                      metal_renderpass_run(struct ra *ra, const struct ra_renderpass_run_params *params);
ra_timer                 *metal_timer_create(struct ra *ra);
void                      metal_timer_destroy(struct ra *ra, ra_timer *timer);
void                      metal_timer_start(struct ra *ra, ra_timer *timer);
uint64_t                  metal_timer_stop(struct ra *ra, ra_timer *timer);
void                      metal_debug_marker(struct ra *ra, const char *msg);

/* From ra_metal_init.m (file-local helper exported just for the .destroy slot) */
extern void ra_metal_destroy_fns(struct ra *ra);

/* mpv uses std140 packing for UBOs; shared with the GL backend. */
extern struct ra_layout std140_layout(struct ra_renderpass_input *inp);
/* std430 for push constants (Vulkan layout(std430, push_constant)). */
extern struct ra_layout std430_layout(struct ra_renderpass_input *inp);

const struct ra_fns ra_fns_metal = {
    .destroy            = ra_metal_destroy_fns,
    .tex_create         = metal_tex_create,
    .tex_destroy        = metal_tex_destroy,
    .tex_upload         = metal_tex_upload,
    .tex_download       = metal_tex_download,
    .buf_create         = metal_buf_create,
    .buf_destroy        = metal_buf_destroy,
    .buf_update         = metal_buf_update,
    .buf_poll           = metal_buf_poll,
    .clear              = metal_clear,
    .blit               = metal_blit,
    .uniform_layout     = std140_layout,
    .push_constant_layout = std430_layout,
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

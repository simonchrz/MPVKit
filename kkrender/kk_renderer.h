// Interner Header des kk-Render-Pfads (Strangler Phase 1). NICHT in include/ —
// nicht Teil der öffentlichen Libkkrender-Schnittstelle, nur kkrender-intern.
#ifndef KK_RENDERER_H
#define KK_RENDERER_H

#include <libplacebo/gpu.h>
#include <libplacebo/renderer.h>

struct kk_renderer;

struct kk_renderer *kk_renderer_create(pl_log log, pl_gpu gpu);
void kk_renderer_destroy(struct kk_renderer **pkr);

// Rendert image->target über den eigenen, schmalen Pass-Graph. Liefert true,
// wenn dieser Frame vom kk-Pfad übernommen wurde; false => der Aufrufer muss
// auf pl_render_image() zurückfallen (Fall noch nicht migriert).
bool kk_render_image(struct kk_renderer *kr,
                     const struct pl_frame *image,
                     const struct pl_frame *target,
                     const struct pl_render_params *params);

#endif

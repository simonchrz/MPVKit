// Minimale öffentliche Schnittstelle des Standalone-Hybrid-Renderers.
// (Ersetzt mpvs render_mtl.h für den entkoppelten libplacebo-Render-Pfad —
// nur die kuckuck_hybrid_*-Entries, keine mpv-render.h-Abhängigkeit.)
#ifndef KK_RENDER_MTL_H
#define KK_RENDER_MTL_H

#ifdef __cplusplus
extern "C" {
#endif

// Erzeugt einen Hybrid-Render-Context auf dem gegebenen id<MTLDevice>
// (Bridge-cast als void*). NULL bei Fehler.
void *kuckuck_hybrid_create(void *mtl_device);

// Rendert einen decodierten CVPixelBuffer in das Ziel-id<MTLTexture>.
// 0 = ok, negativ = Fehler.
int kuckuck_hybrid_render(void *ctx, void *cv_pixbuf, void *target_texture);

void kuckuck_hybrid_destroy(void *ctx);

#ifdef __cplusplus
}
#endif

#endif

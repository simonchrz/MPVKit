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

// Async-Render: Encode synchron, Commit ohne Warten; done(ud, ok) feuert auf Metals
// Completion-Thread wenn der Frame fertig ist. 0 = angenommen (done kommt genau
// einmal), negativ = nichts encodet (done kommt NICHT). ok=0: GPU-Fehler oder
// ausgelassener Pass — das Ziel ist NICHT (vollständig) beschrieben, nicht anzeigen.
int kuckuck_hybrid_render_async(void *ctx, void *cv_pixbuf, void *target_texture,
                                void (*done)(void *ud, int ok), void *ud);

// Render-Env (KUCKUCK_*) setzen bzw. löschen (value NULL), unter der Render-Sperre —
// statt setenv direkt: die Frame-Pfade lesen getenv unter derselben Sperre.
void kuckuck_hybrid_setenv(const char *name, const char *value);


// PSO-Prewarm: alle Render-Kernel einmal kompilieren (gegen Erst-Frame-Hitch).
// Auf der Render-Queue rufen (gleiche Queue wie kuckuck_hybrid_render).
void kuckuck_hybrid_prewarm(void *ctx);

// Pass-Zeiten als Fenster-Mittel (Messmodus), s. kk_gpu.h kk_gpu_timings_fenster.
int kuckuck_hybrid_pass_timings_fenster(const char **namen, double *mittel, double *min, int *n, int max);
// Command-Buffer-Zeiten je Bild, IMMER an: busy (GPU-Laufzeit), wait (bis GPU-Start),
// wake (GPU-Ende bis Completion), ms, Fenster-Mittel; Rückgabe = Bilder im Fenster.
// Trennt in der Wanduhr `gpu=` der App Warten von Laufen. Takt von Arbeit trennt es
// NICHT — dafür gleiche Pixel bei 30 und 60 fps vergleichen.
int kuckuck_hybrid_cb_zeiten(double *busy, double *wait, double *wake, double *busyMin, double *busyMax);

void kuckuck_hybrid_destroy(void *ctx);

#ifdef __cplusplus
}
#endif

#endif

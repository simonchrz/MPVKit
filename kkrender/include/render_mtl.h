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

// Async-Render: Encode synchron, Commit ohne Warten; done(ud) feuert auf Metals
// Completion-Thread wenn der Frame fertig ist. 0 = angenommen (done kommt genau
// einmal), negativ = nichts encodet (done kommt NICHT).
int kuckuck_hybrid_render_async(void *ctx, void *cv_pixbuf, void *target_texture,
                                void (*done)(void *ud), void *ud);


// PSO-Prewarm: alle Render-Kernel einmal kompilieren (gegen Erst-Frame-Hitch).
// Auf der Render-Queue rufen (gleiche Queue wie kuckuck_hybrid_render).
void kuckuck_hybrid_prewarm(void *ctx);

// GPU-Dauer je Render-Pass des zuletzt abgeschlossenen Frames (Diagnose).
// Nur befuellt, wenn die Umgebung KUCKUCK_PASS_TIMING=1 gesetzt hatte, ALS der
// GPU-Kontext angelegt wurde — die Env wird einmal beim Create gelesen.
// ⚠️ Im Messmodus bekommt jeder Pass einen eigenen Metal-Encoder (Apple-GPUs
// koennen Zeitstempel nur an Encoder-Grenzen). Das bricht das Buendeln auf, die
// Summe liegt ueber dem Normalbetrieb. Verwertbar sind die ANTEILE.
// `namen` zeigt auf interne Puffer und gilt bis zum naechsten Frame.
int kuckuck_hybrid_pass_timings(const char **namen, double *ms, int max);

void kuckuck_hybrid_destroy(void *ctx);

#ifdef __cplusplus
}
#endif

#endif

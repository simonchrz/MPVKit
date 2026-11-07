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

// GPU-Dauer je Render-Pass des zuletzt abgeschlossenen Frames (Diagnose).
// Nur befuellt, wenn die Umgebung KUCKUCK_PASS_TIMING=1 gesetzt hatte, ALS der
// GPU-Kontext angelegt wurde — die Env wird einmal beim Create gelesen.
// ⚠️ Im Messmodus bekommt jeder Pass einen eigenen Metal-Encoder (Apple-GPUs
// koennen Zeitstempel nur an Encoder-Grenzen). Das bricht das Buendeln auf, die
// Summe liegt ueber dem Normalbetrieb. Verwertbar sind die ANTEILE.
// `namen` zeigt auf interne Puffer und gilt bis zum naechsten Frame.
//
// 🔑 **WOFUER DAS WERKZEUG TAUGT — UND WOFUER NICHT.** Der Encoder-Split
// benachteiligt systematisch alles, dessen Vorteil im ZUSAMMENFASSEN liegt.
// Belegt 2026-09-02: im Messmodus sah die getrennte Variante DELIN+CAS 22%
// besser aus als das fusionierte DELINCAS — im Normalbetrieb sind beide exakt
// gleich schnell (10,5 ms). Wer nur die Messwerte gelesen haette, haette eine
// funktionierende Optimierung zurueckgebaut.
//   TAUGT FUER:      welcher Pass ist teuer, wie verschiebt sich das je Geraet.
//   TAUGT NICHT FUER: ob sich eine Fusion lohnt. Das entscheidet nur ein A/B der
//                     GESAMT-Renderzeit OHNE Messmodus (hybrid.log `avg=`).
int kuckuck_hybrid_pass_timings(const char **namen, double *ms, int max);

// Deblock VOR VT-SR (2026-09-12): NV12-Decoder-Buffer -> NV12-Zielbuffer (IOSurface-
// backed, gleiche Dims), Luma über den separablen ±3-Bilateral, Chroma kopiert.
// Synchron; darf NEBEN kuckuck_hybrid_render von einer anderen Queue laufen
// (eigener GPU-Kontext). 0 = ok, negativ = nichts geschrieben.
int kuckuck_hybrid_deblock_nv12(void *ctx, void *src_pixbuf, void *dst_pixbuf);

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

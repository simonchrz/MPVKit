// kk_bench.m — misst, WOHIN die Renderzeit geht: jeder Pass einzeln.
//
// ⚠️ **Was diese Messung kann und was nicht.** Sie läuft auf der Mac-GPU. Die
// absoluten Millisekunden gelten NICHT für iPhone oder iPad — dort ist die
// einzige belastbare Quelle weiterhin die `render=`-Zeile in `hybrid.log`.
// Übertragbar sind die VERHÄLTNISSE: welcher Pass den Löwenanteil frisst und wie
// sich eine Änderung darauf auswirkt. Genau das fehlte bisher — `hybrid.log`
// nennt nur die Gesamtzeit, und `kk-passtiming.log` auf den Geräten stammt aus
// der libplacebo-Ära (Pass-Namen, die es in kk_gpu gar nicht gibt).
//
// Deshalb ist der Nutzen: eine Optimierung VORHER am Verhältnis abschätzen,
// statt sie zu bauen und dann on-device zu hoffen. Der Beweis bleibt das Gerät.
//
// Gemessen wird in der Auflösung, die der Renderer wirklich fährt: Quelle 1080p
// (bzw. 576p für den SD-Pfad), Ziel die volle Panel-Fläche eines iPhone 17 Pro
// (2360 px breit) — die Kosten skalieren mit den AUSGABE-Pixeln, nicht mit der
// Quelle (belegt 2026-07-30: 270p kostete dort noch 13 ms).
//
// Aufruf: ./run-tests.sh kk_bench   oder direkt.

#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

typedef struct { float a, b; float m[9]; } L_uniform;
typedef struct { float m[12]; } D_uniform;
typedef struct { float d[12]; float a, b; float m[9]; } DL_uniform;
typedef struct { float radius, threshold, grain; uint32_t iters, index; } DB_uniform;
typedef struct { float scale; uint32_t lutn; float radius; float lut[64]; } EWA_uniform;

static double jetzt_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

/// Misst einen Pass: Warmlauf (PSO-Compile), dann N Durchläufe, Median-nah über
/// den Mittelwert. `kk_gpu_finish` blockt bis die GPU fertig ist — ohne das
/// misst man das Absetzen des Kommandos, nicht die Arbeit.
static double miss(kk_gpu *g, const char *msl, const char *entry,
                   const kk_compute_args *args, int n) {
    kk_gpu_compute(g, msl, entry, args);
    kk_gpu_finish(g);                       // Warmlauf: PSO bauen, Caches füllen
    double t0 = jetzt_ms();
    for (int i = 0; i < n; i++) kk_gpu_compute(g, msl, entry, args);
    kk_gpu_finish(g);
    return (jetzt_ms() - t0) / n;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }

    const int SRC_W = 1920, SRC_H = 1080;       // typische HD-Quelle
    const int OUT_W = 2360, OUT_H = 1090;       // iPhone-17-Pro-Panel (Video-Anteil)
    const int SD_W  = 1040, SD_H  = 576;        // Tuner-SD (VOX/RTL/ProSieben)
    const int N = 40;

    printf("  Quelle %dx%d -> Ziel %dx%d, %d Durchlaeufe je Pass\n", SRC_W, SRC_H, OUT_W, OUT_H, N);
    printf("  (Mac-GPU — absolute Werte gelten NICHT fuers Geraet, die Verhaeltnisse schon)\n\n");

    // Texturen einmal anlegen und wiederverwenden — Allokation ist nicht Teil der
    // Messung (der Renderer hält seine Zwischenpuffer über Frames hinweg).
    kk_tex *luma   = kk_tex_create(g, SRC_W, SRC_H, KK_FMT_R8,  KK_TEX_SAMPLE|KK_TEX_DOWNLOAD, NULL);
    kk_tex *chroma = kk_tex_create(g, SRC_W/2, SRC_H/2, KK_FMT_RG8, KK_TEX_SAMPLE|KK_TEX_DOWNLOAD, NULL);
    kk_tex *rgb    = kk_tex_create(g, SRC_W, SRC_H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
    kk_tex *lin    = kk_tex_create(g, SRC_W, SRC_H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
    kk_tex *tmpx   = kk_tex_create(g, OUT_W, SRC_H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
    kk_tex *skal   = kk_tex_create(g, OUT_W, OUT_H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
    kk_tex *ziel   = kk_tex_create(g, OUT_W, OUT_H, KK_FMT_RGBA8, KK_TEX_STORAGE, NULL);
    kk_tex *sd     = kk_tex_create(g, SD_W, SD_H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);

    D_uniform D = {{ 1.1643f,0.0f,1.7927f, 1.1643f,-0.2132f,-0.5329f,
                     1.1643f,2.1124f,0.0f, -0.9729f,0.3015f,-1.1334f }};
    L_uniform L = { 0.8704f, 0.0595f, {1,0,0, 0,1,0, 0,0,1} };
    DL_uniform DL; memcpy(DL.d, D.m, sizeof DL.d); DL.a=L.a; DL.b=L.b; memcpy(DL.m, L.m, sizeof DL.m);
    DB_uniform DB = { .radius=16.0f, .threshold=0.004f, .grain=0.006f, .iters=1, .index=0 };

    double t_dec, t_lin, t_declin, t_deband, t_lanx, t_lany, t_ewa, t_delin, t_cas, t_delincas;

    printf("  --- Decode/Linearisieren (Quellaufloesung) ---\n");
    t_dec = miss(g, DEC_MSL, "dec", &(kk_compute_args){
        .out=rgb, .in={luma,chroma}, .n_in=2, .uniforms=&D, .uniforms_size=sizeof D }, N);
    t_lin = miss(g, LIN_MSL, "lin", &(kk_compute_args){
        .out=lin, .in={rgb}, .n_in=1, .uniforms=&L, .uniforms_size=sizeof L }, N);
    t_declin = miss(g, DECLIN_MSL, "declin", &(kk_compute_args){
        .out=lin, .in={luma,chroma}, .n_in=2, .uniforms=&DL, .uniforms_size=sizeof DL }, N);
    printf("  DEC              %6.3f ms\n", t_dec);
    printf("  LIN              %6.3f ms\n", t_lin);
    printf("  DECLIN (fusion)  %6.3f ms   gegen %6.3f ms getrennt = %+.0f%%\n",
           t_declin, t_dec+t_lin, 100.0*(t_declin/(t_dec+t_lin)-1.0));

    printf("\n  --- Deband (nur SD-Cartoon-Pfad) ---\n");
    t_deband = miss(g, DEBAND_MSL, "deband", &(kk_compute_args){
        .out=sd, .in={sd}, .n_in=1, .uniforms=&DB, .uniforms_size=sizeof DB }, N);
    printf("  DEBAND (576p)    %6.3f ms\n", t_deband);

    printf("\n  --- Skalieren auf Panel-Flaeche (der dominante Posten) ---\n");
    kk_lanczos_p px = kk_lanczos_params((float)OUT_W/SRC_W, 0);
    kk_lanczos_p py = kk_lanczos_params((float)OUT_H/SRC_H, 1);
    t_lanx = miss(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
        .out=tmpx, .in={lin}, .n_in=1, .uniforms=&px, .uniforms_size=sizeof px }, N);
    t_lany = miss(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
        .out=skal, .in={tmpx}, .n_in=1, .uniforms=&py, .uniforms_size=sizeof py }, N);
    EWA_uniform ew; ew.scale=(float)OUT_W/SRC_W; ew.lutn=64; ew.radius=KK_EWA_RADIUS;
    for (int i=0;i<64;i++) ew.lut[i]=KK_EWA_LUT[i];
    t_ewa = miss(g, EWA_MSL, "ewa", &(kk_compute_args){
        .out=skal, .in={lin}, .n_in=1, .uniforms=&ew, .uniforms_size=sizeof ew }, N);
    printf("  Lanczos separabel %5.3f ms  (X %.3f + Y %.3f)   <- HD-Light\n",
           t_lanx+t_lany, t_lanx, t_lany);
    printf("  EWA polar        %6.3f ms                       <- Standard\n", t_ewa);
    printf("  => HD-Light spart %.0f%% gegenueber EWA\n", 100.0*(1.0-(t_lanx+t_lany)/t_ewa));

    printf("\n  --- Ausgabe (Ziel-Aufloesung) ---\n");
    t_delin = miss(g, DELIN_MSL, "delin", &(kk_compute_args){
        .out=ziel, .in={skal}, .n_in=1 }, N);
    t_cas = miss(g, CAS_MSL, "cas", &(kk_compute_args){
        .out=ziel, .in={skal}, .n_in=1 }, N);
    t_delincas = miss(g, DELINCAS_MSL, "delincas", &(kk_compute_args){
        .out=ziel, .in={skal}, .n_in=1 }, N);
    printf("  DELIN            %6.3f ms\n", t_delin);
    printf("  CAS              %6.3f ms\n", t_cas);
    printf("  DELINCAS (fusion)%6.3f ms   gegen %6.3f ms getrennt = %+.0f%%\n",
           t_delincas, t_delin+t_cas, 100.0*(t_delincas/(t_delin+t_cas)-1.0));

    printf("\n  --- Ketten (so laeuft es wirklich) ---\n");
    double hd_light = t_declin + t_lanx + t_lany + t_delincas;
    double standard = t_dec + t_lin + t_ewa + t_delin + t_cas;
    printf("  HD-Light   DECLIN+Lanczos+DELINCAS   %6.3f ms\n", hd_light);
    printf("  Standard   DEC+LIN+EWA+DELIN+CAS     %6.3f ms\n", standard);
    printf("  => HD-Light ist %.1fx billiger\n", standard/hd_light);

    // --- Gegenprobe der Pass-Zeitmessung (KUCKUCK_PASS_TIMING=1) ---
    // Misst dieselbe Kette nochmal, diesmal von der GPU selbst gestoppt. Die
    // Summe muss ungefaehr zu den Einzelmessungen oben passen — tut sie das
    // nicht, misst die Instrumentierung etwas anderes als sie behauptet.
    if (getenv("KUCKUCK_PASS_TIMING")) {
        printf("\n  --- Gegenprobe: GPU-Zeitstempel je Pass (eine HD-Light-Kette) ---\n");
        kk_gpu_compute(g, DECLIN_MSL, "declin", &(kk_compute_args){
            .out=lin, .in={luma,chroma}, .n_in=2, .uniforms=&DL, .uniforms_size=sizeof DL });
        kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
            .out=tmpx, .in={lin}, .n_in=1, .uniforms=&px, .uniforms_size=sizeof px });
        kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
            .out=skal, .in={tmpx}, .n_in=1, .uniforms=&py, .uniforms_size=sizeof py });
        kk_gpu_compute(g, DELINCAS_MSL, "delincas", &(kk_compute_args){
            .out=ziel, .in={skal}, .n_in=1 });
        kk_gpu_finish(g);
        const char *nm[KK_TIMING_MAX]; double t[KK_TIMING_MAX];
        int n = kk_gpu_timings(g, nm, t, KK_TIMING_MAX);
        double summe = 0;
        for (int i = 0; i < n; i++) { printf("   %-10s %6.3f ms\n", nm[i], t[i]); summe += t[i]; }
        if (n == 0) printf("   (keine Werte — Geraet ohne GPU-Zeitstempel?)\n");
        else printf("   Summe     %6.3f ms   (Einzelmessung oben: %.3f ms)\n", summe, hd_light);
    }

    kk_gpu_destroy(&g);
    printf("\nkk_bench: fertig (Verhaeltnisse uebertragbar, absolute Werte nicht)\n");
    return 0;
} }

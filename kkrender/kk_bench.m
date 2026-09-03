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


    // --- Probe: halbe Praezision ---
    // Alle Produktions-Kernel rechnen in float, obwohl die Zwischentexturen RGBA16F
    // sind — die Daten liegen also schon in half, gerechnet wird in voller Breite.
    // Apple-GPUs haben fuer half den doppelten ALU-Durchsatz und halben Register-
    // druck. Ob das hier etwas bringt, haengt davon ab, ob die Paesse rechen- oder
    // speichergebunden sind; genau das misst diese Probe an den zwei teuersten
    // Paessen (Lanczos-Y und DELINCAS). Die Varianten leben NUR hier im Benchmark.
    printf("\n  --- Probe: halbe Praezision (Varianten nur im Benchmark) ---\n");
    static const char *LANCZOS_HALF_MSL =
    "#include <metal_stdlib>\nusing namespace metal;\nstruct P{float scale;uint axis;float lut[64];};\n"
    "static inline float l3lut(constant P& p, float x){ x=min(abs(x),3.0)*(63.0/3.0);\n"
    "  int i0=int(x); return mix(p.lut[i0], p.lut[min(i0+1,63)], x-float(i0)); }\n"
    "kernel void lanczosh(texture2d<half> src [[texture(0)]], texture2d<half,access::write> dst [[texture(1)]],\n"
    "  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
    "  int sw=int(src.get_width()),sh=int(src.get_height()); float coord=(p.axis==0u?float(id.x):float(id.y));\n"
    "  float sf=min(p.scale,1.0); int R=int(ceil(3.0/sf));\n"
    "  float s=(coord+0.5)/p.scale-0.5; int base=int(floor(s)); half4 acc=half4(0.0h); float wsum=0.0;\n"
    "  for(int t=1-R;t<=R;t++){ int tap=base+t; float w=l3lut(p,(s-float(tap))*sf);\n"
    "    int cx=(p.axis==0u)?clamp(tap,0,sw-1):int(id.x); int cy=(p.axis==1u)?clamp(tap,0,sh-1):int(id.y);\n"
    "    acc+=half(w)*src.read(uint2(cx,cy)); wsum+=w; } dst.write(acc/half(wsum),id);}\n";
    static const char *DELINCAS_HALF_MSL =
    "#include <metal_stdlib>\nusing namespace metal;\n#define SHARP 0.5h\n"
    "static inline half srgb(half c){ return c<=0.0031308h ? 12.92h*c : 1.055h*pow(c,half(1.0/2.4))-0.055h; }\n"
    "static inline half3 srgb3(half3 c){ c=clamp(c,0.0h,1.0h); return half3(srgb(c.r),srgb(c.g),srgb(c.b)); }\n"
    "kernel void delincash(texture2d<half> src [[texture(0)]], texture2d<half,access::write> dst [[texture(1)]],\n"
    "  uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
    "  int sw=int(W),sh=int(H); int2 p=int2(id);\n"
    "#define T(dx,dy) srgb3(src.read(uint2(clamp(p.x+(dx),0,sw-1),clamp(p.y+(dy),0,sh-1))).rgb)\n"
    "  half3 a=T(-1,-1),b=T(0,-1),c=T(1,-1),d=T(-1,0),e=T(0,0),f=T(1,0),g=T(-1,1),h=T(0,1),i=T(1,1);\n"
    "  half3 mn=min(min(min(d,e),min(f,b)),h); half3 mn2=min(mn,min(min(a,c),min(g,i))); mn+=mn2;\n"
    "  half3 mx=max(max(max(d,e),max(f,b)),h); half3 mx2=max(mx,max(max(a,c),max(g,i))); mx+=mx2;\n"
    "  half3 rcpM=1.0h/max(mx,half3(1e-3h)); half3 amp=clamp(min(mn,2.0h-mx)*rcpM,0.0h,1.0h); amp=rsqrt(max(amp,half3(1e-3h)));\n"
    "  half peak=-3.0h*SHARP+8.0h; half3 w=-1.0h/(amp*peak); half3 rcpW=1.0h/(1.0h+4.0h*w);\n"
    "  half3 win=(b+d)+(f+h); half3 o=clamp((win*w+e)*rcpW,0.0h,1.0h); dst.write(half4(o,1.0h),id);}\n";
    double t_lanx_h = miss(g, LANCZOS_HALF_MSL, "lanczosh", &(kk_compute_args){
        .out=tmpx, .in={lin}, .n_in=1, .uniforms=&px, .uniforms_size=sizeof px }, N);
    double t_lany_h = miss(g, LANCZOS_HALF_MSL, "lanczosh", &(kk_compute_args){
        .out=skal, .in={tmpx}, .n_in=1, .uniforms=&py, .uniforms_size=sizeof py }, N);
    double t_delincas_h = miss(g, DELINCAS_HALF_MSL, "delincash", &(kk_compute_args){
        .out=ziel, .in={skal}, .n_in=1 }, N);
    printf("  Lanczos X   float %6.3f  half %6.3f ms  = %+.0f%%\n", t_lanx, t_lanx_h, 100.0*(t_lanx_h/t_lanx-1.0));
    printf("  Lanczos Y   float %6.3f  half %6.3f ms  = %+.0f%%\n", t_lany, t_lany_h, 100.0*(t_lany_h/t_lany-1.0));
    printf("  DELINCAS    float %6.3f  half %6.3f ms  = %+.0f%%\n", t_delincas, t_delincas_h, 100.0*(t_delincas_h/t_delincas-1.0));
    // Auf dem Mac dauert ein Pass ~0,13 ms — das ist nah an der Dispatch-Grundlast,
    // eine ALU-Ersparnis ginge darin unter. Darum dieselbe Probe noch einmal mit
    // 4x Pixeln (UHD-Quelle -> 4720x2180), wo der Kernel selbst dominiert; das
    // Verhaeltnis dort ist das uebertragbare.
    {
        const int SW = 2*SRC_W, SH = 2*SRC_H, OW = 2*OUT_W, OH = 2*OUT_H;
        kk_tex *lin4  = kk_tex_create(g, SW, SH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        kk_tex *tmpx4 = kk_tex_create(g, OW, SH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        kk_tex *skal4 = kk_tex_create(g, OW, OH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        kk_tex *ziel4 = kk_tex_create(g, OW, OH, KK_FMT_RGBA8, KK_TEX_STORAGE, NULL);
        kk_lanczos_p px4 = kk_lanczos_params((float)OW/SW, 0);
        kk_lanczos_p py4 = kk_lanczos_params((float)OH/SH, 1);
        double fx = miss(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
            .out=tmpx4, .in={lin4}, .n_in=1, .uniforms=&px4, .uniforms_size=sizeof px4 }, N);
        double hx = miss(g, LANCZOS_HALF_MSL, "lanczosh", &(kk_compute_args){
            .out=tmpx4, .in={lin4}, .n_in=1, .uniforms=&px4, .uniforms_size=sizeof px4 }, N);
        double fy = miss(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
            .out=skal4, .in={tmpx4}, .n_in=1, .uniforms=&py4, .uniforms_size=sizeof py4 }, N);
        double hy = miss(g, LANCZOS_HALF_MSL, "lanczosh", &(kk_compute_args){
            .out=skal4, .in={tmpx4}, .n_in=1, .uniforms=&py4, .uniforms_size=sizeof py4 }, N);
        double fd = miss(g, DELINCAS_MSL, "delincas", &(kk_compute_args){ .out=ziel4, .in={skal4}, .n_in=1 }, N);
        double hd = miss(g, DELINCAS_HALF_MSL, "delincash", &(kk_compute_args){ .out=ziel4, .in={skal4}, .n_in=1 }, N);
        printf("  4x Pixel (%dx%d -> %dx%d):\n", SW, SH, OW, OH);
        printf("  Lanczos X   float %6.3f  half %6.3f ms  = %+.0f%%\n", fx, hx, 100.0*(hx/fx-1.0));
        printf("  Lanczos Y   float %6.3f  half %6.3f ms  = %+.0f%%\n", fy, hy, 100.0*(hy/fy-1.0));
        printf("  DELINCAS    float %6.3f  half %6.3f ms  = %+.0f%%\n", fd, hd, 100.0*(hd/fd-1.0));
        kk_tex_destroy(g, &lin4); kk_tex_destroy(g, &tmpx4); kk_tex_destroy(g, &skal4); kk_tex_destroy(g, &ziel4);
    }

    // --- ArtCNN auf SD: lohnt der CNN-Upscaler nach dem fp16-Umbau wieder? ---
    //
    // Vorgeschichte: der SD-Zweig lief bis 2026-06-30 auf ArtCNN_C4F16 und wurde
    // auf CAS umgestellt, weil der CNN auf dem A19 real ~22-35 ms/Frame kostete
    // (Spitzen 72 ms) und damit das 25fps-Budget riss. Seither wurde die CNN-Kette
    // auf fp16 umgebaut. Diese Messung beziffert, WIE VIEL das gebracht hat —
    // relativ zu CAS/EWA in derselben Auflösung, auf derselben GPU.
    //
    // ⚠️ kk_gpu_artcnn ist kein einzelner Kernel, sondern eine Kette (6 Conv-Stufen
    // + Depth-to-Space) mit eigenen Zwischentexturen. Deshalb nicht ueber miss(),
    // sondern die ganze Funktion takten. Die Weights liegen im App-Repo; ohne sie
    // wird der Abschnitt uebersprungen statt zu luegen.
    {
        const char *wpath = getenv("KK_ARTCNN_WEIGHTS");
        if (!wpath) wpath = "/Users/simon/Documents/Kuckuck/Kuckuck/Kuckuck/"
                            "Resources/artcnn_c4f16.weights";
        FILE *wf = fopen(wpath, "rb");
        if (!wf) {
            printf("\n  --- ArtCNN (SD) --- uebersprungen: Weights nicht gefunden (%s)\n", wpath);
            printf("      Pfad ueber KK_ARTCNN_WEIGHTS setzen.\n");
        } else {
            fclose(wf);
            printf("\n  --- ArtCNN auf SD-Luma %dx%d -> 2x (fp16-Kette) ---\n", SD_W, SD_H);
            kk_tex *sdluma = kk_tex_create(g, SD_W, SD_H, KK_FMT_R8,
                                           KK_TEX_SAMPLE|KK_TEX_DOWNLOAD, NULL);
            if (!sdluma) {
                printf("      Textur fehlgeschlagen\n");
            } else {
                kk_tex *out = kk_gpu_artcnn(g, sdluma, wpath);   // Warmlauf: PSOs + Weights
                kk_gpu_finish(g);
                if (!out) {
                    printf("      kk_gpu_artcnn lieferte NULL (Weights unbrauchbar?)\n");
                } else {
                    const int NC = 20;
                    double t0 = jetzt_ms();
                    for (int i = 0; i < NC; i++) (void)kk_gpu_artcnn(g, sdluma, wpath);
                    kk_gpu_finish(g);
                    double cnn = (jetzt_ms() - t0) / NC;
                    printf("    ArtCNN gesamt      %6.3f ms  (%dx%d -> %dx%d Luma)\n",
                           cnn, SD_W, SD_H, SD_W*2, SD_H*2);
                    printf("    zum Vergleich: CAS und EWA oben in Ziel-Aufloesung.\n");
                    printf("    ⚠️ Mac-GPU. Uebertragbar ist das VERHAELTNIS, nicht die ms.\n");
                }
                kk_gpu_artcnn_release(g);
                kk_tex_destroy(g, &sdluma);
            }
        }
    }

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

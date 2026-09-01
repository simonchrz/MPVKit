// kk_fusion_test.m — prüft die PASS-FUSION aus renderpl.70 gegen die getrennten Pässe.
//
// Worum es geht: um schneller zu werden, wurden im Juli 2026 zwei Schrittpaare zu je
// einem Kernel zusammengelegt —
//   DECLIN  = DEC (YUV→RGB) + LIN (Gamma→Linearlicht + Primärfarben)
//   DELINCAS = DELIN (Linearlicht→sRGB) + CAS (Schärfung)
// Beide tragen im Produktionscode den Kommentar „mathematisch identisch zu … in
// Serie". **Genau diese Behauptung prüft dieser Prüfstand** — sie war bisher nur
// durch Hinschauen auf dem Gerät belegt.
//
// ⚠️ Warum das die heikelste Stelle im Renderer ist: ein Fehler in einer Fusion
// stürzt nicht ab und fällt nicht auf. Er verschiebt Farben oder Helligkeit um ein
// paar Prozent — sichtbar nur im direkten Vergleich, den nach der Umstellung
// niemand mehr ziehen kann, weil beide Wege dasselbe Bild liefern SOLLEN.
//
// Erwartete Abweichung ist nicht exakt null: der getrennte Weg schreibt sein
// Zwischenergebnis in eine Textur und quantisiert dabei, der fusionierte rechnet
// durch. Der Prüfstand nutzt deshalb RGBA16F als Zwischenstufe (wie die
// Produktion) und erlaubt Halbbit-Rauschen, aber keinen systematischen Versatz.
//
// Wie kk_lut_test bindet der Prüfstand `kk_gpu_render.c` direkt ein, um an die
// `static`-Kernel zu kommen, ohne den Produktionscode dafür aufzubohren.

#import <Foundation/Foundation.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

typedef struct { float d[12]; float a, b; float m[9]; } DL_uniform;
typedef struct { float m[12]; } D_uniform;
typedef struct { float a, b; float m[9]; } L_uniform;

/// Baut ein Testbild: Luma mit Verlauf + Kanten, Chroma mit Farbwechseln.
/// Bewusst NICHT uniform — eine Fläche würde einen Matrixfehler verstecken.
static void testbild(unsigned char *luma, int lw, int lh,
                     unsigned char *chroma, int cw, int ch) {
    for (int y = 0; y < lh; y++)
        for (int x = 0; x < lw; x++)
            luma[y * lw + x] = (unsigned char)(16 + (219 * x) / (lw - 1));   // TV-Range-Rampe
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            chroma[2 * (y * cw + x) + 0] = (unsigned char)(16 + (224 * y) / (ch - 1));
            chroma[2 * (y * cw + x) + 1] = (unsigned char)(255 - (224 * x) / (cw - 1));
        }
}

static int pruefe_declin(kk_gpu *g) {
    const int W = 64, H = 32, CW = W / 2, CH = H / 2;
    unsigned char *luma = malloc(W * H), *chroma = malloc(2 * CW * CH);
    testbild(luma, W, H, chroma, CW, CH);

    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE, luma);
    kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE, chroma);

    // EXAKT die Produktions-Parameter aus kk_gpu_render.c (Fallback-Zweig, wenn der
    // Hook keine Matrix liefert): BT.709 limited-range YUV->RGB und die BT.1886-
    // Koeffizienten mit Primärfarben-Identität. Bewusst abgeschrieben statt
    // hergeleitet — der Prüfstand soll die FUSION prüfen, nicht meine Farbmathe.
    D_uniform D = {{ 1.1643f, 0.0f, 1.7927f,
                     1.1643f, -0.2132f, -0.5329f,
                     1.1643f, 2.1124f, 0.0f,
                     -0.9729f, 0.3015f, -1.1334f }};
    L_uniform L = { 0.8704f, 0.0595f, { 1,0,0, 0,1,0, 0,0,1 } };

    // --- Weg A: DEC -> LIN, getrennt, mit Zwischentextur (wie vor renderpl.70)
    kk_tex *zwischen = kk_tex_create(g, W, H, KK_FMT_RGBA16F,
                                     KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    // ⚠️ Ausgabe als RGBA8: `kk_tex_download` rechnet fest mit 4 Byte/Pixel
    // („Annahme: Download-Targets sind RGBA8", kk_gpu.m). Höhere Präzision liesse
    // sich nicht auslesen. Für den Zweck reicht es — gesucht ist ein SYSTEMATISCHER
    // Versatz, nicht das letzte Bit. Die Zwischenstufe von Weg A bleibt RGBA16F,
    // genau wie in der Produktion.
    kk_tex *outA = kk_tex_create(g, W, H, KK_FMT_RGBA8,
                                 KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_gpu_compute(g, DEC_MSL, "dec", &(kk_compute_args){
        .out = zwischen, .in = { tl, tc }, .n_in = 2, .uniforms = &D, .uniforms_size = sizeof D });
    kk_gpu_compute(g, LIN_MSL, "lin", &(kk_compute_args){
        .out = outA, .in = { zwischen }, .n_in = 1, .uniforms = &L, .uniforms_size = sizeof L });

    // --- Weg B: DECLIN, fusioniert (Produktionspfad seit renderpl.70)
    DL_uniform DL;
    memcpy(DL.d, D.m, sizeof DL.d); DL.a = L.a; DL.b = L.b; memcpy(DL.m, L.m, sizeof DL.m);
    kk_tex *outB = kk_tex_create(g, W, H, KK_FMT_RGBA8,
                                 KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_gpu_compute(g, DECLIN_MSL, "declin", &(kk_compute_args){
        .out = outB, .in = { tl, tc }, .n_in = 2, .uniforms = &DL, .uniforms_size = sizeof DL });
    kk_gpu_finish(g);

    unsigned char *a = malloc(4 * W * H), *b = malloc(4 * W * H);
    kk_tex_download(g, outA, a);
    kk_tex_download(g, outB, b);

    double maxabs = 0.0, summe = 0.0;
    int n = 0;
    for (int i = 0; i < W * H; i++) {
        for (int k = 0; k < 3; k++) {
            double fa = a[4 * i + k] / 255.0, fb = b[4 * i + k] / 255.0;
            maxabs = fmax(maxabs, fabs(fa - fb));
            summe += fa - fb;      // Vorzeichenbehaftet: verrät systematischen Versatz
            n++;
        }
    }
    double mittel = summe / n;
    printf("  DECLIN vs DEC->LIN     maxdiff=%.1f LSB  mittlerer Versatz=%+.3f LSB", maxabs*255.0, mittel*255.0);
    // maxdiff darf Halbbit-Rauschen der RGBA16F-Zwischenstufe sein (~1e-3 im
    // Linearlicht). Ein systematischer Versatz wäre ein echter Mathefehler.
    // 1 LSB Rundungsunterschied ist erlaubt (Weg A quantisiert einmal mehr).
    // Ein mittlerer Versatz über 0,2 LSB wäre dagegen systematisch = Mathefehler.
    int schlecht = (maxabs > 1.5 / 255.0) || (fabs(mittel) > 0.2 / 255.0);
    printf("%s\n", schlecht ? "   FEHLER" : "   ok");

    free(luma); free(chroma); free(a); free(b);
    kk_tex_destroy(g, &tl); kk_tex_destroy(g, &tc); kk_tex_destroy(g, &zwischen);
    kk_tex_destroy(g, &outA); kk_tex_destroy(g, &outB);
    return schlecht;
}

/// DELINCAS (Delin+CAS fusioniert) gegen DELIN -> CAS in Serie.
///
/// ⚠️ Hier ist „identische Mathe" eine STÄRKERE Behauptung als bei DECLIN: der
/// getrennte Weg schreibt das sRGB-Ergebnis in eine RGBA8-Textur und schärft
/// DANACH auf den quantisierten Werten. Der fusionierte Kernel encodiert pro
/// Abtastpunkt in voller Präzision und schärft auf den unquantisierten Werten.
/// Die Fusion ist also nicht nur schneller, sondern genauer — erwartbar ist ein
/// kleiner Unterschied. Was NICHT sein darf, ist ein systematischer Versatz:
/// der hiesse, dass Bilder seit renderpl.70 heller oder dunkler sind.
static int pruefe_delincas(kk_gpu *g) {
    const int W = 64, H = 32;
    // Linearlicht-Eingang mit Kanten (CAS ist ein Kantenfilter — eine Fläche
    // würde jeden Unterschied verstecken).
    unsigned char *lin = malloc(4 * W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int hell = ((x / 5 + y / 5) % 2);
            unsigned char v = hell ? 200 : 40;
            if (x % 5 == 0) v = hell ? 240 : 10;      // harte Kanten dazwischen
            lin[4*(y*W+x)+0] = v;
            lin[4*(y*W+x)+1] = (unsigned char)(v * 0.8);
            lin[4*(y*W+x)+2] = (unsigned char)(v * 0.6);
            lin[4*(y*W+x)+3] = 255;
        }
    kk_tex *src = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_SAMPLE, lin);

    // --- Weg A: DELIN -> CAS (getrennt, mit RGBA8-Zwischenstufe wie vor renderpl.70)
    kk_tex *srgb = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    kk_tex *outA = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_gpu_compute(g, DELIN_MSL, "delin", &(kk_compute_args){
        .out = srgb, .in = { src }, .n_in = 1 });
    kk_gpu_compute(g, CAS_MSL, "cas", &(kk_compute_args){
        .out = outA, .in = { srgb }, .n_in = 1 });

    // --- Weg B: DELINCAS (Produktionspfad im HD-Light)
    kk_tex *outB = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_gpu_compute(g, DELINCAS_MSL, "delincas", &(kk_compute_args){
        .out = outB, .in = { src }, .n_in = 1 });
    kk_gpu_finish(g);

    unsigned char *a = malloc(4 * W * H), *b = malloc(4 * W * H);
    kk_tex_download(g, outA, a);
    kk_tex_download(g, outB, b);

    double maxabs = 0.0, summe = 0.0; int n = 0;
    for (int i = 0; i < W * H; i++)
        for (int k = 0; k < 3; k++) {
            double fa = a[4*i+k] / 255.0, fb = b[4*i+k] / 255.0;
            maxabs = fmax(maxabs, fabs(fa - fb));
            summe += fa - fb; n++;
        }
    double mittel = summe / n;
    printf("  DELINCAS vs DELIN->CAS maxdiff=%.1f LSB  mittlerer Versatz=%+.3f LSB",
           maxabs * 255.0, mittel * 255.0);
    // Grosszügiger als bei DECLIN: die eingesparte Quantisierung wirkt sich an
    // Kanten aus, wo CAS die Nachbarn gewichtet. Der Versatz muss trotzdem ~0 sein.
    int schlecht = (maxabs > 4.0 / 255.0) || (fabs(mittel) > 0.3 / 255.0);
    printf("%s\n", schlecht ? "   FEHLER" : "   ok");

    free(lin); free(a); free(b);
    kk_tex_destroy(g, &src); kk_tex_destroy(g, &srgb);
    kk_tex_destroy(g, &outA); kk_tex_destroy(g, &outB);
    return schlecht;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }

    int fehler = pruefe_declin(g);
    fehler |= pruefe_delincas(g);

    kk_gpu_destroy(&g);
    printf(fehler ? "kk_fusion_test: FEHLGESCHLAGEN\n"
                  : "kk_fusion_test: Pass-Fusion identisch zu getrennten Paessen  PASS\n");
    return fehler ? 1 : 0;
} }

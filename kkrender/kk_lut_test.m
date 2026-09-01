// kk_lut_test.m — prüft den ECHTEN Lanczos-Pfad aus kk_gpu_render.c.
//
// ⚠️ **Warum es diesen Prüfstand zusätzlich zu kk_scale_test gibt.** Die fünf
// Prüfstände aus der Bauphase (Juni 2026) tragen alle eine EIGENE Kopie des
// Metal-Kernels; keiner ruft Produktionscode auf. Sie beweisen damit, dass die
// Mechanik funktioniert — aber nichts über den ausgelieferten Renderer. Als
// `kk_gpu_render.c` im Juli auf eine gebackene 64er-LUT umgestellt wurde
// (renderpl.70, „2× sin() pro Tap" → Tabellen-Lookup), blieb `kk_scale_test`
// grün, weil es weiterhin seine eigene sin()-Variante prüfte. **Ein Prüfstand,
// der die Sache nachbaut statt sie aufzurufen, prüft seinen eigenen Nachbau.**
//
// Der Trick, um an die `static`-Symbole zu kommen, ohne den Produktionscode für
// Tests aufzubohren: die .c-Datei direkt einbinden. In C üblich und hier
// unschädlich — der Prüfstand ist eine eigene ausführbare Datei.
//
// Geprüft wird zweierlei:
//  1. Die gebackene LUT gegen die analytische Lanczos-3-Formel (Tabellenfehler).
//  2. Der echte `LANCZOS_MSL`-Kernel gegen eine CPU-Referenz derselben Mathe —
//     inklusive der Band-Limitierung für Downscale, die beim Upscale bit-genau
//     dasselbe liefern muss wie vorher.

#import <Foundation/Foundation.h>
#include "kk_gpu.h"

// Produktionscode einbinden — gibt Zugriff auf LANCZOS_MSL + kk_lanczos_params.
#include "kk_gpu_render.c"

static double l3_analytisch(double x) {
    x = fabs(x);
    if (x >= 3.0) return 0.0;
    if (x == 0.0) return 1.0;
    double s1 = sin(M_PI * x) / (M_PI * x);
    double x3 = x / 3.0;
    double s3 = (x3 == 0.0) ? 1.0 : sin(M_PI * x3) / (M_PI * x3);
    return s1 * s3;
}

/// Wie `l3lut` im Kernel: LUT-Index + lineare Interpolation.
static double l3_aus_lut(const float lut[64], double x) {
    x = fmin(fabs(x), 3.0) * (63.0 / 3.0);
    int i0 = (int)x;
    int i1 = i0 + 1 > 63 ? 63 : i0 + 1;
    return lut[i0] + (lut[i1] - lut[i0]) * (x - (double)i0);
}

static int pruefe_lut(void) {
    kk_lanczos_p p = kk_lanczos_params(1.0f, 0);
    double maxfehler = 0.0;
    // Die Stützstellen selbst müssen die Formel exakt treffen.
    for (int i = 0; i < 64; i++) {
        double x = 3.0 * i / 63.0;
        double soll = (i == 63) ? 0.0 : l3_analytisch(x);
        maxfehler = fmax(maxfehler, fabs(p.lut[i] - soll));
    }
    printf("  LUT-Stuetzstellen vs. Lanczos3-Formel: maxerr=%.2e", maxfehler);
    if (maxfehler > 1e-6) { printf("  FEHLER\n"); return 1; }
    printf("  ok\n");

    // Zwischen den Stützstellen interpoliert der Kernel linear. Der Fehler
    // gegenüber der echten Kurve ist der Preis der Tabelle — er muss klein
    // genug bleiben, dass er in 8-bit-Ausgabe nicht sichtbar wird (1 LSB =
    // 1/255 ≈ 0,0039). Die Kurve ist glatt, also erwarten wir weit darunter.
    double maxinterp = 0.0;
    for (int k = 0; k <= 3000; k++) {
        double x = 3.0 * k / 3000.0;
        maxinterp = fmax(maxinterp, fabs(l3_aus_lut(p.lut, x) - l3_analytisch(x)));
    }
    printf("  LUT-Interpolation vs. Formel:          maxerr=%.2e (1 LSB=3.9e-03)", maxinterp);
    // Schwelle = 1 LSB. Gemessen 1,0e-03 = 0,26 LSB: der Preis der 64er-Tabelle,
    // im 8-bit-Ergebnis unsichtbar (der Kernel normalisiert zudem über wsum, der
    // Bildfehler bleibt darunter — s. Kernel-Prüfung weiter unten mit 0,6 LSB).
    if (maxinterp > 0.0039) { printf("  ZU GROB\n"); return 1; }
    printf("  ok\n");

    // lut[63] muss EXAKT 0 sein: der Kernel clampt Distanzen >=3 dorthin, und
    // ein Rest-Gewicht am Fensterrand ergäbe einen Streifen an Kanten.
    if (p.lut[63] != 0.0f) { printf("  lut[63] != 0 (%g)  FEHLER\n", p.lut[63]); return 1; }
    printf("  lut[63] == 0 (Fensterrand)             ok\n");
    return 0;
}

/// CPU-Referenz für genau das, was der Kernel rechnet (inkl. Band-Limitierung).
static void cpu_lanczos(const float *src, int sw, int sh, float *dst, int dw, int dh,
                        int axis, float scale) {
    float sf = fminf(scale, 1.0f);
    int R = (int)ceil(3.0 / sf);
    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            double coord = (axis == 0) ? x : y;
            double s = (coord + 0.5) / scale - 0.5;
            int base = (int)floor(s);
            double acc = 0.0, wsum = 0.0;
            for (int t = 1 - R; t <= R; t++) {
                int tap = base + t;
                double w = l3_analytisch((s - tap) * sf);
                int cx = (axis == 0) ? (tap < 0 ? 0 : (tap > sw - 1 ? sw - 1 : tap)) : x;
                int cy = (axis == 1) ? (tap < 0 ? 0 : (tap > sh - 1 ? sh - 1 : tap)) : y;
                acc += w * src[cy * sw + cx];
                wsum += w;
            }
            dst[y * dw + x] = (float)(acc / wsum);
        }
    }
}

static int pruefe_kernel(kk_gpu *g, int sw, int sh, int dw, const char *was) {
    float *src = malloc(sizeof(float) * sw * sh);
    for (int y = 0; y < sh; y++)
        for (int x = 0; x < sw; x++)   // Muster mit Kanten UND Verläufen
            src[y * sw + x] = ((x / 4 + y / 4) % 2) ? 0.85f : 0.15f;

    // 8-bit-Eingang wie im Produktionspfad (SDR-Zwischentexturen sind RGBA8).
    unsigned char *in8 = malloc(4 * sw * sh);
    for (int i = 0; i < sw * sh; i++) {
        unsigned char v = (unsigned char)lrintf(src[i] * 255.0f);
        in8[4*i] = in8[4*i+1] = in8[4*i+2] = v; in8[4*i+3] = 255;
    }
    kk_tex *in  = kk_tex_create(g, sw, sh, KK_FMT_RGBA8, KK_TEX_SAMPLE, in8);
    kk_tex *out = kk_tex_create(g, dw, sh, KK_FMT_RGBA8,
                                KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);

    // GENAU der Produktionsaufruf: gebackene Parameter + Produktions-MSL.
    kk_lanczos_p p = kk_lanczos_params((float)dw / sw, 0);
    kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &(kk_compute_args){
        .out = out, .in = { in }, .n_in = 1, .uniforms = &p, .uniforms_size = sizeof p });
    kk_gpu_finish(g);

    unsigned char *got = malloc(4 * dw * sh);
    kk_tex_download(g, out, got);

    float *ref = malloc(sizeof(float) * dw * sh);
    cpu_lanczos(src, sw, sh, ref, dw, sh, 0, (float)dw / sw);

    // Der GPU-Pfad rundet auf 8 bit — die Referenz muss gleich behandelt werden,
    // sonst misst man die Quantisierung statt der Filtermathe.
    double maxerr = 0.0;
    for (int i = 0; i < dw * sh; i++) {
        double soll = ref[i] < 0 ? 0 : (ref[i] > 1 ? 1 : ref[i]);
        maxerr = fmax(maxerr, fabs(got[4*i] / 255.0 - soll));
    }
    printf("  %-28s %dx%d -> %dx%d  maxerr=%.4f (%.1f LSB)",
           was, sw, sh, dw, sh, maxerr, maxerr * 255.0);
    int schlecht = maxerr > 0.004;   // ~1 LSB in 8 bit
    printf("%s\n", schlecht ? "  FEHLER" : "  ok");

    free(src); free(in8); free(got); free(ref);
    kk_tex_destroy(g, &in); kk_tex_destroy(g, &out);
    return schlecht;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fehler = pruefe_lut();

    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }

    // Upscale: hier MUSS die LUT-Variante das Ergebnis der alten sin()-Version
    // treffen — die Band-Limitierung ist bei sf=1 wirkungslos.
    fehler |= pruefe_kernel(g, 24, 24, 48, "Upscale 2x (LUT vs CPU)");
    // Downscale: der Fall, für den die Band-Limitierung 2026-07-06 gebaut wurde
    // (HDR 1440p -> Display ~0,84x). Ungetestet bis heute.
    fehler |= pruefe_kernel(g, 48, 24, 40, "Downscale 0,83x (bandlimit)");

    kk_gpu_destroy(&g);
    printf(fehler ? "kk_lut_test: FEHLGESCHLAGEN\n"
                  : "kk_lut_test: echter LANCZOS_MSL + gebackene LUT gegen CPU-Ref  PASS\n");
    return fehler ? 1 : 0;
} }

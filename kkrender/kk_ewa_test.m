// kk_ewa_test.m — prüft den EWA-lanczossharp-Scaler aus kk_gpu_render.c.
//
// EWA ist der **meistgenutzte** Weg im Renderer: alles außer HD-Light läuft
// darüber (SD-Tuner und Aufnahmen auf tier=high, Cartoon-Post-Scale, HDR vor
// renderpl.69). Trotzdem war er bisher durch keinen Prüfstand abgedeckt.
//
// ⚠️ **Warum hier keine CPU-Referenz steht wie bei Lanczos.** Die 64er-Filter-LUT
// stammt aus libplacebos `pl_filter_generate` und ist eingebacken — seit dem
// libplacebo-Drop (2026-06-25) gibt es nichts mehr, wogegen man sie nachrechnen
// könnte. Eine „Referenz", die ich aus derselben Tabelle bilde, würde nur sich
// selbst bestätigen (genau der Fehler, den kk_scale_test seit Juni macht).
//
// Stattdessen werden EIGENSCHAFTEN geprüft, die jeder korrekte Resampler hat und
// die bei den typischen Fehlern brechen — falsche LUT, falscher Radius, fehlende
// Normalisierung, Indexfehler:
//
//   1. Flächentreue: eine einfarbige Fläche muss einfarbig bleiben. Bricht bei
//      jedem Normalisierungsfehler (wsum) und bei falscher LUT-Adressierung.
//   2. Identität: Skalierung 1:1 muss das Bild nahezu unverändert lassen.
//      Bricht bei Radius- oder Zentrierungsfehlern (das klassische Halbpixel).
//   3. Wertebereich: kein Ergebnis darf unter das Minimum oder über das Maximum
//      der Quelle schießen. EWA-lanczossharp hat Antiring; ohne funktionierende
//      LUT gäbe es sichtbare Überschwinger an Kanten.
//   4. Monotonie: eine monoton steigende Rampe bleibt beim Hochskalieren
//      monoton. Bricht bei vertauschten Gewichten.

#import <Foundation/Foundation.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

typedef struct { float scale; uint32_t lutn; float radius; float lut[64]; } EWA_uniform;

static EWA_uniform ewa_params(float scale) {
    EWA_uniform e;
    e.scale = scale; e.lutn = 64; e.radius = KK_EWA_RADIUS;
    for (int i = 0; i < 64; i++) e.lut[i] = KK_EWA_LUT[i];
    return e;
}

/// Skaliert ein RGBA8-Bild mit dem ECHTEN EWA-Kernel.
static void ewa_lauf(kk_gpu *g, const unsigned char *in, int sw, int sh,
                     unsigned char *out, int dw, int dh) {
    kk_tex *src = kk_tex_create(g, sw, sh, KK_FMT_RGBA8, KK_TEX_SAMPLE, (void *)in);
    kk_tex *dst = kk_tex_create(g, dw, dh, KK_FMT_RGBA8,
                                KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    EWA_uniform e = ewa_params((float)dw / sw);
    kk_gpu_compute(g, EWA_MSL, "ewa", &(kk_compute_args){
        .out = dst, .in = { src }, .n_in = 1, .uniforms = &e, .uniforms_size = sizeof e });
    kk_gpu_finish(g);
    kk_tex_download(g, dst, out);
    kk_tex_destroy(g, &src); kk_tex_destroy(g, &dst);
}

static int pruefe_flaeche(kk_gpu *g) {
    const int SW = 32, SH = 32, DW = 64, DH = 64;
    unsigned char *in = malloc(4 * SW * SH), *out = malloc(4 * DW * DH);
    for (int i = 0; i < SW * SH; i++) {
        in[4*i] = 137; in[4*i+1] = 82; in[4*i+2] = 201; in[4*i+3] = 255;
    }
    ewa_lauf(g, in, SW, SH, out, DW, DH);

    int maxabw = 0;
    for (int i = 0; i < DW * DH; i++) {
        int d0 = abs((int)out[4*i]   - 137);
        int d1 = abs((int)out[4*i+1] -  82);
        int d2 = abs((int)out[4*i+2] - 201);
        maxabw = d0 > maxabw ? d0 : maxabw;
        maxabw = d1 > maxabw ? d1 : maxabw;
        maxabw = d2 > maxabw ? d2 : maxabw;
    }
    printf("  Flaechentreue (uni 2x)    maxabw=%d LSB", maxabw);
    int schlecht = maxabw > 1;
    printf("%s\n", schlecht ? "   FEHLER (Normalisierung?)" : "   ok");
    free(in); free(out);
    return schlecht;
}

static int pruefe_identitaet(kk_gpu *g) {
    const int W = 48, H = 32;
    unsigned char *in = malloc(4 * W * H), *out = malloc(4 * W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char v = ((x / 6 + y / 6) % 2) ? 210 : 45;
            in[4*(y*W+x)] = in[4*(y*W+x)+1] = in[4*(y*W+x)+2] = v;
            in[4*(y*W+x)+3] = 255;
        }
    ewa_lauf(g, in, W, H, out, W, H);   // scale = 1.0

    double summe = 0.0; int maxabw = 0;
    for (int i = 0; i < W * H; i++) {
        int d = abs((int)out[4*i] - (int)in[4*i]);
        maxabw = d > maxabw ? d : maxabw;
        summe += (int)out[4*i] - (int)in[4*i];
    }
    double versatz = summe / (W * H);
    // 1:1 ist bei einem schärfenden Kernel nicht bitgenau — lanczossharp hebt
    // Kanten leicht an. Ein VERSATZ über das ganze Bild wäre dagegen ein Zeichen
    // für eine falsche Zentrierung (das klassische Halbpixel).
    printf("  Identitaet (scale 1.0)    maxabw=%d LSB  Versatz=%+.3f LSB", maxabw, versatz);
    int schlecht = (maxabw > 40) || (fabs(versatz) > 1.0);
    printf("%s\n", schlecht ? "   FEHLER (Zentrierung?)" : "   ok");
    free(in); free(out);
    return schlecht;
}

static int pruefe_bereich_und_monotonie(kk_gpu *g) {
    const int SW = 24, SH = 8, DW = 96, DH = 32;
    unsigned char *in = malloc(4 * SW * SH), *out = malloc(4 * DW * DH);
    for (int y = 0; y < SH; y++)
        for (int x = 0; x < SW; x++) {
            unsigned char v = (unsigned char)(20 + (200 * x) / (SW - 1));   // monotone Rampe
            in[4*(y*SW+x)] = in[4*(y*SW+x)+1] = in[4*(y*SW+x)+2] = v;
            in[4*(y*SW+x)+3] = 255;
        }
    ewa_lauf(g, in, SW, SH, out, DW, DH);

    // Wertebereich: nichts darf über die Quell-Extreme hinausschiessen.
    int unter = 0, ueber = 0;
    for (int i = 0; i < DW * DH; i++) {
        if (out[4*i] < 20 - 2)  unter++;
        if (out[4*i] > 220 + 2) ueber++;
    }
    printf("  Wertebereich (Rampe 4x)   unterschwinger=%d  ueberschwinger=%d", unter, ueber);
    int schlecht = (unter + ueber) > 0;
    printf("%s\n", schlecht ? "   FEHLER (Antiring?)" : "   ok");

    // Monotonie in der mittleren Zeile (Ränder ausgenommen — dort clampt der Kernel).
    int y = DH / 2, brueche = 0;
    for (int x = 5; x < DW - 6; x++)
        if ((int)out[4*(y*DW+x+1)] + 1 < (int)out[4*(y*DW+x)]) brueche++;
    printf("  Monotonie (Rampe 4x)      Brueche=%d", brueche);
    schlecht |= brueche > 0;
    printf("%s\n", brueche ? "   FEHLER (Gewichte vertauscht?)" : "   ok");

    free(in); free(out);
    return schlecht;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }

    int fehler = pruefe_flaeche(g);
    fehler |= pruefe_identitaet(g);
    fehler |= pruefe_bereich_und_monotonie(g);

    kk_gpu_destroy(&g);
    printf(fehler ? "kk_ewa_test: FEHLGESCHLAGEN\n"
                  : "kk_ewa_test: EWA-Scaler-Invarianten (Flaeche/Identitaet/Bereich/Monotonie)  PASS\n");
    return fehler ? 1 : 0;
} }

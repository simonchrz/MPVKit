// kk_hdr_test.m — prüft die HDR-Kette (Tonemap-LUT, IPT-Matrizen, CMHDR-Kernel).
//
// Der HDR-Pfad hat die komplexeste Mathematik im Renderer: PQ-Kurven, vier
// 3×3-Matrizen im IPT-Farbraum, eine bt2390-Tonemap-Tabelle und eine
// Chroma-Hülle. Bis heute war davon nichts abgedeckt — verifiziert wurde durch
// Hinschauen auf einem HDR-Video.
//
// ⚠️ Wie beim EWA-Scaler gibt es keine externe Referenz mehr (die IPT-Mathe kam
// aus libplacebo, das seit 2026-06-25 draußen ist). Geprüft werden deshalb
// Invarianten, die bei den typischen Fehlern brechen — vertauschte Matrizen,
// falsch herum multipliziert, kaputte LUT-Adressierung:
//
//   1. Matrizen-Rundlauf: rgb2lms→lms2rgb und lms2ipt→ipt2lms müssen je die
//      Identität ergeben. Fängt vertauschte oder transponierte Matrizen sofort.
//   2. Tonemap-Kurve: monoton, im deklarierten Bereich, und — der schärfste
//      Fall — bei gleicher Quell- und Zielhelligkeit nahezu die Identität.
//      Wenn nichts zu komprimieren ist, darf die Kurve nichts tun.
//   3. Kernel: Neutralgrau bleibt neutral (R=G=B rein ⇒ R=G=B raus). Ein
//      Matrixfehler im Farbraum verfärbt Grau sofort sichtbar.
//   4. Kernel: heller rein ⇒ heller raus, Ausgang im gültigen Bereich.

#import <Foundation/Foundation.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

extern void kk_hdr_tone(float src_max_nits, float dst_max_nits, float min_nits,
                        float *in_min, float *in_max, float *out_min, float *out_max,
                        float lut[256]);
extern const float KK_IPT_RGB2LMS_2020[9], KK_IPT_LMS2RGB_2020[9];
extern const float KK_IPT_LMS2IPT[9], KK_IPT_IPT2LMS[9];

static void mul33(const float a[9], const float b[9], double out[9]) {
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += (double)a[r*3+k] * b[k*3+c];
            out[r*3+c] = s;
        }
}

static int pruefe_matrizen(void) {
    struct { const float *m1, *m2; const char *name; } paare[] = {
        { KK_IPT_RGB2LMS_2020, KK_IPT_LMS2RGB_2020, "rgb2lms -> lms2rgb" },
        { KK_IPT_LMS2IPT,      KK_IPT_IPT2LMS,      "lms2ipt -> ipt2lms" },
    };
    int schlecht = 0;
    for (int p = 0; p < 2; p++) {
        double prod[9];
        mul33(paare[p].m2, paare[p].m1, prod);   // hin und zurück
        double maxabw = 0;
        for (int i = 0; i < 9; i++) {
            double soll = (i % 4 == 0) ? 1.0 : 0.0;   // Diagonale
            maxabw = fmax(maxabw, fabs(prod[i] - soll));
        }
        printf("  %-22s Identitaet? maxabw=%.2e", paare[p].name, maxabw);
        // Die Matrizen sind als float32 abgelegt und aufeinander invertiert —
        // ein paar 1e-6 sind Rundung, alles darüber wäre ein echter Fehler.
        int s = maxabw > 1e-4;
        printf("%s\n", s ? "   FEHLER" : "   ok");
        schlecht |= s;
    }
    return schlecht;
}

static int pruefe_tonekurve(void) {
    float in_min, in_max, out_min, out_max, lut[256];
    int schlecht = 0;

    // Fall A: echtes Tonemapping (1000-nit-Master auf 203-nit-Ziel = Normalfall).
    kk_hdr_tone(1000.0f, 203.0f, 0.005f, &in_min, &in_max, &out_min, &out_max, lut);
    int nicht_monoton = 0, ausserhalb = 0;
    for (int i = 0; i < 256; i++) {
        if (i && lut[i] + 1e-6f < lut[i-1]) nicht_monoton++;
        if (lut[i] < out_min - 1e-5f || lut[i] > out_max + 1e-5f) ausserhalb++;
    }
    printf("  Tonemap 1000->203 nit    monoton=%s  im Bereich=%s",
           nicht_monoton ? "NEIN" : "ja", ausserhalb ? "NEIN" : "ja");
    schlecht |= (nicht_monoton || ausserhalb);
    printf("%s\n", (nicht_monoton || ausserhalb) ? "   FEHLER" : "   ok");

    // Fall B — der schärfste: Quelle und Ziel gleich hell. Dann gibt es nichts zu
    // komprimieren, die Kurve MUSS praktisch die Identität sein. Ein Fehler in der
    // bt2390-Rechnung fällt hier auf, während er bei Fall A in der Kurve untergeht.
    kk_hdr_tone(203.0f, 203.0f, 0.005f, &in_min, &in_max, &out_min, &out_max, lut);
    double maxabw = 0;
    for (int i = 0; i < 256; i++) {
        double x = (double)i / 255.0;
        double soll = x * (in_max - in_min) + in_min;   // unveränderte Rampe im PQ-Raum
        maxabw = fmax(maxabw, fabs(lut[i] - soll));
    }
    printf("  Tonemap 203->203 nit     Identitaet? maxabw=%.2e", maxabw);
    int s = maxabw > 0.02;   // 2 % im PQ-Raum
    printf("%s\n", s ? "   FEHLER (Kurve arbeitet ohne Anlass)" : "   ok");
    return schlecht | s;
}

static int pruefe_kernel(kk_gpu *g) {
    const int W = 32, H = 8;
    // Linearlicht-Eingang: Graurampe (R=G=B). Was neutral reingeht, muss neutral
    // rauskommen — sonst stimmt eine der vier Matrizen nicht.
    unsigned char *in = malloc(4 * W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char v = (unsigned char)(8 + (240 * x) / (W - 1));
            in[4*(y*W+x)] = in[4*(y*W+x)+1] = in[4*(y*W+x)+2] = v;
            in[4*(y*W+x)+3] = 255;
        }
    kk_tex *src = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_SAMPLE, in);
    kk_tex *dst = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);

    // Produktions-Parameter wie in hybrid_render.c.
    kk_hdr_params hp;
    memcpy(hp.rgb2lms, KK_IPT_RGB2LMS_2020, 9 * sizeof(float));
    memcpy(hp.lms2rgb, KK_IPT_LMS2RGB_2020, 9 * sizeof(float));
    memcpy(hp.lms2ipt, KK_IPT_LMS2IPT, 9 * sizeof(float));
    memcpy(hp.ipt2lms, KK_IPT_IPT2LMS, 9 * sizeof(float));
    kk_hdr_tone(1000.0f, 203.0f, 0.005f,
                &hp.in_min, &hp.in_max, &hp.out_min, &hp.out_max, hp.tone_lut);

    kk_gpu_compute(g, CMHDR_MSL, "cmh", &(kk_compute_args){
        .out = dst, .in = { src }, .n_in = 1, .uniforms = &hp, .uniforms_size = sizeof hp });
    kk_gpu_finish(g);

    unsigned char *out = malloc(4 * W * H);
    kk_tex_download(g, dst, out);

    int maxfarbstich = 0, nicht_monoton = 0;
    for (int i = 0; i < W * H; i++) {
        int r = out[4*i], gg = out[4*i+1], b = out[4*i+2];
        int stich = abs(r - gg); stich = abs(r - b) > stich ? abs(r - b) : stich;
        maxfarbstich = stich > maxfarbstich ? stich : maxfarbstich;
    }
    for (int x = 0; x + 1 < W; x++)
        if ((int)out[4*x+4] + 1 < (int)out[4*x]) nicht_monoton++;

    printf("  CMHDR Graurampe          Farbstich=%d LSB  Monotonie-Brueche=%d",
           maxfarbstich, nicht_monoton);
    // 2020-Primärfarben sind nicht exakt neutral in dieser Kette; ein paar LSB
    // sind normal, ein deutlicher Stich wäre eine vertauschte Matrix.
    int schlecht = (maxfarbstich > 6) || (nicht_monoton > 0);
    printf("%s\n", schlecht ? "   FEHLER" : "   ok");

    free(in); free(out);
    kk_tex_destroy(g, &src); kk_tex_destroy(g, &dst);
    return schlecht;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    int fehler = pruefe_matrizen();
    fehler |= pruefe_tonekurve();

    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }
    fehler |= pruefe_kernel(g);
    kk_gpu_destroy(&g);

    printf(fehler ? "kk_hdr_test: FEHLGESCHLAGEN\n"
                  : "kk_hdr_test: IPT-Matrizen, bt2390-Kurve und CMHDR-Kernel  PASS\n");
    return fehler ? 1 : 0;
} }

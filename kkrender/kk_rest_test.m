// kk_rest_test.m — die verbliebenen Produktions-Kernel: LINSIG, DELINU, DEBAND.
//
// Damit ist jeder Kernel aus kk_gpu_render.c mindestens einmal abgedeckt.
//
// 1. SIGMOID-RUNDLAUF (LINSIG -> DELINU). Die beiden Kurven sind zueinander
//    invers — die Konstanten passen exakt aufeinander (1/0,82796854 = 1,20778…,
//    0,00757286/0,82796854 = 0,0091463…). Ein Tippfehler in einer der sechs
//    Konstanten bricht den Rundlauf sofort.
//    ⚠️ Der Rundlauf ist NICHT die Identität: LINSIG linearisiert nach BT.1886
//    (a·(c+b)^2,4), DELINU encodiert nach sRGB. Das sind verschiedene Kurven, der
//    Unterschied ist gewollt und systematisch. Geprüft wird deshalb, was ein
//    korrekter Rundlauf leisten MUSS: Monotonie, Extremwert-Treue und —
//    entscheidend — keine Plateaus. Ein Plateau hiesse, dass die Sigmoid-Kurve
//    sättigt und Tonwerte unwiederbringlich zusammenfallen.
//
// 2. DEBAND. Ein PRNG-gestützter Filter, dessen Ausgabe man nicht nachrechnen
//    kann. Prüfbar ist, was er NICHT tun darf: die Helligkeit verschieben,
//    echte Kanten verwischen, oder bei gleichem Bild und gleichem Index
//    unterschiedlich ausfallen.

#import <Foundation/Foundation.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

typedef struct { float a, b; float m[9]; } L_uniform;
typedef struct { float radius, threshold, grain; uint32_t iters, index; } DB_uniform;

static int sigmoid_rundlauf(kk_gpu *g) {
    const int W = 256, H = 4;
    unsigned char *in = malloc(4 * W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {          // volle Tonwert-Rampe 0..255
            in[4*(y*W+x)] = in[4*(y*W+x)+1] = in[4*(y*W+x)+2] = (unsigned char)x;
            in[4*(y*W+x)+3] = 255;
        }
    kk_tex *src = kk_tex_create(g, W, H, KK_FMT_RGBA8,
                                KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, in);
    // Zwischenstufe in RGBA16F: der Sigmoid-Raum braucht Präzision, in der
    // Produktion steht hier ebenfalls eine Float-Textur.
    kk_tex *sig = kk_tex_create(g, W, H, KK_FMT_RGBA16F, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    kk_tex *dst = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);

    L_uniform L = { 0.8704f, 0.0595f, { 1,0,0, 0,1,0, 0,0,1 } };   // Produktionswerte
    kk_gpu_compute(g, LINSIG_MSL, "linsig", &(kk_compute_args){
        .out = sig, .in = { src }, .n_in = 1, .uniforms = &L, .uniforms_size = sizeof L });
    kk_gpu_compute(g, DELINU_MSL, "delinu", &(kk_compute_args){
        .out = dst, .in = { sig }, .n_in = 1 });
    kk_gpu_finish(g);

    unsigned char *out = malloc(4 * W * H);
    kk_tex_download(g, dst, out);

    int brueche = 0, plateau = 0, lauf = 1;
    for (int x = 1; x < W; x++) {
        int v = out[4*x], vp = out[4*(x-1)];
        if (v + 1 < vp) brueche++;                  // muss monoton bleiben
        if (v == vp) { lauf++; if (lauf > plateau) plateau = lauf; } else lauf = 1;
    }
    int schwarz = out[0], weiss = out[4*(W-1)];
    printf("  Sigmoid-Rundlauf         Brueche=%d  laengstes Plateau=%d  0->%d  255->%d",
           brueche, plateau, schwarz, weiss);
    // Plateau >4 hiesse: mehr als vier benachbarte Eingangswerte fallen auf denselben
    // Ausgang — die Kurve sättigt. Extremwerte dürfen wandern (andere Gamma-Kurve),
    // aber Schwarz muss dunkel und Weiss muss hell bleiben.
    int schlecht = (brueche > 0) || (plateau > 4) || (schwarz > 40) || (weiss < 215);
    printf("%s\n", schlecht ? "   FEHLER" : "   ok");

    free(in); free(out);
    kk_tex_destroy(g, &src); kk_tex_destroy(g, &sig); kk_tex_destroy(g, &dst);
    return schlecht;
}

static void deband_lauf(kk_gpu *g, const unsigned char *in, int W, int H,
                        unsigned char *out, uint32_t index) {
    // ⚠️ KK_TEX_DOWNLOAD, obwohl nichts heruntergeladen wird: nur damit legt
    // kk_tex_create die Textur SHARED an. Ohne das Flag wird sie PRIVATE und das
    // `replaceRegion` mit den Anfangsdaten ist laut Metal ungültig — es geht
    // meistens gut und kracht bei bestimmten Grössen (hier 64×64, SIGSEGV).
    kk_tex *src = kk_tex_create(g, W, H, KK_FMT_RGBA8,
                                KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, (void *)in);
    kk_tex *dst = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    DB_uniform db = { .radius = 16.0f, .threshold = 0.004f, .grain = 0.0f,
                      .iters = 1, .index = index };   // grain=0: reiner Filter, kein Rauschen
    kk_gpu_compute(g, DEBAND_MSL, "deband", &(kk_compute_args){
        .out = dst, .in = { src }, .n_in = 1, .uniforms = &db, .uniforms_size = sizeof db });
    kk_gpu_finish(g);
    kk_tex_download(g, dst, out);
    kk_tex_destroy(g, &src); kk_tex_destroy(g, &dst);
}

static int pruefe_deband(kk_gpu *g) {
    const int W = 64, H = 64;
    unsigned char *flaeche = malloc(4 * W * H);
    for (int i = 0; i < W * H; i++) {
        flaeche[4*i] = flaeche[4*i+1] = flaeche[4*i+2] = 128; flaeche[4*i+3] = 255;
    }
    unsigned char *a = malloc(4 * W * H), *b = malloc(4 * W * H);

    // (a) Helligkeitstreue: auf einer Fläche gibt es keine Bänder — Deband darf
    //     dort nichts verändern. Verschöbe er den Pegel, wäre jedes Bild betroffen.
    deband_lauf(g, flaeche, W, H, a, 0);
    int maxabw = 0;
    for (int i = 0; i < W * H; i++) {
        int d = abs((int)a[4*i] - 128);
        maxabw = d > maxabw ? d : maxabw;
    }
    printf("  Deband Flaechentreue     maxabw=%d LSB", maxabw);
    int schlecht = maxabw > 1;
    printf("%s\n", schlecht ? "   FEHLER (verschiebt den Pegel)" : "   ok");

    // (b) Determinismus: gleiches Bild + gleicher Index ⇒ gleiches Ergebnis. Der
    //     PRNG wird aus (x, y, index) gespeist; hinge er an etwas anderem, flackerte
    //     das Bild bei Standbild.
    unsigned char *kante = malloc(4 * W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            unsigned char v = (x < W/2) ? 30 : 220;      // harte Kante
            kante[4*(y*W+x)] = kante[4*(y*W+x)+1] = kante[4*(y*W+x)+2] = v;
            kante[4*(y*W+x)+3] = 255;
        }
    deband_lauf(g, kante, W, H, a, 7);
    deband_lauf(g, kante, W, H, b, 7);
    int unterschiede = 0;
    for (int i = 0; i < 4 * W * H; i++) if (a[i] != b[i]) unterschiede++;
    printf("  Deband Determinismus     Abweichungen=%d", unterschiede);
    int s2 = unterschiede > 0;
    printf("%s\n", s2 ? "   FEHLER (PRNG haengt an etwas anderem)" : "   ok");

    // (c) Kantenerhaltung: die Schwelle schützt echte Kanten. Nach dem Filter muss
    //     der Sprung noch da sein — sonst wäre es ein Weichzeichner.
    int links = a[4*(H/2*W + W/4)], rechts = a[4*(H/2*W + 3*W/4)];
    printf("  Deband Kantenerhalt      links=%d rechts=%d (Soll 30/220)", links, rechts);
    int s3 = (abs(links - 30) > 8) || (abs(rechts - 220) > 8);
    printf("%s\n", s3 ? "   FEHLER (verwischt Kanten)" : "   ok");

    free(flaeche); free(kante); free(a); free(b);
    return schlecht | s2 | s3;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }

    int fehler = sigmoid_rundlauf(g);
    fehler |= pruefe_deband(g);

    kk_gpu_destroy(&g);
    printf(fehler ? "kk_rest_test: FEHLGESCHLAGEN\n"
                  : "kk_rest_test: LINSIG/DELINU-Rundlauf + DEBAND-Invarianten  PASS\n");
    return fehler ? 1 : 0;
} }

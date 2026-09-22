// kk_iq_probe.m — MISST Bildqualitäts-Eigenschaften der Produktions-Kernel.
//
// Wie kk_bench bewusst NICHT in der Standardliste von run-tests.sh: prüft nichts,
// schlägt nie fehl. Gezielt aufrufen:  ./run-tests.sh kk_iq_probe
//
// Drei Fragen (2026-09-22), alle gegen eine CPU-Referenz in double gerechnet:
//
//  1. SCHWARZWERT: wo landet Video-Schwarz (Y=16) nach DECLIN -> DELIN?
//     BT.1886 mit a/b aus Kontrast 1000:1 liefert 0,001 Linearlicht für Schwarz;
//     DELIN kodiert ohne Schwarzpunkt-Anpassung nach sRGB.
//  2. QUANTISIERUNG: wie gleichmäßig werden 219 TV-Stufen auf 256 sRGB-Codes
//     abgebildet, und wie weit weicht der Flächen-MITTELWERT je Stufe vom exakten
//     Wert ab — ohne und mit Dither (TPDF ±1 LSB vor der 8-Bit-Rundung)?
//  3. CHROMA-ORT: alle gemessenen Quellen (Tuner, Aufnahmen, ZDF) melden
//     chroma_location=left. DEC/DECLIN tasten die Chroma aber mittig ab.
//     Gemessen wird der Farbfehler an Farbkanten gegen ein 4:4:4-Original.
//
// Bindet kk_gpu_render.c direkt ein (wie kk_fusion_test), um an die static-Kernel
// zu kommen.
//
// ⚠️ Quelltexturen mit Startdaten brauchen KK_TEX_DOWNLOAD (= Shared): ohne das
// legt kk_tex_create sie PRIVATE an, und der Upload stürzt ab (s. Fork 1ecbd8d).

#import <Foundation/Foundation.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

// Produktions-Parameter wie in kk_fusion_test (BT.709 limited, BT.1886 1000:1).
static const float DM[12] = { 1.1643f, 0.0f, 1.7927f,
                              1.1643f, -0.2132f, -0.5329f,
                              1.1643f, 2.1124f, 0.0f,
                              -0.9729f, 0.3015f, -1.1334f };
static const float LA = 0.8704f, LB = 0.0595f;

typedef struct { float d[12]; float a, b; float m[9]; float o; float co[2]; } DL_uniform;
typedef struct { float m[12]; float co[2]; } D_uniform;

static double srgb_d(double c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055; }

/// Exakter Ausgabewert (0..255, ungerundet) für eine Y/Cb/Cr-Eingabe, Kanal k.
static double wahr(double Y, double Cb, double Cr, int k, int bpc) {
    double v[3] = { Y / 255.0, Cb / 255.0, Cr / 255.0 };
    double rgb = DM[3*k] * v[0] + DM[3*k+1] * v[1] + DM[3*k+2] * v[2] + DM[9+k];
    if (rgb < 0) rgb = 0;
    double lin = LA * pow(rgb + LB, 2.4);
    if (bpc) { double lb = LA * pow(LB, 2.4); lin = (lin - lb) / (1 - lb); }
    if (lin > 1) lin = 1;
    return 255.0 * srgb_d(lin);
}

// Seit renderpl.73 misst die Sonde die PRODUKTIONS-Kernel: DELIN_D_MSL (Dither),
// DEC_MSL mit Chroma-Ort-Uniform, kk_lin_params (Schwarzpunkt). „alt" = die Werte,
// mit denen die Kernel exakt den Stand vor .73 rechnen (o=0, co=0, ohne _D).

// ---------------------------------------------------------------- 1 + 2
static void probe_quant(kk_gpu *g) {
    const int STUFEN = 220, BW = 16, W = STUFEN * BW, H = 16;  // je TV-Stufe ein 16×16-Feld
    const int CW = W / 2, CH = H / 2;
    unsigned char *luma = malloc(W * H), *chroma = malloc(2 * CW * CH);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) luma[y*W+x] = (unsigned char)(16 + x / BW);
    memset(chroma, 128, 2 * CW * CH);
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, luma);
    kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, chroma);
    kk_tex *lin = kk_tex_create(g, W, H, KK_FMT_RGBA16F, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    kk_tex *linN = kk_tex_create(g, W, H, KK_FMT_RGBA16F, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    kk_tex *oAlt = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_tex *o0 = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_tex *o1 = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    DL_uniform DL; memcpy(DL.d, DM, sizeof DL.d); DL.a = LA; DL.b = LB; DL.o = 0; DL.co[0] = DL.co[1] = 0;
    float id3[9] = {1,0,0, 0,1,0, 0,0,1}; memcpy(DL.m, id3, sizeof DL.m);
    kk_gpu_compute(g, DECLIN_MSL, "declin", &(kk_compute_args){ .out = lin, .in = { tl, tc }, .n_in = 2,
        .linear = { false, true }, .uniforms = &DL, .uniforms_size = sizeof DL });
    kk_gpu_compute(g, DELIN_MSL, "delin", &(kk_compute_args){ .out = oAlt, .in = { lin }, .n_in = 1 });
    DL_uniform DN = DL; kk_lin_params(&DN.a, &DN.o);            // Produktion ab .73
    kk_gpu_compute(g, DECLIN_MSL, "declin", &(kk_compute_args){ .out = linN, .in = { tl, tc }, .n_in = 2,
        .linear = { false, true }, .uniforms = &DN, .uniforms_size = sizeof DN });
    kk_gpu_compute(g, DELIN_MSL, "delin", &(kk_compute_args){ .out = o0, .in = { linN }, .n_in = 1 });
    kk_gpu_compute(g, DELIN_D_MSL, "delin", &(kk_compute_args){ .out = o1, .in = { linN }, .n_in = 1 });
    kk_gpu_finish(g);
    unsigned char *a = malloc(4 * W * H), *b = malloc(4 * W * H), *alt = malloc(4 * W * H);
    kk_tex_download(g, o0, a); kk_tex_download(g, o1, b); kk_tex_download(g, oAlt, alt);

    // Kanal G (Grau: R=G=B bis auf Rundung).
    int code[220], codeAlt[220]; double mA[220], mB[220], w[220];
    for (int s = 0; s < STUFEN; s++) {
        double sa = 0, sb = 0;
        for (int y = 0; y < H; y++) for (int x = s*BW; x < (s+1)*BW; x++) { sa += a[4*(y*W+x)+1]; sb += b[4*(y*W+x)+1]; }
        mA[s] = sa / (BW*H); mB[s] = sb / (BW*H);
        code[s] = a[4*(8*W + s*BW + 8) + 1];
        codeAlt[s] = alt[4*(8*W + s*BW + 8) + 1];
        w[s] = wahr(16 + s, 128, 128, 1, 1);
    }
    printf("\n[1] Schwarzwert (Y -> sRGB-Code, 0 = echtes Schwarz)   alt | neu\n");
    const int zeig[] = { 0, 1, 2, 4, 8, 14, 24, 44, 84, 144, 219 };
    for (unsigned i = 0; i < sizeof zeig / sizeof *zeig; i++)
        printf("    Y=%3d  %3d | %3d\n", 16 + zeig[i], codeAlt[zeig[i]], code[zeig[i]]);
    printf("    Linearlicht bei Y=16: %.5f (= %.2f %% Weiss)\n", LA * pow(LB, 2.4), 100 * LA * pow(LB, 2.4));

    printf("\n[2] Stufen: 1 TV-Schritt -> wie viele sRGB-Codes?\n");
    int hist[8] = {0}, histD[8] = {0};
    for (int s = 1; s < STUFEN; s++) {
        int d = code[s] - code[s-1]; if (d < 0) d = 0; if (d > 7) d = 7;
        hist[d]++; if (s <= 30) histD[d]++;
    }
    printf("    gesamt:        0:%d 1:%d 2:%d 3:%d 4+:%d\n", hist[0], hist[1], hist[2], hist[3], hist[4]+hist[5]+hist[6]+hist[7]);
    printf("    Schatten Y16-46: 0:%d 1:%d 2:%d 3:%d 4+:%d\n", histD[0], histD[1], histD[2], histD[3], histD[4]+histD[5]+histD[6]+histD[7]);
    printf("    erste Codes: "); for (int s = 0; s < 16; s++) printf("%d ", code[s]); printf("...\n");
    int benutzt = 0, seen[256] = {0}; for (int s = 0; s < STUFEN; s++) if (!seen[code[s]]++) benutzt++;
    printf("    genutzte Ausgabe-Codes: %d von %d (Spanne %d..%d)\n", benutzt, code[219]-code[0]+1, code[0], code[219]);

    double eA = 0, eB = 0, qA = 0, qB = 0;
    for (int s = 0; s < STUFEN; s++) {
        double da = fabs(mA[s] - w[s]), db = fabs(mB[s] - w[s]);
        eA = fmax(eA, da); eB = fmax(eB, db); qA += da*da; qB += db*db;
    }
    printf("    Flächen-Mittel vs. exakt:  ohne Dither max %.3f / rms %.3f LSB   mit Dither max %.3f / rms %.3f LSB\n",
           eA, sqrt(qA/STUFEN), eB, sqrt(qB/STUFEN));
    double rausch = 0; int n = 0;
    for (int s = 0; s < STUFEN; s++) for (int y = 0; y < H; y++) for (int x = s*BW; x < (s+1)*BW; x++) {
        double d = b[4*(y*W+x)+1] - mB[s]; rausch += d*d; n++; }
    printf("    Preis des Dithers: Pixelrauschen %.2f LSB rms\n", sqrt(rausch / n));

    free(luma); free(chroma); free(a); free(b); free(alt);
    kk_tex_destroy(g,&tl); kk_tex_destroy(g,&tc); kk_tex_destroy(g,&lin); kk_tex_destroy(g,&linN);
    kk_tex_destroy(g,&oAlt); kk_tex_destroy(g,&o0); kk_tex_destroy(g,&o1);
}

// ---------------------------------------------------------------- 3
// Original 4:4:4: graue Luma, farbige Balken mit harten Kanten an geraden UND
// ungeraden Positionen. Encoder-Seite wie H.264/MPEG-2 left-sited: horizontal
// [1 2 1]/4 um die geraden Luma-Positionen, vertikal Mittel aus zwei Zeilen.
static void probe_chroma(kk_gpu *g) {
    const int W = 256, H = 8, CW = W / 2, CH = H / 2;
    double cb[256], cr[256];   // W
    for (int x = 0; x < W; x++) {
        int balken = ((x + (x / 37)) / 11) % 3;          // unregelmäßige Kantenlage
        cb[x] = balken == 0 ? 90 : balken == 1 ? 200 : 128;
        cr[x] = balken == 0 ? 220 : balken == 1 ? 60 : 128;
    }
    unsigned char *luma = malloc(W * H), *chroma = malloc(2 * CW * CH);
    memset(luma, 126, W * H);
    for (int j = 0; j < CH; j++) for (int i = 0; i < CW; i++) {
        int x = 2 * i, xm = x > 0 ? x - 1 : 0, xp = x + 1 < W ? x + 1 : W - 1;
        chroma[2*(j*CW+i)+0] = (unsigned char)lround((cb[xm] + 2*cb[x] + cb[xp]) / 4);
        chroma[2*(j*CW+i)+1] = (unsigned char)lround((cr[xm] + 2*cr[x] + cr[xp]) / 4);
    }
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, luma);
    kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, chroma);
    D_uniform U; memcpy(U.m, DM, sizeof U.m); U.co[1] = 0;
    // co in Chroma-Texeln: +0,25 = left-sited = 0,5 Luma-Pixel.
    const float versatz[3] = { 0.0f, 0.25f, -0.25f };
    const char *name[3] = { "mittig (alt)", "left (neu, +0,25 T)", "Gegenprobe (-0,25)" };
    printf("\n[3] Chroma-Ort: Farbfehler gegen das 4:4:4-Original (encodete RGB, LSB)\n");
    for (int v = 0; v < 3; v++) {
        U.co[0] = versatz[v];
        kk_tex *o = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
        kk_gpu_compute(g, DEC_MSL, "dec", &(kk_compute_args){ .out = o, .in = { tl, tc }, .n_in = 2,
            .linear = { false, true }, .uniforms = &U, .uniforms_size = sizeof U });
        kk_gpu_finish(g);
        unsigned char *px = malloc(4 * W * H); kk_tex_download(g, o, px);
        double q = 0, mx = 0, qk = 0; int n = 0, nk = 0;
        for (int x = 2; x < W - 2; x++) {
            int kante = (cb[x-1] != cb[x]) || (cb[x] != cb[x+1]);
            for (int k = 0; k < 3; k += 2) {           // R und B tragen die Chroma
                double vv[3] = { 126 / 255.0, cb[x] / 255.0, cr[x] / 255.0 };
                double t = 255.0 * (DM[3*k]*vv[0] + DM[3*k+1]*vv[1] + DM[3*k+2]*vv[2] + DM[9+k]);
                t = t < 0 ? 0 : t > 255 ? 255 : t;
                double d = px[4*(4*W + x) + k] - t;
                q += d*d; n++; mx = fmax(mx, fabs(d));
                if (kante) { qk += d*d; nk++; }
            }
        }
        printf("    %-20s rms %.2f   an Kanten rms %.2f   max %.0f\n", name[v], sqrt(q/n), sqrt(qk/nk), mx);
        free(px); kk_tex_destroy(g, &o);
    }
    free(luma); free(chroma); kk_tex_destroy(g,&tl); kk_tex_destroy(g,&tc);
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }
    probe_quant(g);
    probe_chroma(g);
    kk_gpu_destroy(&g);
    printf("\nkk_iq_probe: gemessen (prüft nichts)\n");
    return 0;
} }

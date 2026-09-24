// kk_iq_test.m — prüft die drei Bildqualitäts-Korrekturen aus renderpl.73.
//
// Gemessen wurden sie mit kk_iq_probe (misst, prüft nichts). Hier stehen die
// Invarianten, die danach halten MÜSSEN — sonst geht eine davon still verloren:
//
//  1. SCHWARZPUNKT: mit kk_lin_params landet Y=16 bei Code 0 und Y=235 bei 255.
//     Mit KUCKUCK_BLACKPOINT=0 kommt exakt der alte Wert (Code 3) zurück.
//  2. DITHER: DELIN_D verschiebt den Pegel einer Fläche nicht (Mittel ±0,15 LSB),
//     weicht je Pixel höchstens 1 Code ab und rauscht tatsächlich (sonst wäre der
//     Dither still abgeschaltet). DELIN ohne _D bleibt rauschfrei.
//  3. CHROMA-ORT: bei left-sited kodierter Quelle ist der Farbfehler mit dem
//     Versatz aus kk_chroma_offset (+0,25 Texel) KLEINER als ohne; das Vorzeichen
//     umzudrehen macht es schlechter. Ein Pixelbuffer ohne Attachment ergibt left.
//  4. CHROMA-LANCZOS3: CHH + DEC_L3 hat an denselben Farbkanten einen kleineren
//     Fehler als bilinear (beide left-sited) und lässt eine flache Farbfläche
//     unverändert (die Gewichte summieren sich zu 1, kein Pegelversatz). Default nur
//     ausserhalb HD-Light, KUCKUCK_CHROMA_UP=bilinear|lanczos erzwingt.

#import <Foundation/Foundation.h>
#include "kk_gpu.h"
#include "kk_gpu_render.c"

typedef struct { float d[12]; float a, b; float m[9]; float o; float co[2]; } DL_uniform;
typedef struct { float m[12]; float co[2]; } D_uniform;

static const float DM[12] = { 1.1643f, 0.0f, 1.7927f, 1.1643f, -0.2132f, -0.5329f,
                              1.1643f, 2.1124f, 0.0f, -0.9729f, 0.3015f, -1.1334f };

/// Grauwert-Code (Kanal G) für eine flache Luma-Fläche nach DECLIN -> DELIN[_D].
static void grau(kk_gpu *g, unsigned char Y, float a, float o, const char *delin,
                 unsigned char *px, int W, int H) {
    unsigned char *l = malloc(W * H), *c = malloc(2 * (W/2) * (H/2));
    memset(l, Y, W * H); memset(c, 128, 2 * (W/2) * (H/2));
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, l);
    kk_tex *tc = kk_tex_create(g, W/2, H/2, KK_FMT_RG8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, c);
    kk_tex *lin = kk_tex_create(g, W, H, KK_FMT_RGBA16F, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    kk_tex *out = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    DL_uniform DL; memcpy(DL.d, DM, sizeof DL.d); DL.a = a; DL.b = 0.0595f; DL.o = o;
    DL.co[0] = DL.co[1] = 0; float id3[9] = {1,0,0, 0,1,0, 0,0,1}; memcpy(DL.m, id3, sizeof DL.m);
    kk_gpu_compute(g, DECLIN_MSL, "declin", &(kk_compute_args){ .out = lin, .in = { tl, tc }, .n_in = 2,
        .linear = { false, true }, .uniforms = &DL, .uniforms_size = sizeof DL });
    kk_gpu_compute(g, delin, "delin", &(kk_compute_args){ .out = out, .in = { lin }, .n_in = 1 });
    kk_gpu_finish(g);
    kk_tex_download(g, out, px);
    free(l); free(c);
    kk_tex_destroy(g,&tl); kk_tex_destroy(g,&tc); kk_tex_destroy(g,&lin); kk_tex_destroy(g,&out);
}

static int pruefe_schwarzpunkt(kk_gpu *g) {
    const int W = 16, H = 16; unsigned char px[4 * 16 * 16];
    float a, o;
    unsetenv("KUCKUCK_BLACKPOINT"); kk_lin_params(&a, &o);
    grau(g, 16, a, o, DELIN_MSL, px, W, H);  int schwarz = px[1];
    grau(g, 235, a, o, DELIN_MSL, px, W, H); int weiss = px[1];
    setenv("KUCKUCK_BLACKPOINT", "0", 1); kk_lin_params(&a, &o); unsetenv("KUCKUCK_BLACKPOINT");
    grau(g, 16, a, o, DELIN_MSL, px, W, H);  int altSchwarz = px[1];
    int schlecht = schwarz != 0 || weiss != 255 || altSchwarz != 3;
    printf("  Schwarzpunkt: Y=16 -> %d, Y=235 -> %d, Schalter aus -> %d (alt 3)%s\n",
           schwarz, weiss, altSchwarz, schlecht ? "   FEHLER" : "   ok");
    return schlecht;
}

static int pruefe_dither(kk_gpu *g) {
    const int W = 64, H = 64; unsigned char *p0 = malloc(4*W*H), *p1 = malloc(4*W*H);
    float a, o; kk_lin_params(&a, &o);
    int fehler = 0;
    for (int Y = 30; Y <= 200; Y += 17) {
        grau(g, (unsigned char)Y, a, o, DELIN_MSL, p0, W, H);
        grau(g, (unsigned char)Y, a, o, DELIN_D_MSL, p1, W, H);
        double m1 = 0; int maxd = 0, verschieden0 = 0, verschieden1 = 0;
        for (int i = 0; i < W*H; i++) {
            m1 += p1[4*i+1];
            int d = abs(p1[4*i+1] - p0[4*i+1]); if (d > maxd) maxd = d;
            verschieden0 += p0[4*i+1] != p0[1]; verschieden1 += p1[4*i+1] != p1[1];
        }
        m1 /= W*H;
        // Ohne Dither ist jeder Pixel GERUNDET (bis 0,5 LSB daneben). Mit Dither muss
        // das Flächenmittel den exakten Wert treffen — Vergleich gegen die CPU-Wahrheit.
        double v[3] = { Y/255.0, 128/255.0, 128/255.0 };   // Textur hält 128, nicht 0,5
        double rgb = DM[3]*v[0] + DM[4]*v[1] + DM[5]*v[2] + DM[10]; if (rgb < 0) rgb = 0;
        double lin = a * pow(rgb + 0.0595, 2.4) - o; if (lin > 1) lin = 1;
        double wahr = 255.0 * (lin <= 0.0031308 ? 12.92*lin : 1.055*pow(lin, 1/2.4) - 0.055);
        if (fabs(m1 - wahr) > 0.15 || maxd > 1 || verschieden1 == 0 || verschieden0 != 0) {
            printf("  Dither Y=%d: Mittel %.3f (exakt %.3f), max %d, rauscht %d/%d, ohne %d   FEHLER\n",
                   Y, m1, wahr, maxd, verschieden1, W*H, verschieden0);
            fehler = 1;
        }
    }
    if (!fehler) printf("  Dither: Pegel treu (±0,15 LSB), je Pixel ≤1 Code, ohne _D rauschfrei   ok\n");
    free(p0); free(p1);
    return fehler;
}

static int pruefe_chroma(kk_gpu *g) {
    const int W = 256, H = 8, CW = W/2, CH = H/2;
    double cb[256], cr[256];
    for (int x = 0; x < W; x++) {
        int b = ((x + (x / 37)) / 11) % 3;
        cb[x] = b == 0 ? 90 : b == 1 ? 200 : 128;  cr[x] = b == 0 ? 220 : b == 1 ? 60 : 128;
    }
    unsigned char *l = malloc(W*H), *c = malloc(2*CW*CH); memset(l, 126, W*H);
    for (int j = 0; j < CH; j++) for (int i = 0; i < CW; i++) {   // left-sited [1 2 1]/4
        int x = 2*i, xm = x > 0 ? x-1 : 0, xp = x+1 < W ? x+1 : W-1;
        c[2*(j*CW+i)+0] = (unsigned char)lround((cb[xm] + 2*cb[x] + cb[xp]) / 4);
        c[2*(j*CW+i)+1] = (unsigned char)lround((cr[xm] + 2*cr[x] + cr[xp]) / 4);
    }
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, l);
    kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, c);

    // Pixelbuffer OHNE Chroma-Attachment -> kk_chroma_offset muss left liefern.
    CVPixelBufferRef pb = NULL;
    CVPixelBufferCreate(NULL, 16, 16, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, NULL, &pb);
    float co[2] = { -9, -9 }; kk_chroma_offset(pb, co);
    float coC[2]; CVBufferSetAttachment(pb, kCVImageBufferChromaLocationTopFieldKey,
                                        kCVImageBufferChromaLocation_Center, kCVAttachmentMode_ShouldPropagate);
    kk_chroma_offset(pb, coC);
    CVPixelBufferRelease(pb);

    double rms[3]; const float vs[3] = { 0.0f, co[0], -co[0] };
    for (int v = 0; v < 3; v++) {
        D_uniform U; memcpy(U.m, DM, sizeof U.m); U.co[0] = vs[v]; U.co[1] = 0;
        kk_tex *o = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
        kk_gpu_compute(g, DEC_MSL, "dec", &(kk_compute_args){ .out = o, .in = { tl, tc }, .n_in = 2,
            .linear = { false, true }, .uniforms = &U, .uniforms_size = sizeof U });
        kk_gpu_finish(g);
        unsigned char *px = malloc(4*W*H); kk_tex_download(g, o, px);
        double q = 0; int n = 0;
        for (int x = 2; x < W-2; x++) for (int k = 0; k < 3; k += 2) {
            double vv[3] = { 126/255.0, cb[x]/255.0, cr[x]/255.0 };
            double t = 255.0 * (DM[3*k]*vv[0] + DM[3*k+1]*vv[1] + DM[3*k+2]*vv[2] + DM[9+k]);
            t = t < 0 ? 0 : t > 255 ? 255 : t;
            double d = px[4*(4*W+x)+k] - t; q += d*d; n++;
        }
        rms[v] = sqrt(q/n); free(px); kk_tex_destroy(g, &o);
    }
    int schlecht = co[0] != 0.25f || co[1] != 0.0f || coC[0] != 0.0f
                || !(rms[1] < rms[0] * 0.9) || !(rms[2] > rms[0]);
    printf("  Chroma-Ort: ohne Attachment co=%.2f (left), Center -> %.2f; Fehler mittig %.1f, left %.1f, verkehrt %.1f%s\n",
           co[0], coC[0], rms[0], rms[1], rms[2], schlecht ? "   FEHLER" : "   ok");
    free(l); free(c); kk_tex_destroy(g,&tl); kk_tex_destroy(g,&tc);
    return schlecht;
}

/// Farbfehler (rms, R+B) des Balkenbilds aus pruefe_chroma: bilinear oder Lanczos3.
static double balken_fehler(kk_gpu *g, int lanczos, double *flach_max) {
    const int W = 256, H = 8, CW = W/2, CH = H/2;
    double cb[256], cr[256];
    for (int x = 0; x < W; x++) {
        int b = ((x + (x / 37)) / 11) % 3;
        cb[x] = b == 0 ? 90 : b == 1 ? 200 : 128;  cr[x] = b == 0 ? 220 : b == 1 ? 60 : 128;
    }
    unsigned char *l = malloc(W*H), *c = malloc(2*CW*CH), *f = malloc(2*CW*CH); memset(l, 126, W*H);
    for (int j = 0; j < CH; j++) for (int i = 0; i < CW; i++) {
        int x = 2*i, xm = x > 0 ? x-1 : 0, xp = x+1 < W ? x+1 : W-1;
        c[2*(j*CW+i)+0] = (unsigned char)lround((cb[xm] + 2*cb[x] + cb[xp]) / 4);
        c[2*(j*CW+i)+1] = (unsigned char)lround((cr[xm] + 2*cr[x] + cr[xp]) / 4);
        f[2*(j*CW+i)+0] = 90; f[2*(j*CW+i)+1] = 220;              // flache Farbfläche
    }
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, l);
    double ergebnis[2] = { 0, 0 };
    for (int flach = 0; flach < 2; flach++) {
        kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, flach ? f : c);
        D_uniform U; memcpy(U.m, DM, sizeof U.m); U.co[0] = 0.25f; U.co[1] = 0;
        kk_tex *chh = lanczos ? kk_chroma_h(g, tc, &c_chh, W, U.co[0]) : NULL;
        kk_tex *o = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
        kk_gpu_compute(g, chh ? DEC_L3_MSL : DEC_MSL, "dec", &(kk_compute_args){ .out = o,
            .in = { tl, chh ? chh : tc }, .n_in = 2, .linear = { false, true }, .uniforms = &U, .uniforms_size = sizeof U });
        kk_gpu_finish(g);
        unsigned char *px = malloc(4*W*H); kk_tex_download(g, o, px);
        double q = 0, mx = 0; int n = 0;
        for (int x = 2; x < W-2; x++) for (int k = 0; k < 3; k += 2) {
            double vv[3] = { 126/255.0, (flach ? 90 : cb[x])/255.0, (flach ? 220 : cr[x])/255.0 };
            double t = 255.0 * (DM[3*k]*vv[0] + DM[3*k+1]*vv[1] + DM[3*k+2]*vv[2] + DM[9+k]);
            t = t < 0 ? 0 : t > 255 ? 255 : t;
            double d = px[4*(4*W+x)+k] - t; q += d*d; n++; mx = fmax(mx, fabs(d));
        }
        ergebnis[flach] = flach ? mx : sqrt(q/n);
        free(px); kk_tex_destroy(g, &o); kk_tex_destroy(g, &tc);
    }
    *flach_max = ergebnis[1];
    free(l); free(c); free(f); kk_tex_destroy(g, &tl);
    return ergebnis[0];
}

/// Exakte CPU-Referenz für CHH+DEC_L3 auf einem 2D-Muster (ändert sich in BEIDE
/// Richtungen, sonst fiele eine falsche vertikale Phase nicht auf). Toleranz 1 LSB.
static double l3_cpu(double x) {
    x = fabs(x); if (x < 1e-9) return 1.0; if (x >= 3.0) return 0.0;
    return 3.0 * sin(M_PI * x) * sin(M_PI * x / 3.0) / (M_PI * M_PI * x * x);
}
static double l3_1d(const double *src, int n, int stride, double s) {
    int b = (int)floor(s); double acc = 0, ws = 0;
    for (int t = -2; t <= 3; t++) { int i = b + t; i = i < 0 ? 0 : i >= n ? n-1 : i;
        double w = l3_cpu(s - (b + t)); acc += w * src[i * stride]; ws += w; }
    return acc / ws;
}
static int l3_gegen_cpu(kk_gpu *g, double *maxd) {
    const int W = 96, H = 64, CW = W/2, CH = H/2;
    unsigned char *l = malloc(W*H), *c = malloc(2*CW*CH);
    for (int i = 0; i < W*H; i++) l[i] = (unsigned char)(60 + (i * 37) % 120);
    double cu[48*32], cv[48*32];   // = CW*CH (feste Größe: keine VLA-Warnung)
    for (int j = 0; j < CH; j++) for (int i = 0; i < CW; i++) {
        cu[j*CW+i] = ((i/5 + j/3) % 2) ? 200 : 70;                  // Kanten in x UND y
        cv[j*CW+i] = 128 + 90 * sin(i * 0.9) * cos(j * 0.7);
        c[2*(j*CW+i)] = (unsigned char)cu[j*CW+i]; c[2*(j*CW+i)+1] = (unsigned char)lround(cv[j*CW+i]);
        cv[j*CW+i] = c[2*(j*CW+i)+1];
    }
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, l);
    kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, c);
    D_uniform U; memcpy(U.m, DM, sizeof U.m); U.co[0] = 0.25f; U.co[1] = 0.25f;   // top-left: beide Achsen
    kk_tex *chh = kk_chroma_h(g, tc, &c_chh, W, U.co[0]);
    kk_tex *o = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    kk_gpu_compute(g, DEC_L3_MSL, "dec", &(kk_compute_args){ .out = o, .in = { tl, chh }, .n_in = 2,
        .linear = { false, true }, .uniforms = &U, .uniforms_size = sizeof U });
    kk_gpu_finish(g);
    unsigned char *px = malloc(4*W*H); kk_tex_download(g, o, px);
    double hu[96*32], hv[96*32], md = 0;   // = W*CH
    for (int j = 0; j < CH; j++) for (int x = 0; x < W; x++) {
        double sx = (x + 0.5) * CW / (double)W - 0.5 + 0.25;
        hu[j*W+x] = l3_1d(cu + j*CW, CW, 1, sx); hv[j*W+x] = l3_1d(cv + j*CW, CW, 1, sx);
    }
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        double sy = (y + 0.5) * CH / (double)H - 0.5 + 0.25;
        double u = l3_1d(hu + x, CH, W, sy), v = l3_1d(hv + x, CH, W, sy);
        double vv[3] = { l[y*W+x]/255.0, u/255.0, v/255.0 };
        for (int k = 0; k < 3; k++) {
            double t = 255.0 * (DM[3*k]*vv[0] + DM[3*k+1]*vv[1] + DM[3*k+2]*vv[2] + DM[9+k]);
            t = t < 0 ? 0 : t > 255 ? 255 : t;
            md = fmax(md, fabs(px[4*(y*W+x)+k] - t));
        }
    }
    *maxd = md;
    free(l); free(c); free(px); kk_tex_destroy(g,&tl); kk_tex_destroy(g,&tc); kk_tex_destroy(g,&o);
    return md > 1.0;
}

static int pruefe_chroma_l3(kk_gpu *g) {
    double fB, fL, eB = balken_fehler(g, 0, &fB), eL = balken_fehler(g, 1, &fL);
    // Schaltlogik: Default Lanczos3 nur ausserhalb HD-Light, Env erzwingt beides.
    unsetenv("KUCKUCK_CHROMA_UP");
    int gate = kk_chroma_l3_an(false) && !kk_chroma_l3_an(true);
    setenv("KUCKUCK_CHROMA_UP", "bilinear", 1); gate &= !kk_chroma_l3_an(false);
    setenv("KUCKUCK_CHROMA_UP", "lanczos", 1);  gate &= kk_chroma_l3_an(true);
    unsetenv("KUCKUCK_CHROMA_UP");
    if (!gate) printf("  Chroma-Lanczos3: Schaltlogik (HD-Light/Env) falsch   FEHLER\n");
    double md; int cpuFehler = l3_gegen_cpu(g, &md);
    int schlecht = !gate || cpuFehler || !(eL < eB * 0.95) || fL > 1.0;
    printf("  Chroma-Lanczos3: gegen CPU-Referenz (2D, top-left) max %.2f LSB; Kanten bilinear %.1f -> %.1f; flach max %.1f%s\n",
           md, eB, eL, fL, schlecht ? "   FEHLER" : "   ok");
    return schlecht;
}

/// HDR: MKPQ_L3 (CHH auf P010-Chroma) gegen MKPQ bilinear. Flache Fläche muss
/// gleich bleiben (fängt Fehler in der 10-bit-Skalierung ×65535/64 durch CHH), an
/// einer Farbkante in y muss L3 die exakte CPU-Referenz treffen (≤1 LSB).
static int pruefe_hdr_l3(kk_gpu *g) {
    const int W = 64, H = 16, CW = W/2, CH = H/2;
    uint16_t *l = malloc(2*W*H), *cf = malloc(4*CW*CH), *ck = malloc(4*CW*CH);
    for (int i = 0; i < W*H; i++) l[i] = (uint16_t)(500 << 6);              // 10-bit Code 500
    for (int j = 0; j < CH; j++) for (int i = 0; i < CW; i++) {
        cf[2*(j*CW+i)] = 300 << 6; cf[2*(j*CW+i)+1] = 700 << 6;              // flach
        ck[2*(j*CW+i)] = (j < CH/2 ? 200 : 800) << 6; ck[2*(j*CW+i)+1] = 512 << 6;  // Kante in y
    }
    kk_tex *tl = kk_tex_create(g, W, H, KK_FMT_R16, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, l);
    double diff[2];
    for (int kante = 0; kante < 2; kante++) {
        kk_tex *tc = kk_tex_create(g, CW, CH, KK_FMT_RG16, KK_TEX_SAMPLE | KK_TEX_DOWNLOAD, kante ? ck : cf);
        struct { float co[2]; } K = { { 0.25f, 0.0f } };
        unsigned char *px[2];
        for (int v = 0; v < 2; v++) {
            kk_tex *in = v ? kk_chroma_h(g, tc, &h_chh, W, K.co[0]) : tc;
            // PQ-Linearlicht ist 10000-normiert und klein -> über RGBA8-Download
            // nicht auflösbar. Darum ×40 skaliert in ein RGBA8 schreiben lassen
            // geht nicht ohne Kernel-Änderung; stattdessen Float-Textur + eigener Readback.
            kk_tex *o = kk_tex_create(g, W, H, KK_FMT_RGBA16F, KK_TEX_STORAGE | KK_TEX_SAMPLE, NULL);
            kk_gpu_compute(g, v ? MKPQ_L3_MSL : MKPQ_MSL, "mk", &(kk_compute_args){ .out = o, .in = { tl, in },
                .n_in = 2, .linear = { false, true }, .uniforms = &K, .uniforms_size = sizeof K });
            // PQ-codieren zurück in RGBA8 (0..1 = voller PQ-Bereich), dann vergleichen.
            kk_tex *o8 = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
            kk_gpu_compute(g,
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "kernel void enc(texture2d<float> s [[texture(0)]], texture2d<float,access::write> d [[texture(1)]], uint2 id [[thread_position_in_grid]]){\n"
                "  if(id.x>=d.get_width()||id.y>=d.get_height())return; float3 L=max(s.read(id).rgb,0.0);\n"
                "  const float m1=0.1593017578125,m2=78.84375,c1=0.8359375,c2=18.8515625,c3=18.6875;\n"
                "  float3 p=pow(L,float3(m1)); d.write(float4(pow((c1+c2*p)/(1.0+c3*p),float3(m2)),1.0),id);}\n",
                "enc", &(kk_compute_args){ .out = o8, .in = { o }, .n_in = 1 });
            kk_gpu_finish(g);
            px[v] = malloc(4*W*H); kk_tex_download(g, o8, px[v]);
            kk_tex_destroy(g, &o); kk_tex_destroy(g, &o8);
        }
        double md = 0;
        if (!kante) {   // flach: L3 == bilinear
            for (int y = 2; y < H-2; y++) for (int x = 4; x < W-4; x++) for (int k = 0; k < 3; k++)
                md = fmax(md, fabs((double)px[0][4*(y*W+x)+k] - px[1][4*(y*W+x)+k]));
        } else {        // Kante in y: L3 gegen exakte CPU-Referenz (enc(pqe(rgb)) = rgb)
            double col[8]; for (int j = 0; j < CH; j++) col[j] = j < CH/2 ? 200 : 800;
            for (int y = 0; y < H; y++) {
                double sy = (y + 0.5) * CH / (double)H - 0.5;
                double Cb = (l3_1d(col, CH, 1, sy) - 512.0) / 896.0, Cr = 0.0, Y = (500 - 64) / 876.0;
                double rgb[3] = { Y + 1.4746*Cr, Y - 0.16455*Cb - 0.57135*Cr, Y + 1.8814*Cb };
                for (int x = 4; x < W-4; x++) for (int k = 0; k < 3; k++) {
                    double t = 255.0 * fmin(fmax(rgb[k], 0.0), 1.0);
                    md = fmax(md, fabs(px[1][4*(y*W+x)+k] - t));
                }
            }
        }
        diff[kante] = md; free(px[0]); free(px[1]); kk_tex_destroy(g, &tc);
    }
    free(l); free(cf); free(ck); kk_tex_destroy(g, &tl);
    int schlecht = diff[0] > 1.0 || diff[1] > 1.0;
    printf("  HDR-Chroma-Lanczos3: flach L3==bilinear max %.0f LSB, Kante gegen CPU-Referenz max %.2f LSB%s\n",
           diff[0], diff[1], schlecht ? "   FEHLER" : "   ok");
    return schlecht;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    unsetenv("KUCKUCK_CHROMA_LOC"); unsetenv("KUCKUCK_DITHER");
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { printf("kk_gpu_create fehlgeschlagen\n"); return 1; }
    int f = pruefe_schwarzpunkt(g);
    f |= pruefe_dither(g);
    f |= pruefe_chroma(g);
    f |= pruefe_chroma_l3(g);
    f |= pruefe_hdr_l3(g);
    kk_gpu_destroy(&g);
    printf(f ? "kk_iq_test: FEHLGESCHLAGEN\n"
             : "kk_iq_test: Schwarzpunkt, Dither, Chroma-Ort und Chroma-Lanczos3 (SDR+HDR) halten  PASS\n");
    return f ? 1 : 0;
} }

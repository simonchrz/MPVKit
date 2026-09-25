// kk_matrix_test.m — prüft, welche Farbmatrix/Primaries hybrid_render aus den
// CoreVideo-Etiketten wählt. Anlass (2026-09-25): SD-Aufnahmen tragen den rohen
// H.273-Codepunkt „YCbCrMatrix#5" (BT.470BG), für den CoreVideo keine benannte
// Konstante hat — der alte Konstanten-Vergleich fiel auf 709 durch, BT.601-Material
// wurde als 709 gerechnet (Grün/Rot verschoben). Unsichtbar ohne direkten Vergleich.

#import <Foundation/Foundation.h>
#include <CoreVideo/CoreVideo.h>
int hybrid_sysidx_fuer(CFTypeRef m);
int hybrid_primidx_fuer(CFTypeRef pr);
float hybrid_frame_peak_nits(CVPixelBufferRef pb);

/// P010-Testbild (limited range, 10 Bit in den oberen Bits) mit überall gleichem Y/Cb/Cr.
static CVPixelBufferRef p010_flaeche(int y10, int cb10, int cr10) {
    CVPixelBufferRef pb = NULL;
    if (CVPixelBufferCreate(NULL, 64, 32, kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, NULL, &pb) != kCVReturnSuccess) return NULL;
    CVPixelBufferLockBaseAddress(pb, 0);
    for (int y = 0; y < 32; y++) {
        uint16_t *r = (uint16_t *)((uint8_t *) CVPixelBufferGetBaseAddressOfPlane(pb, 0) + y * CVPixelBufferGetBytesPerRowOfPlane(pb, 0));
        for (int x = 0; x < 64; x++) r[x] = (uint16_t)(y10 << 6);
    }
    for (int y = 0; y < 16; y++) {
        uint16_t *r = (uint16_t *)((uint8_t *) CVPixelBufferGetBaseAddressOfPlane(pb, 1) + y * CVPixelBufferGetBytesPerRowOfPlane(pb, 1));
        for (int x = 0; x < 32; x++) { r[2*x] = (uint16_t)(cb10 << 6); r[2*x+1] = (uint16_t)(cr10 << 6); }
    }
    CVPixelBufferUnlockBaseAddress(pb, 0);
    return pb;
}

/// HDR-Spitzenwert (2026-09-25): die Schätzung muss gesättigte Farben erfassen. Reines
/// Rot mit 1000 nit (BT.2020/PQ: R'=0,7518) hat Y'≈0,1975 — die alte Luma-Schätzung
/// meldete dafür ~6 nit, das Tonemapping schnitt die Lichter dort ab.
static int pruefe_peak(const char *name, int y10, int cb10, int cr10, float lo, float hi) {
    CVPixelBufferRef pb = p010_flaeche(y10, cb10, cr10);
    float n = pb ? hybrid_frame_peak_nits(pb) : -1;
    if (pb) CVPixelBufferRelease(pb);
    int schlecht = !(n >= lo && n <= hi);
    printf("  %-34s %.0f nit (erwartet %.0f..%.0f)%s\n", name, n, lo, hi, schlecht ? "   FEHLER" : "");
    return schlecht;
}

static int pruefe(const char *name, int ist, int soll) {
    printf("  %-34s ist=%2d soll=%2d%s\n", name, ist, soll, ist == soll ? "" : "   FEHLER");
    return ist != soll;
}

int main(void) { @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    int f = 0;
    // Matrix: 0=601, 1=709, 2=240M, 3=2020
    f |= pruefe("Matrix fehlt → 709", hybrid_sysidx_fuer(NULL), 1);
    f |= pruefe("Matrix Konstante 601", hybrid_sysidx_fuer(kCVImageBufferYCbCrMatrix_ITU_R_601_4), 0);
    f |= pruefe("Matrix Konstante 709", hybrid_sysidx_fuer(kCVImageBufferYCbCrMatrix_ITU_R_709_2), 1);
    f |= pruefe("Matrix Konstante 2020", hybrid_sysidx_fuer(kCVImageBufferYCbCrMatrix_ITU_R_2020), 3);
    f |= pruefe("Matrix Codepunkt 5 (BT.470BG) → 601",
                hybrid_sysidx_fuer(CVYCbCrMatrixGetStringForIntegerCodePoint(5)), 0);
    f |= pruefe("Matrix Codepunkt 6 (170M) → 601",
                hybrid_sysidx_fuer(CVYCbCrMatrixGetStringForIntegerCodePoint(6)), 0);
    f |= pruefe("Matrix Codepunkt 1 → 709",
                hybrid_sysidx_fuer(CVYCbCrMatrixGetStringForIntegerCodePoint(1)), 1);
    // Primaries: 0=601-525, 1=601-625 (EBU), 2=2020, -1=709
    f |= pruefe("Primaries fehlt → 709", hybrid_primidx_fuer(NULL), -1);
    f |= pruefe("Primaries Konstante EBU", hybrid_primidx_fuer(kCVImageBufferColorPrimaries_EBU_3213), 1);
    f |= pruefe("Primaries Codepunkt 5 → EBU",
                hybrid_primidx_fuer(CVColorPrimariesGetStringForIntegerCodePoint(5)), 1);
    f |= pruefe("Primaries Codepunkt 6 → SMPTE-C",
                hybrid_primidx_fuer(CVColorPrimariesGetStringForIntegerCodePoint(6)), 0);
    f |= pruefe("Primaries Konstante 709", hybrid_primidx_fuer(kCVImageBufferColorPrimaries_ITU_R_709_2), -1);
    // Rot 1000 nit: Y'=64+876*0,1975=237, Cb=512+896*(-0,1050)=418, Cr=512+896*0,3759=849.
    f |= pruefe_peak("Peak: reines Rot 1000 nit", 237, 418, 849, 800, 1250);
    // Grau 100 nit: Y'=PQ(100)=0,5081 → 64+876*0,5081=509, Cb=Cr=512.
    f |= pruefe_peak("Peak: Grau 100 nit", 509, 512, 512, 85, 120);
    printf(f ? "kk_matrix_test: FEHLGESCHLAGEN\n"
             : "kk_matrix_test: Farbmatrix/Primaries (Codepunkt) + HDR-Peak (MaxRGB)  PASS\n");
    return f;
} }

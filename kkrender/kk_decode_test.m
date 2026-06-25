// kk_decode_test.m — Etappe-2-Verifikation: YUV-Decode+Chroma-Upscale-MSL-Kernel
// auf kk_gpu. Round-Trip: RGB -> (CPU) BT.709-limited-encode in biplanar 4:2:0 ->
// (GPU) MSL-decode -> RGB. Bei uniformer Farbe muss das Original exakt zurückkommen
// (Chroma-Bilinear ist auf Flächen verlustfrei). Beweist Decode-Matrix + Sampler +
// kk_gpu-Plumbing. (libplacebo-Pixel-A/B folgt bei Integration in kk_render_image.)
#include "kk_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BT.709 limited 8-bit Decode (genau die Mathe, die der MSL-Kernel macht).
static const char *DECODE_MSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void yuv_decode(texture2d<float> luma   [[texture(0)]],\n"
"                       texture2d<float> chroma [[texture(1)]],\n"
"                       texture2d<float, access::write> dst [[texture(2)]],\n"
"                       sampler near [[sampler(0)]],\n"
"                       sampler lin  [[sampler(1)]],\n"
"                       uint2 id [[thread_position_in_grid]]) {\n"
"  uint w = dst.get_width(), h = dst.get_height();\n"
"  if (id.x >= w || id.y >= h) return;\n"
"  float2 uv = (float2(id) + 0.5) / float2(w, h);\n"
"  float Yr = luma.read(id).r;\n"
"  float2 C = chroma.sample(lin, uv).rg;   // Chroma bilinear auf Luma-Res\n"
"  // limited-range 8-bit normalisieren\n"
"  float Y  = (Yr * 255.0 - 16.0) / 219.0;\n"
"  float Cb = (C.r * 255.0 - 128.0) / 224.0;\n"
"  float Cr = (C.g * 255.0 - 128.0) / 224.0;\n"
"  // BT.709 YCbCr -> RGB\n"
"  float R = Y + 1.5748 * Cr;\n"
"  float G = Y - 0.1873 * Cb - 0.4681 * Cr;\n"
"  float B = Y + 1.8556 * Cb;\n"
"  dst.write(float4(R, G, B, 1.0), id);\n"
"}\n";

// Toleranz: 8-bit-Round-Trip (Y+Cb+Output je quantisiert) akkumuliert ~2-3 LSB.
static int approx(float a, float b) { return fabsf(a - b) <= 3.0f / 255.0f; }

int main(void) {
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { fprintf(stderr, "gpu create failed\n"); return 1; }

    struct { float r, g, b; const char *name; } colors[] = {
        {0.5f, 0.5f, 0.5f, "grau"}, {0.8f, 0.2f, 0.2f, "rot"},
        {0.2f, 0.7f, 0.3f, "gruen"}, {0.3f, 0.4f, 0.9f, "blau"},
    };
    const int W = 32, H = 32, CW = W/2, CH = H/2;
    int fails = 0;
    for (int c = 0; c < 4; c++) {
        float R = colors[c].r, G = colors[c].g, B = colors[c].b;
        // CPU-Encode BT.709 limited 8-bit
        float Y = 0.2126f*R + 0.7152f*G + 0.0722f*B;
        float Cb = (B - Y) / 1.8556f, Cr = (R - Y) / 1.5748f;
        unsigned char yv = (unsigned char)(16.0f + 219.0f*Y + 0.5f);
        unsigned char cbv = (unsigned char)(128.0f + 224.0f*Cb + 0.5f);
        unsigned char crv = (unsigned char)(128.0f + 224.0f*Cr + 0.5f);
        unsigned char *yp = malloc(W*H); for (int i=0;i<W*H;i++) yp[i]=yv;
        unsigned char *cp = malloc(CW*CH*2); for (int i=0;i<CW*CH;i++){cp[i*2]=cbv;cp[i*2+1]=crv;}
        kk_tex *luma = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE, yp);
        kk_tex *chroma = kk_tex_create(g, CW, CH, KK_FMT_RG8, KK_TEX_SAMPLE, cp);
        kk_tex *dst = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE|KK_TEX_DOWNLOAD, NULL);
        kk_compute_args a = { .out = dst, .in = { luma, chroma }, .n_in = 2, .linear = { false, true } };
        if (!kk_gpu_compute(g, DECODE_MSL, "yuv_decode", &a)) { fprintf(stderr, "compute failed\n"); return 1; }
        kk_gpu_finish(g);
        unsigned char *out = malloc(W*H*4);
        kk_tex_download(g, dst, out);
        float gr = out[0]/255.0f, gg = out[1]/255.0f, gb = out[2]/255.0f;
        int ok = approx(gr,R) && approx(gg,G) && approx(gb,B);
        printf("  %-6s in(%.2f,%.2f,%.2f) -> out(%.2f,%.2f,%.2f)  %s\n",
               colors[c].name, R,G,B, gr,gg,gb, ok?"PASS":"FAIL");
        if (!ok) fails++;
        free(yp); free(cp); free(out);
        kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&dst);
    }
    kk_gpu_destroy(&g);
    printf("kk_decode round-trip: %s\n", fails ? "FAIL" : "PASS (alle Farben)");
    return fails ? 1 : 0;
}

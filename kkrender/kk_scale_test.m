// kk_scale_test.m — Etappe-3-Baustein: separabler Lanczos-Scaler (2-Pass) auf kk_gpu.
// Verifiziert die Convolution-Mechanik gegen eine CPU-Lanczos-Referenz (gleiche Mathe
// -> muss matchen). libplacebo-EXAKT-Match (LUT/Antiring) folgt bei Integration via kk_ab.
#include "kk_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Separabler Lanczos-Pass (a=3, 6 Taps). axis=0: X skalieren, axis=1: Y. RGBA16F
// in/out (Linearlicht-Zwischentextur). scale per uniform. Clamp-to-edge an den Rändern.
static const char *LANCZOS_MSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct P { float scale; uint axis; };\n"
"static inline float sinc(float x){ if (x==0.0) return 1.0; x*=M_PI_F; return sin(x)/x; }\n"
"static inline float lanczos3(float x){ x=abs(x); if (x>=3.0) return 0.0; return sinc(x)*sinc(x/3.0); }\n"
"kernel void lanczos(texture2d<float> src [[texture(0)]],\n"
"                    texture2d<float, access::write> dst [[texture(1)]],\n"
"                    constant P& p [[buffer(0)]],\n"
"                    uint2 id [[thread_position_in_grid]]) {\n"
"  uint W=dst.get_width(), H=dst.get_height(); if (id.x>=W||id.y>=H) return;\n"
"  int sw=int(src.get_width()), sh=int(src.get_height());\n"
"  float coord = (p.axis==0u ? float(id.x) : float(id.y));\n"
"  float s = (coord + 0.5)/p.scale - 0.5;\n"
"  int base = int(floor(s));\n"
"  float4 acc = float4(0.0); float wsum=0.0;\n"
"  for (int t=-2; t<=3; t++){\n"
"    int tap = base + t; float w = lanczos3(s - float(tap));\n"
"    int cx = (p.axis==0u) ? clamp(tap,0,sw-1) : int(id.x);\n"
"    int cy = (p.axis==1u) ? clamp(tap,0,sh-1) : int(id.y);\n"
"    acc += w * src.read(uint2(cx,cy)); wsum += w;\n"
"  }\n"
"  dst.write(acc/wsum, id);\n"
"}\n";

static float sinc(float x){ if(x==0)return 1; x*=(float)M_PI; return sinf(x)/x; }
static float lanc(float x){ x=fabsf(x); if(x>=3)return 0; return sinc(x)*sinc(x/3.0f); }

int main(void){
    kk_gpu *g = kk_gpu_create(NULL);
    if(!g){fprintf(stderr,"gpu fail\n");return 1;}
    const int SW=24, SH=24; float scale=2.0f; int DW=(int)(SW*scale), DH=(int)(SH*scale);
    // Test-Pattern: Rampe + Karo (rgba16f als float vorbereiten -> r8-Upload geht nicht,
    // also rgba16f via initial_data nicht trivial; wir nutzen R8 input, RGBA-intern egal).
    // Einfacher: Input als RGBA8 (jeder Pixel grau = x-Rampe), CPU+GPU gleich behandeln.
    float *in = malloc(sizeof(float)*SW*SH*4);
    for(int y=0;y<SH;y++)for(int x=0;x<SW;x++){ float v=(float)x/(SW-1); int i=(y*SW+x)*4; in[i]=v;in[i+1]=v;in[i+2]=v;in[i+3]=1; }
    // RGBA16F-Textur via Upload (half-floats wären nötig); zur Verifikation reicht RGBA8.
    unsigned char *in8=malloc(SW*SH*4); for(int i=0;i<SW*SH*4;i++) in8[i]=(unsigned char)(in[i]*255+0.5f);
    kk_tex *src = kk_tex_create(g,SW,SH,KK_FMT_RGBA8,KK_TEX_SAMPLE,in8);
    kk_tex *tmp = kk_tex_create(g,DW,SH,KK_FMT_RGBA8,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
    kk_tex *dst = kk_tex_create(g,DW,DH,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
    struct { float scale; unsigned int axis; } px = {scale,0}, py={scale,1};
    kk_compute_args ax={.out=tmp,.in={src},.n_in=1};   // X-Pass
    kk_gpu_compute(g,LANCZOS_MSL,"lanczos",&(kk_compute_args){.out=tmp,.in={src},.n_in=1,.uniforms=&px,.uniforms_size=sizeof px});
    kk_gpu_compute(g,LANCZOS_MSL,"lanczos",&(kk_compute_args){.out=dst,.in={tmp},.n_in=1,.uniforms=&py,.uniforms_size=sizeof py});
    (void)ax;
    kk_gpu_finish(g);
    unsigned char *out=malloc(DW*DH*4); kk_tex_download(g,dst,out);
    // CPU-Referenz: dieselbe separable Lanczos (X dann Y) auf in8.
    float *cx=malloc(sizeof(float)*DW*SH); // X-skaliert, 1 Kanal (grau) reicht
    for(int y=0;y<SH;y++)for(int x=0;x<DW;x++){ float s=(x+0.5f)/scale-0.5f; int b=(int)floorf(s); float a=0,ws=0;
        for(int t=-2;t<=3;t++){int tp=b+t; int cxi=tp<0?0:tp>SW-1?SW-1:tp; float w=lanc(s-tp); a+=w*(in8[(y*SW+cxi)*4]/255.0f); ws+=w;} cx[y*DW+x]=a/ws; }
    int fails=0; float maxerr=0;
    for(int y=0;y<DH;y++)for(int x=0;x<DW;x++){ float s=(y+0.5f)/scale-0.5f; int b=(int)floorf(s); float a=0,ws=0;
        for(int t=-2;t<=3;t++){int tp=b+t; int cyi=tp<0?0:tp>SH-1?SH-1:tp; float w=lanc(s-tp); a+=w*cx[cyi*DW+x]; ws+=w;}
        float ref=a/ws; float got=out[(y*DW+x)*4]/255.0f; float e=fabsf(ref-got); if(e>maxerr)maxerr=e; if(e>3.0f/255.0f)fails++; }
    printf("kk_scale Lanczos %dx%d->%dx%d: maxerr=%.4f (%.1f LSB)  %s\n",SW,SH,DW,DH,maxerr,maxerr*255,
           fails?"FAIL":"PASS");
    kk_gpu_destroy(&g);
    return fails?1:0;
}

// ⚠️ NACHTRAG 2026-09-01: Dieser Prüfstand trägt eine EIGENE Kopie des
// Lanczos-Kernels (LANCZOS_MSL oben) und ruft KEINEN Produktionscode auf. Als
// kk_gpu_render.c im Juli 2026 auf eine gebackene 64er-LUT umgestellt wurde
// (renderpl.70), blieb er trotzdem grün — er prüfte weiter seine eigene
// sin()-Variante. Er beweist damit die MECHANIK (Convolution, 2-Pass, Sampler),
// nicht den ausgelieferten Kernel.
// Den echten Pfad prüft `kk_lut_test.m`. Dasselbe gilt sinngemäß für die vier
// anderen Prüfstände aus der Bauphase.

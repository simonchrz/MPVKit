// kk_gpu_render.c — GATED Prod-SDR-Render-Pfad auf kk_gpu (libplacebo-frei).
// Strangler-Einstieg: kuckuck_hybrid_render ruft kk_gpu_render() bei KUCKUCK_KK_GPU=1;
// handhabt NUR SDR-NV12 (sonst false -> Fallback auf pl_render_image). Pipeline (alle
// Pässe headless gegen libplacebo verifiziert): YUV-Decode(709 limited) -> linearize
// (BT.1886) -> Lanczos-Scale-in-Linear -> delinearize(sRGB) -> eigener BGRA-Output ->
// Blit zum Display-Target (umgeht ShaderWrite-Usage-Frage).
//
// ⚠️ STARTCODE — ON-DEVICE ZU VALIDIEREN + ERWEITERN. Default AUS (Env-gated).
// Offen für Voll-Deckung: echte Color-Params (hybrid_trc/prim/matrix statt 709/1886/sRGB
// hartcodiert) · Sigmoid-Upscale · Deband · HDR (P010 -> IPT-color_map) · CNN-Hooks ·
// EWA statt Lanczos. Alle als verifizierte Bausteine vorhanden (iq-harness/kk_*_ab).
#include "kk_gpu.h"
#include <CoreVideo/CoreVideo.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
extern kk_tex *kk_gpu_anime4k(kk_gpu *gpu, kk_tex *in, const char *weights_path);
extern kk_tex *kk_gpu_artcnn(kk_gpu *gpu, kk_tex *luma, const char *weights_path);
// CNN-Cache-Freigabe (kk_gpu_cnn.c) — nicht-aktive Pfade freigeben (Speicher).
extern void kk_gpu_anime4k_release(kk_gpu *gpu);
extern void kk_gpu_artcnn_release(kk_gpu *gpu);
// Cache-Freigabe pro Domäne (Def. am Dateiende) — nur der aktive Pfad bleibt resident.
static void kk_gpu_sdr_release(kk_gpu *g);
static void kk_gpu_hdr_release(kk_gpu *g);

// Chroma-Abtastung für DEC/DECLIN (zwei Varianten per #define, wie der Dither):
//  KK_CHROMA_L3 0: bilinear direkt aus der Chroma-Ebene (Stand bis renderpl.73).
//  KK_CHROMA_L3 1: Lanczos3, separabel. `chroma` ist dann die schon HORIZONTAL
//                  hochskalierte Stufe aus CHH (Ausgabebreite × Chroma-Höhe); hier
//                  laufen nur noch die 6 vertikalen Taps. Gemessen 2026-09-22 an
//                  echten Bildern (Referenz mit nativer Chroma-Auflösung): Farbfehler
//                  an Kanten VOX −22 %, Disney-Cartoon −34 %, ZDF-720p −26 %, auch in
//                  den Ausreißern besser (kein sichtbares Überschwingen).
// co = Chroma-Ort-Versatz in CHROMA-Texeln (kk_chroma_offset): mittig 0, left-sited
// +0,25 horizontal. In Texel-Einheiten, damit er auch für den 2×-Luma-Eingang des
// ArtCNN-Pfads stimmt (dort sind es 1,0 statt 0,5 Luma-Pixel).
#define KK_CHROMA_FN \
"static inline float kk_l3(float x){ x=abs(x); if(x<1e-5) return 1.0; if(x>=3.0) return 0.0;\n" \
"  return 3.0*sinpi(x)*sinpi(x/3.0)/(M_PI_F*M_PI_F*x*x); }\n" \
"static inline float2 kk_chroma(texture2d<float> chroma, sampler lin, uint2 id, uint w, uint h, float cox, float coy){\n" \
"#if KK_CHROMA_L3\n" \
"  int ch=int(chroma.get_height()); float s=(float(id.y)+0.5)*float(ch)/float(h)-0.5+coy; int b=int(floor(s));\n" \
"  float2 acc=float2(0.0); float ws=0.0;\n" \
"  for(int t=-2;t<=3;t++){ int tap=b+t; float wt=kk_l3(s-float(tap));\n" \
"    acc+=wt*chroma.read(uint2(id.x,uint(clamp(tap,0,ch-1)))).rg; ws+=wt; }\n" \
"  return acc/ws;\n" \
"#else\n" \
"  float2 uv=(float2(id)+0.5)/float2(w,h);\n" \
"  return chroma.sample(lin,uv+float2(cox,coy)/float2(chroma.get_width(),chroma.get_height())).rg;\n" \
"#endif\n" \
"}\n"
// CHH: Chroma horizontal auf die Ausgabebreite (Lanczos3, 6 Taps), Zeilen bleiben in
// Chroma-Höhe. Beliebiges Verhältnis (2× für DEC/DECLIN, 4× für den ArtCNN-2×-Eingang).
// Ziel RG16Unorm: Chroma liegt bei 16..240/255, das lässt Luft fürs Überschwingen.
static const char *CHH_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{float cox;};\n"
KK_CHROMA_FN
"kernel void chh(texture2d<float> c [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  int cw=int(c.get_width()); float s=(float(id.x)+0.5)*float(cw)/float(W)-0.5+p.cox; int b=int(floor(s));\n"
"  float2 acc=float2(0.0); float ws=0.0;\n"
"  for(int t=-2;t<=3;t++){ int tap=b+t; float wt=kk_l3(s-float(tap)); acc+=wt*c.read(uint2(uint(clamp(tap,0,cw-1)),id.y)).rg; ws+=wt; }\n"
"  dst.write(float4(acc/ws,0.0,1.0),id);}\n";

// DEC: YUV -> encodete RGB (exakte Matrix via pl_color_repr_decode, 9+3). KEIN linearize
// (Deband sitzt auf der encodeten Quelle, wie libplacebos source-deband).
#define DEC_SRC \
"#include <metal_stdlib>\nusing namespace metal;\nstruct D{float m[12];float co[2];};\n" \
KK_CHROMA_FN \
"kernel void dec(texture2d<float> luma [[texture(0)]], texture2d<float> chroma [[texture(1)]],\n" \
"  texture2d<float,access::write> dst [[texture(2)]], sampler near [[sampler(0)]], sampler lin [[sampler(1)]],\n" \
"  constant D& d [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n" \
"  float Y=luma.read(id).r; float2 C=kk_chroma(chroma,lin,id,w,h,d.co[0],d.co[1]);\n" \
"  float3 v=float3(Y,C.r,C.g);\n" \
"  float3 rgb=float3(d.m[0]*v.x+d.m[1]*v.y+d.m[2]*v.z, d.m[3]*v.x+d.m[4]*v.y+d.m[5]*v.z, d.m[6]*v.x+d.m[7]*v.y+d.m[8]*v.z)+float3(d.m[9],d.m[10],d.m[11]);\n" \
"  dst.write(float4(rgb,1.0),id);}\n"
static const char *DEC_MSL    = "#define KK_CHROMA_L3 0\n" DEC_SRC;
static const char *DEC_L3_MSL = "#define KK_CHROMA_L3 1\n" DEC_SRC;   // chroma = CHH-Stufe
// DECLIN: DEC+LIN fusioniert (HD-Light, kein Deband/CNN dazwischen -> c_dec-Roundtrip
// in voller Quellauflösung gespart). Mathematisch identisch zu dec->lin in Serie.
#define DECLIN_SRC \
"#include <metal_stdlib>\nusing namespace metal;\nstruct DL{float d[12];float a,b;float m[9];float o;float co[2];};\n" \
KK_CHROMA_FN \
"kernel void declin(texture2d<float> luma [[texture(0)]], texture2d<float> chroma [[texture(1)]],\n" \
"  texture2d<float,access::write> dst [[texture(2)]], sampler near [[sampler(0)]], sampler lin [[sampler(1)]],\n" \
"  constant DL& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n" \
"  float Y=luma.read(id).r; float2 C=kk_chroma(chroma,lin,id,w,h,p.co[0],p.co[1]);\n" \
"  float3 v=float3(Y,C.r,C.g);\n" \
"  float3 rgb=float3(p.d[0]*v.x+p.d[1]*v.y+p.d[2]*v.z, p.d[3]*v.x+p.d[4]*v.y+p.d[5]*v.z, p.d[6]*v.x+p.d[7]*v.y+p.d[8]*v.z)+float3(p.d[9],p.d[10],p.d[11]);\n" \
"  float3 c=max(rgb,0.0); float3 vl=p.a*pow(c+p.b,float3(2.4))-p.o;\n" \
"  float3 o=float3(p.m[0]*vl.x+p.m[1]*vl.y+p.m[2]*vl.z, p.m[3]*vl.x+p.m[4]*vl.y+p.m[5]*vl.z, p.m[6]*vl.x+p.m[7]*vl.y+p.m[8]*vl.z);\n" \
"  dst.write(float4(o,1.0),id);}\n"
static const char *DECLIN_MSL    = "#define KK_CHROMA_L3 0\n" DECLIN_SRC;
static const char *DECLIN_L3_MSL = "#define KK_CHROMA_L3 1\n" DECLIN_SRC;   // chroma = CHH-Stufe
// DEBLOCK: separabler 1D-Bilateral (±3, Luma-Range-gewichtet) auf der encodeten
// Quelle, Port von Resources/deblock_cas.glsl (App). Fuer SD-Privatsender am
// Kabel-Tuner (VOX/RTL ~2,4-2,9 Mbit/s): die 16-px-DCT-Kacheln sind in die Pixel
// eingebacken, man kann sie nur glaetten. ±1 war nachweislich wirkungslos (glaettet
// nur die 1-px-Stufe), erst ±3 reicht in beide Nachbarkacheln. Raeumliche Gewichte
// 0,88/0,61/0,32 ≈ Gauss σ≈2; SIGMA_R 0,12 (~30/255) trennt Block von Kante.
// Zwei Dispatches (axis 0 = H, 1 = V), 7+7 statt 49 Taps. Gate: ~deblock.
// ⚠️ Bis 2026-09-02 lief die GLSL-Datei NIE: kk_gpu liest keine GLSL, der Pfad
// ist nur ein Substring-Schalter — „deblock_cas" traf nur „cas".
static const char *DEBLOCK_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{uint axis;};\n#define SIGMA_R 0.12\n"
"static inline float dl_luma(float3 c){ return dot(c,float3(0.299,0.587,0.114)); }\n"
"static inline float dl_w(float ln,float le,float sw){ float d=ln-le; return sw*exp(-(d*d)/(2.0*SIGMA_R*SIGMA_R)); }\n"
"kernel void deblock(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  int2 c=int2(id); int2 st=(p.axis==0u)?int2(1,0):int2(0,1); int2 mx=int2(int(W)-1,int(H)-1);\n"
"#define T(k) src.read(uint2(clamp(c+(k)*st,int2(0),mx))).rgb\n"
"  float3 e=T(0); float le=dl_luma(e);\n"
"  float3 p1=T(1),p2=T(2),p3=T(3),m1=T(-1),m2=T(-2),m3=T(-3);\n"
"  float w1=dl_w(dl_luma(p1),le,0.88),w2=dl_w(dl_luma(p2),le,0.61),w3=dl_w(dl_luma(p3),le,0.32);\n"
"  float v1=dl_w(dl_luma(m1),le,0.88),v2=dl_w(dl_luma(m2),le,0.61),v3=dl_w(dl_luma(m3),le,0.32);\n"
"  float3 sum=e+p1*w1+p2*w2+p3*w3+m1*v1+m2*v2+m3*v3; float ws=1.0+w1+w2+w3+v1+v2+v3;\n"
"  dst.write(float4(sum/ws,1.0),id);}\n";
// DEBLOCK_Y: derselbe separable Bilateral (±3, SIGMA_R 0,12), aber auf der LUMA-
// EBENE eines NV12-Buffers — für den Lauf VOR VT-SR. Mit VT-SR entfällt der RGB-
// Deblock oben bewusst (er liefe NACH der 1,5×-Skalierung, Kachelraster 16→24 px,
// ±3 zu kurz). Hier sieht der Filter das Original-Raster. Chroma wird kopiert.
static const char *DEBLOCK_Y_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{uint axis;};\n#define SIGMA_R 0.12\n"
"static inline float dy_w(float ln,float le,float sw){ float d=ln-le; return sw*exp(-(d*d)/(2.0*SIGMA_R*SIGMA_R)); }\n"
"kernel void deblock_y(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  int2 c=int2(id); int2 st=(p.axis==0u)?int2(1,0):int2(0,1); int2 mx=int2(int(W)-1,int(H)-1);\n"
"#define TY(k) src.read(uint2(clamp(c+(k)*st,int2(0),mx))).r\n"
"  float e=TY(0);\n"
"  float p1=TY(1),p2=TY(2),p3=TY(3),m1=TY(-1),m2=TY(-2),m3=TY(-3);\n"
"  float w1=dy_w(p1,e,0.88),w2=dy_w(p2,e,0.61),w3=dy_w(p3,e,0.32);\n"
"  float v1=dy_w(m1,e,0.88),v2=dy_w(m2,e,0.61),v3=dy_w(m3,e,0.32);\n"
"  float sum=e+p1*w1+p2*w2+p3*w3+m1*v1+m2*v2+m3*v3; float ws=1.0+w1+w2+w3+v1+v2+v3;\n"
"  dst.write(float4(sum/ws,0.0,0.0,1.0),id);}\n";
// DEBAND: pl_shader_deband (pcg3d-PRNG, 4-Sample-Quarter-Turn-Smoothing + Grain),
// headless gegen libplacebo verifiziert (mild/strong 0.1 LSB). radius/threshold/grain/
// iters/index als Uniform (threshold/grain = param/1000).
static const char *DEBAND_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct DB{float radius,threshold,grain;uint iters,index;};\n"
"static inline float3 rnd(thread uint3& s){ s=1664525u*s+uint3(1013904223u);\n"
"  s.x+=s.y*s.z; s.y+=s.z*s.x; s.z+=s.x*s.y; s^=s>>16u;\n"
"  s.x+=s.y*s.z; s.y+=s.z*s.x; s.z+=s.x*s.y; return float3(s)*(1.0/float(0xFFFFFFFFu)); }\n"
"kernel void deband(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  sampler near [[sampler(0)]], sampler lin [[sampler(1)]], constant DB& p [[buffer(0)]],\n"
"  uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  float2 sz=float2(W,H); float2 pos=(float2(id)+0.5)/sz; float2 pt=1.0/sz;\n"
"#define GET(X,Y) (src.sample(lin, pos+pt*float2(X,Y)).rgb)\n"
"  uint3 s=uint3(id.x,id.y,p.index); float3 res=src.sample(lin,pos).rgb;\n"
"  for(uint i=1;i<=p.iters;i++){ float3 r=rnd(s); float2 d=r.xy*float2(float(i)*p.radius,6.283185307);\n"
"    d=d.x*float2(cos(d.y),sin(d.y));\n"
"    float3 avg=(GET(+d.x,+d.y)+GET(-d.x,+d.y)+GET(-d.x,-d.y)+GET(+d.x,-d.y))*0.25;\n"
"    float3 diff=abs(res-avg); float bound=p.threshold/float(i); res=select(avg,res,diff>float3(bound)); }\n"
"  if(p.grain>0.0){ float3 r2=rnd(s); float3 st=min(abs(res),float3(p.grain)); res+=st*(r2-0.5); }\n"
"  dst.write(float4(res,1.0),id);}\n";
// LIN: BT.1886 a*pow(c+b,2.4) (encoded -> linear) + Primaries-Gamut (601/2020 -> 709,
// RGB-RGB-Matrix in Linear-Light; identity bei 709-Quelle = no-op).
// `- o`: Schwarzpunkt-Ausgleich (kk_lin_params). Mit o=0 und dem rohen a exakt wie vorher.
static const char *LIN_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct L{float a,b;float m[9];float o;};\n"
"kernel void lin(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant L& l [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  float3 c=max(src.read(id).rgb,0.0); float3 v=l.a*pow(c+l.b,float3(2.4))-l.o;\n"
"  float3 o=float3(l.m[0]*v.x+l.m[1]*v.y+l.m[2]*v.z, l.m[3]*v.x+l.m[4]*v.y+l.m[5]*v.z, l.m[6]*v.x+l.m[7]*v.y+l.m[8]*v.z);\n"
"  dst.write(float4(o,1.0),id);}\n";
static const char *LANCZOS_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{float scale;uint axis;float lut[64];};\n"
// Lanczos3-Gewichte als 64er-LUT (host-seitig gebacken, s. kk_lanczos_params) statt
// 2x sin() pro Tap — gleiche LUT-Dichte/-Interpolation wie der verifizierte EWA-Pfad;
// lut[63]=l3(3.0)=0, min() clampt Distanzen >=3 exakt auf 0.
"static inline float l3lut(constant P& p, float x){ x=min(abs(x),3.0)*(63.0/3.0);\n"
"  int i0=int(x); return mix(p.lut[i0], p.lut[min(i0+1,63)], x-float(i0)); }\n"
"kernel void lanczos(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  int sw=int(src.get_width()),sh=int(src.get_height()); float coord=(p.axis==0u?float(id.x):float(id.y));\n"
// Band-limitiert für Downscale (HDR 1440p -> Display ~0,84x): Filter in den Quellraum
// gestreckt (sf<1, Fenster 3/sf). Upscale (sf=1): Gewichte identisch zu vorher — die
// alten Rand-Taps -3/+4 hatten exakt Gewicht 0 (l3>=3 -> 0), R=3 lässt sie nur weg.
"  float sf=min(p.scale,1.0); int R=int(ceil(3.0/sf));\n"
"  float s=(coord+0.5)/p.scale-0.5; int base=int(floor(s)); float4 acc=float4(0.0); float wsum=0.0;\n"
"  for(int t=1-R;t<=R;t++){ int tap=base+t; float w=l3lut(p,(s-float(tap))*sf);\n"
"    int cx=(p.axis==0u)?clamp(tap,0,sw-1):int(id.x); int cy=(p.axis==1u)?clamp(tap,0,sh-1):int(id.y);\n"
"    acc+=w*src.read(uint2(cx,cy)); wsum+=w; } dst.write(acc/wsum,id);}\n";
// Host-Seite: Lanczos-Uniforms inkl. gebackener l3-LUT (einmal berechnet, dann memcpy).
typedef struct { float scale; uint32_t axis; float lut[64]; } kk_lanczos_p;
static kk_lanczos_p kk_lanczos_params(float scale, uint32_t axis) {
    static float tbl[64]; static bool init = false;
    if (!init) {
        for (int i = 0; i < 64; i++) {
            double x = 3.0 * i / 63.0;
            double s1 = (x == 0.0) ? 1.0 : sin(M_PI * x) / (M_PI * x);
            double x3 = x / 3.0;
            double s3 = (x3 == 0.0) ? 1.0 : sin(M_PI * x3) / (M_PI * x3);
            tbl[i] = (float)(s1 * s3);
        }
        tbl[63] = 0.0f;   // exakt 0 am Fensterrand (l3(3.0))
        init = true;
    }
    kk_lanczos_p p; p.scale = scale; p.axis = axis;
    memcpy(p.lut, tbl, sizeof tbl);
    return p;
}
#define DELIN_SRC \
"#include <metal_stdlib>\nusing namespace metal;\n" \
"static inline float3 kk_dither(float3 v, uint2 p){\n" \
"#if KK_DITHER\n" \
"  uint x=p.x*1973u+p.y*9277u; x^=x>>15; x*=0x2c1b3c6du; x^=x>>12; x*=0x297a2d39u; x^=x>>15;\n" \
"  float n=(float(x&0xFFFFu)+float(x>>16))*(1.0/65536.0)-1.0;   // TPDF, -1..+1 LSB\n" \
"  return saturate(v+n*(1.0/255.0));\n" \
"#else\n" \
"  return v;\n" \
"#endif\n" \
"}\n" \
"static inline float srgb(float c){ return c<=0.0031308 ? 12.92*c : 1.055*pow(c,1.0/2.4)-0.055; }\n" \
"kernel void delin(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n" \
"  uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n" \
"  float3 c=clamp(src.read(id).rgb,0.0,1.0); dst.write(float4(kk_dither(float3(srgb(c.r),srgb(c.g),srgb(c.b)),id),1.0),id);}\n"
static const char *DELIN_MSL   = "#define KK_DITHER 0\n" DELIN_SRC;
static const char *DELIN_D_MSL = "#define KK_DITHER 1\n" DELIN_SRC;   // + Dither (8-Bit-Ziel)
// CAS (FidelityFX Contrast-Adaptive-Sharpening, cas.glsl-Port, headless gegen libplacebo
// maxerr≤3) auf der encodeten sRGB-Ausgabe (HD/Tuner-Sharpener, gated via ~cas).
#define CAS_SRC \
"#include <metal_stdlib>\nusing namespace metal;\n#define SHARP 0.5\n" \
"static inline float3 kk_dither(float3 v, uint2 p){\n" \
"#if KK_DITHER\n" \
"  uint x=p.x*1973u+p.y*9277u; x^=x>>15; x*=0x2c1b3c6du; x^=x>>12; x*=0x297a2d39u; x^=x>>15;\n" \
"  float n=(float(x&0xFFFFu)+float(x>>16))*(1.0/65536.0)-1.0;   // TPDF, -1..+1 LSB\n" \
"  return saturate(v+n*(1.0/255.0));\n" \
"#else\n" \
"  return v;\n" \
"#endif\n" \
"}\n" \
"kernel void cas(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n" \
"  uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n" \
"  int sw=int(W),sh=int(H); int2 p=int2(id);\n" \
"#define T(dx,dy) src.read(uint2(clamp(p.x+(dx),0,sw-1),clamp(p.y+(dy),0,sh-1))).rgb\n" \
"  float3 a=T(-1,-1),b=T(0,-1),c=T(1,-1),d=T(-1,0),e=T(0,0),f=T(1,0),g=T(-1,1),h=T(0,1),i=T(1,1);\n" \
"  float3 mn=min(min(min(d,e),min(f,b)),h); float3 mn2=min(mn,min(min(a,c),min(g,i))); mn+=mn2;\n" \
"  float3 mx=max(max(max(d,e),max(f,b)),h); float3 mx2=max(mx,max(max(a,c),max(g,i))); mx+=mx2;\n" \
"  float3 rcpM=1.0/max(mx,float3(1e-5)); float3 amp=clamp(min(mn,2.0-mx)*rcpM,0.0,1.0); amp=rsqrt(max(amp,float3(1e-5)));\n" \
"  float peak=-3.0*SHARP+8.0; float3 w=-1.0/(amp*peak); float3 rcpW=1.0/(1.0+4.0*w);\n" \
"  float3 win=(b+d)+(f+h); float3 o=clamp((win*w+e)*rcpW,0.0,1.0); dst.write(float4(kk_dither(o,id),1.0),id);}\n"
static const char *CAS_MSL   = "#define KK_DITHER 0\n" CAS_SRC;
static const char *CAS_D_MSL = "#define KK_DITHER 1\n" CAS_SRC;   // + Dither (8-Bit-Ziel)
// DELINCAS: Delin+CAS fusioniert (HD-Light) — der c_srgb-Roundtrip in voller Output-
// Auflösung entfällt; das sRGB-Encode läuft pro Tap inline (9x3 pow, ALU gegen
// Bandbreite getauscht — der Renderer ist bandbreiten-gebunden). Identische Mathe.
#define DELINCAS_SRC \
"#include <metal_stdlib>\nusing namespace metal;\n#define SHARP 0.5\n" \
"static inline float3 kk_dither(float3 v, uint2 p){\n" \
"#if KK_DITHER\n" \
"  uint x=p.x*1973u+p.y*9277u; x^=x>>15; x*=0x2c1b3c6du; x^=x>>12; x*=0x297a2d39u; x^=x>>15;\n" \
"  float n=(float(x&0xFFFFu)+float(x>>16))*(1.0/65536.0)-1.0;   // TPDF, -1..+1 LSB\n" \
"  return saturate(v+n*(1.0/255.0));\n" \
"#else\n" \
"  return v;\n" \
"#endif\n" \
"}\n" \
"static inline float srgb(float c){ return c<=0.0031308 ? 12.92*c : 1.055*pow(c,1.0/2.4)-0.055; }\n" \
"static inline float3 srgb3(float3 c){ c=clamp(c,0.0,1.0); return float3(srgb(c.r),srgb(c.g),srgb(c.b)); }\n" \
"kernel void delincas(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n" \
"  uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n" \
"  int sw=int(W),sh=int(H); int2 p=int2(id);\n" \
"#define T(dx,dy) srgb3(src.read(uint2(clamp(p.x+(dx),0,sw-1),clamp(p.y+(dy),0,sh-1))).rgb)\n" \
"  float3 a=T(-1,-1),b=T(0,-1),c=T(1,-1),d=T(-1,0),e=T(0,0),f=T(1,0),g=T(-1,1),h=T(0,1),i=T(1,1);\n" \
"  float3 mn=min(min(min(d,e),min(f,b)),h); float3 mn2=min(mn,min(min(a,c),min(g,i))); mn+=mn2;\n" \
"  float3 mx=max(max(max(d,e),max(f,b)),h); float3 mx2=max(mx,max(max(a,c),max(g,i))); mx+=mx2;\n" \
"  float3 rcpM=1.0/max(mx,float3(1e-5)); float3 amp=clamp(min(mn,2.0-mx)*rcpM,0.0,1.0); amp=rsqrt(max(amp,float3(1e-5)));\n" \
"  float peak=-3.0*SHARP+8.0; float3 w=-1.0/(amp*peak); float3 rcpW=1.0/(1.0+4.0*w);\n" \
"  float3 win=(b+d)+(f+h); float3 o=clamp((win*w+e)*rcpW,0.0,1.0); dst.write(float4(kk_dither(o,id),1.0),id);}\n"
static const char *DELINCAS_MSL   = "#define KK_DITHER 0\n" DELINCAS_SRC;
static const char *DELINCAS_D_MSL = "#define KK_DITHER 1\n" DELINCAS_SRC;   // + Dither (8-Bit-Ziel)
// EWA-lanczossharp (libplacebos Default-Upscaler) — radiale Filter-LUT (pl_filter_generate,
// 64 Einträge) EINGEBACKEN (konstant, kein Runtime-libplacebo). Headless vs libplacebo verifiziert.
#define KK_EWA_RADIUS 3.17759895f
static const float KK_EWA_LUT[64]={
 1.00000000f,0.99628311f,0.98519361f,0.96691382f,0.94174308f,0.91009110f,0.87246895f,0.82947797f,
 0.78179675f,0.73016614f,0.67537409f,0.61823839f,0.55959070f,0.50025868f,0.44105068f,0.38273951f,
 0.32604882f,0.27163979f,0.22010081f,0.17193846f,0.12757106f,0.08732384f,0.05142748f,0.02001836f,
 -0.00685941f,-0.02924910f,-0.04727522f,-0.06113534f,-0.07109106f,-0.07745767f,-0.08059313f,-0.08088668f,
 -0.07874756f,-0.07459370f,-0.06884123f,-0.06189458f,-0.05413772f,-0.04592650f,-0.03758239f,-0.02938765f,
 -0.02158188f,-0.01436004f,-0.00787209f,-0.00222358f,0.00252217f,0.00634110f,0.00924443f,0.01127407f,
 0.01249739f,0.01300187f,0.01288953f,0.01227151f,0.01126289f,0.00997788f,0.00852562f,0.00700655f,
 0.00550949f,0.00410946f,0.00286628f,0.00182385f,0.00101015f,0.00043793f,0.00010582f,-0.00000000f};
// LIN+SIG fusioniert (Perf: ein Full-Res-Pass statt zwei — LIN und SIG sind beide
// reine Per-Pixel-Transformationen): BT.1886-Linearize + Primaries-Gamut + Sigmoidize
// (center=0.75 slope=6.5, libplacebos Default-Upscale-Pre). Ergebnis identisch zur
// LIN->SIG-Kette (SIG clampte eh auf [0,1]).
static const char *LINSIG_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct L{float a,b;float m[9];float o;};\n"
"kernel void linsig(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant L& l [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  float3 c=max(src.read(id).rgb,0.0); float3 v=l.a*pow(c+l.b,float3(2.4))-l.o;\n"
"  float3 o=float3(l.m[0]*v.x+l.m[1]*v.y+l.m[2]*v.z, l.m[3]*v.x+l.m[4]*v.y+l.m[5]*v.z, l.m[6]*v.x+l.m[7]*v.y+l.m[8]*v.z);\n"
"  o=clamp(o,0.0,1.0); o=0.75-(1.0/6.5)*log(1.0/(o*0.82796854+0.00757286)-1.0);\n"
"  dst.write(float4(o,1.0),id);}\n";
// EWA-polar-Gather (eingebackene LUT als Uniform).
static const char *EWA_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{float scale;uint lutn;float radius;float lut[64];};\n"
"kernel void ewa(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  int sw=int(src.get_width()),sh=int(src.get_height());\n"
"  float2 sc=(float2(id)+0.5)/p.scale-0.5;\n"
"  float sf=min(p.scale,1.0); float Rsrc=p.radius/sf;\n"
"  int2 lo=int2(floor(sc-Rsrc)), hi=int2(ceil(sc+Rsrc));\n"
"  float4 acc=float4(0.0); float wsum=0.0;\n"
"  for(int sy=lo.y;sy<=hi.y;sy++)for(int sx=lo.x;sx<=hi.x;sx++){\n"
"    float2 dv=(float2(sx,sy)-sc)*sf; float d=length(dv); if(d>=p.radius)continue;\n"
"    float fidx=d/p.radius*float(p.lutn-1); int i0=int(fidx); float fr=fidx-float(i0);\n"
"    float w=mix(p.lut[i0], p.lut[min(i0+1,int(p.lutn)-1)], fr);\n"
"    int cx=clamp(sx,0,sw-1),cy=clamp(sy,0,sh-1); acc+=w*src.read(uint2(cx,cy)); wsum+=w; }\n"
"  dst.write(wsum>0.0?acc/wsum:float4(0.0),id);}\n";
// Unsigmoidize + sRGB-Delin fusioniert (Upscale-Post).
#define DELINU_SRC \
"#include <metal_stdlib>\nusing namespace metal;\n" \
"static inline float3 kk_dither(float3 v, uint2 p){\n" \
"#if KK_DITHER\n" \
"  uint x=p.x*1973u+p.y*9277u; x^=x>>15; x*=0x2c1b3c6du; x^=x>>12; x*=0x297a2d39u; x^=x>>15;\n" \
"  float n=(float(x&0xFFFFu)+float(x>>16))*(1.0/65536.0)-1.0;   // TPDF, -1..+1 LSB\n" \
"  return saturate(v+n*(1.0/255.0));\n" \
"#else\n" \
"  return v;\n" \
"#endif\n" \
"}\n" \
"static inline float srgb(float c){ return c<=0.0031308 ? 12.92*c : 1.055*pow(c,1.0/2.4)-0.055; }\n" \
"kernel void delinu(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n" \
"  uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n" \
"  float3 c=src.read(id).rgb; c=1.20778572/(1.0+exp(6.5*(0.75-c)))-0.00914634; c=clamp(c,0.0,1.0);\n" \
"  dst.write(float4(kk_dither(float3(srgb(c.r),srgb(c.g),srgb(c.b)),id),1.0),id);}\n"
static const char *DELINU_MSL   = "#define KK_DITHER 0\n" DELINU_SRC;
static const char *DELINU_D_MSL = "#define KK_DITHER 1\n" DELINU_SRC;   // + Dither (8-Bit-Ziel)

// ===== HDR (P010 -> IPT-Tonemap -> PQ/2020-Output) =====
// MKPQ: BT.2020-limited-10bit-Decode (Y r16 + Chroma rg16) -> PQ-RGB -> PQ-EOTF ->
// linear (10000-norm). Aus kk_hdr_render_ab (verifiziert).
static const char *MKPQ_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"static inline float3 pqe(float3 e){ const float m1=0.1593017578125,m2=78.84375,c1=0.8359375,c2=18.8515625,c3=18.6875;\n"
"  float3 ep=pow(max(e,0.0),float3(1.0/m2)); float3 n=max(ep-c1,0.0); float3 d=c2-c3*ep; return pow(n/d,float3(1.0/m1)); }\n"
"struct K{float co[2];};\n"
"kernel void mk(texture2d<float> y [[texture(0)]],texture2d<float> c [[texture(1)]],texture2d<float,access::write> o [[texture(2)]],\n"
" sampler near [[sampler(0)]], sampler lin [[sampler(1)]], constant K& k [[buffer(0)]], uint2 id [[thread_position_in_grid]]){uint w=o.get_width(),h=o.get_height();if(id.x>=w||id.y>=h)return;\n"
" float2 uv=(float2(id)+0.5)/float2(w,h); float Yc=y.read(id).r*65535.0/64.0; float2 C=c.sample(lin,uv+float2(k.co[0],k.co[1])/float2(c.get_width(),c.get_height())).rg*65535.0/64.0;\n"
" float Y=(Yc-64.0)/876.0, Cb=(C.r-512.0)/896.0, Cr=(C.g-512.0)/896.0;\n"
" float3 rgb=clamp(float3(Y+1.4746*Cr, Y-0.16455*Cb-0.57135*Cr, Y+1.8814*Cb),0.0,1.0);\n"
" o.write(float4(pqe(rgb),1.0),id);}\n";
// CMHDR: IPT-color_map -> PQ/2020-Output (statt sRGB). Tone-Map I via LUT + Chroma-Hull;
// 3D-Gamut-LUT weggelassen (HDR->HDR, 2020->2020 = in-gamut; Hull+Clip). lms2rgb=2020,
// KEIN ×10000/SDRW (PQ-absolut), pq_oetf-Output. IPT-Machinerie aus kk_colormap_ab (verifiziert).
static const char *CMHDR_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"struct CM{float rgb2lms[9],lms2ipt[9],ipt2lms[9],lms2rgb[9];float in_min,in_max,out_min,out_max;float lut[256];};\n"
"#define M1 0.1593017578125\n#define M2 78.84375\n#define C1 0.8359375\n#define C2 18.8515625\n#define C3 18.6875\n"
"static inline float3 mul3(constant float* m, float3 v){return float3(m[0]*v.x+m[1]*v.y+m[2]*v.z, m[3]*v.x+m[4]*v.y+m[5]*v.z, m[6]*v.x+m[7]*v.y+m[8]*v.z);}\n"
"static inline float3 pq_oetf3(float3 l){ float3 lm=pow(max(l,0.0),float3(M1)); return pow((C1+C2*lm)/(1.0+C3*lm),float3(M2)); }\n"
"static inline float3 pq_eotf3(float3 e){ float3 ep=pow(max(e,0.0),float3(1.0/M2)); float3 num=max(ep-C1,0.0); float3 den=C2-C3*ep; return pow(num/den,float3(1.0/M1)); }\n"
"kernel void cmh(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant CM& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  float3 linr=src.read(id).rgb;\n"
"  float3 lms=mul3(p.rgb2lms,linr); float3 lmspq=pq_oetf3(lms); float3 ipt=mul3(p.lms2ipt,lmspq); float i_orig=ipt.x;\n"
"  float ipos=clamp((ipt.x-p.in_min)/(p.in_max-p.in_min),0.0,1.0);\n"
"  float fidx=clamp(ipos*256.0-0.5,0.0,255.0); int i0=int(fidx); float fr=fidx-float(i0); ipt.x=mix(p.lut[i0],p.lut[min(i0+1,255)],fr);\n"
"  float ix=max(ipt.x,1e-6); float2 hull=float2(i_orig,ix); hull=((hull-6.0)*hull+9.0)*hull; ipt.yz *= min(i_orig/ix, hull.y/hull.x);\n"
"  float3 lmspq2=mul3(p.ipt2lms,ipt); float3 lms2=pq_eotf3(lmspq2);\n"     // 10000-norm, KEIN SDRW-Scale
"  float3 o=clamp(mul3(p.lms2rgb,lms2),0.0,1.0); dst.write(float4(pq_oetf3(o),1.0),id);}\n"; // PQ/2020-Output

static kk_gpu *g_kk = NULL;  // lazy, teilt libplacebos Device

// Pass-Zeiten des zuletzt abgeschlossenen Frames — oeffentlicher Zugang fuer die
// App, die den internen kk_gpu nicht kennt. Nur mit KUCKUCK_PASS_TIMING=1 gefuellt.
int kuckuck_hybrid_pass_timings(const char **namen, double *ms, int max) {
    if (!g_kk) return 0;
    return kk_gpu_timings(g_kk, namen, ms, max);
}

// Async-Abschluss: die Frame-Wraps (luma/chroma/tgt inkl. CVMetalTextureRef!) MÜSSEN
// bis GPU-Completion leben (Pool-Recycling/Decoder-Write sonst mitten im GPU-Read) —
// deshalb hängen sie an diesem Heap-Kontext und werden im Completion-Handler
// freigegeben, DANN erst der Caller-Callback. done==NULL → synchroner finish-Pfad.
typedef struct { kk_gpu *g; kk_tex *a, *b, *c; void (*done)(void*); void *ud; } kk_done_ctx;
static void kk_render_done(void *p) {
    kk_done_ctx *d = (kk_done_ctx *) p;
    kk_tex_destroy(d->g, &d->a); kk_tex_destroy(d->g, &d->b); kk_tex_destroy(d->g, &d->c);
    if (d->done) d->done(d->ud);
    free(d);
}
static bool kk_finish_or_submit(kk_gpu *g, kk_tex **luma, kk_tex **chroma, kk_tex **tgt,
                                void (*done)(void*), void *ud) {
    if (!done) {
        kk_gpu_finish(g);
        kk_tex_destroy(g, luma); kk_tex_destroy(g, chroma); kk_tex_destroy(g, tgt);
        return true;
    }
    kk_done_ctx *d = calloc(1, sizeof *d);
    if (!d) {   // OOM-Fallback: synchron abschließen, Caller trotzdem benachrichtigen
        kk_gpu_finish(g);
        kk_tex_destroy(g, luma); kk_tex_destroy(g, chroma); kk_tex_destroy(g, tgt);
        done(ud);
        return true;
    }
    d->g = g; d->a = *luma; d->b = *chroma; d->c = *tgt; d->done = done; d->ud = ud;
    *luma = NULL; *chroma = NULL; *tgt = NULL;   // Ownership → Completion-Handler
    kk_gpu_submit(g, kk_render_done, d);
    return true;
}
// Gecachte Intermediates (über Frames wiederverwendet — KEIN Per-Frame-Alloc, sonst
// Jetsam-OOM durch Allok-Churn in Ziel-Auflösung). Re-create nur bei Dim-Wechsel.
static kk_tex *c_dec = NULL, *c_deb = NULL, *c_lin = NULL, *c_tmpx = NULL, *c_liny = NULL, *c_out = NULL;
static kk_tex *c_alin = NULL, *c_atmpx = NULL;   // Anime4K-Post (2×-Input -> Ziel)
static kk_tex *c_srgb = NULL;                    // CAS-Input (sRGB-Ausgabe vor Sharpen)
static kk_tex *c_a2rgb = NULL;                   // ArtCNN: 2×-Luma decodet zu encodeter RGB
// c_sig entfernt: LIN+SIG fusioniert (LINSIG_MSL) → kein separates Sigmoid-Intermediate
static kk_tex *c_adeb = NULL;                    // ArtCNN-Pfad: entbandete 2×-RGB
static kk_tex *c_dbl = NULL;                     // Deblock: H-Zwischenstufe (W×H), nur bei ~deblock
static int c_W = 0, c_H = 0, c_OW = 0, c_OH = 0;

// ---- Bildqualität (2026-09-22, gemessen mit kk_iq_probe) ------------------------
//
// SCHWARZPUNKT: BT.1886 mit Kontrast 1000:1 (a/b unten) legt Video-Schwarz auf
// 0,001 Linearlicht. DELIN kodierte das unverändert nach sRGB -> Y=16 landete bei
// Code 3 statt 0, auf OLED sichtbar grau neben den echten schwarzen Balken der
// Display-Ebene. Ausgleich (lin-Lb)/(1-Lb), gefaltet in a und einen Abzug o:
// Y=16 -> 0, Y=20 -> 3 (statt 6), ab Y~40 praktisch unverändert.
// KUCKUCK_BLACKPOINT=0 -> alter Zustand (o=0, rohes a).
static void kk_lin_params(float *a, float *o) {
    const float a0 = 0.8704f, b0 = 0.0595f;
    const char *e = getenv("KUCKUCK_BLACKPOINT");
    if (e && e[0] == '0') { *a = a0; *o = 0.0f; return; }
    float lb = a0 * powf(b0, 2.4f);
    *a = a0 / (1.0f - lb);
    *o = lb / (1.0f - lb);
}

// CHROMA-ORT: 4:2:0-Chroma sitzt bei H.264/HEVC-Broadcast LINKS (horizontal auf den
// geraden Luma-Spalten) — gemessen per ffprobe an Tuner, Aufnahmen und ZDF-Mediathek,
// alle `left`. Abgetastet wurde mittig -> Farbe 0,5 Luma-Pixel versetzt (Farbfehler
// an Kanten rms 62 -> 49 mit Korrektur). Rückgabe in CHROMA-Texeln (left = +0,25).
// Quelle: Pixelbuffer-Attachment; fehlt es, gilt left (Default der Codec-Specs für
// 4:2:0 und der von mpv). KUCKUCK_CHROMA_LOC=center|left erzwingt.
static void kk_chroma_offset(CVPixelBufferRef pb, float co[2]) {
    co[0] = 0.25f; co[1] = 0.0f;                  // left
    const char *e = getenv("KUCKUCK_CHROMA_LOC");
    if (e && e[0] == 'c') { co[0] = 0.0f; return; }
    if (e && e[0] == 'l') return;
    CFTypeRef loc = NULL;
    // Slice-Minimum ist iOS 14, die App läuft ab 27 — der Zweig ist dort immer wahr.
    if (__builtin_available(iOS 15.0, macOS 12.0, *))
        loc = CVBufferCopyAttachment(pb, kCVImageBufferChromaLocationTopFieldKey, NULL);
    if (!loc) return;
    if (CFGetTypeID(loc) == CFStringGetTypeID()) {
        CFStringRef l = (CFStringRef) loc;
        if      (CFEqual(l, kCVImageBufferChromaLocation_Center))     { co[0] = 0.0f;  co[1] = 0.0f; }
        else if (CFEqual(l, kCVImageBufferChromaLocation_Top))        { co[0] = 0.0f;  co[1] = 0.25f; }
        else if (CFEqual(l, kCVImageBufferChromaLocation_Bottom))     { co[0] = 0.0f;  co[1] = -0.25f; }
        else if (CFEqual(l, kCVImageBufferChromaLocation_TopLeft))    { co[0] = 0.25f; co[1] = 0.25f; }
        else if (CFEqual(l, kCVImageBufferChromaLocation_BottomLeft)) { co[0] = 0.25f; co[1] = -0.25f; }
        else if (CFEqual(l, kCVImageBufferChromaLocation_DV420))      { co[0] = 0.0f;  co[1] = 0.0f; }
        // Left und Unbekanntes: bleibt left
    }
    CFRelease(loc);
}

// DITHER: TPDF ±1 LSB vor der 8-Bit-Rundung, nur beim Schreiben ins BGRA8-Ziel
// (nie in eine Float-Zwischenstufe — CAS würde das Rauschen mitschärfen). Statisches
// Muster: kein Flimmern im Standbild. Gewinn klein (Flächen-Bias 0,57 -> 0,14 LSB),
// Quell-Stufen einer 8-Bit-Quelle kann Dither nicht beheben. KUCKUCK_DITHER=0 -> aus.
static bool kk_dither_an(void) {
    const char *e = getenv("KUCKUCK_DITHER");
    return !(e && e[0] == '0');
}

// CHROMA-HOCHSKALIERUNG: Lanczos3 separabel statt bilinear (s. KK_CHROMA_FN).
// Default nur AUSSERHALB von HD-Light (Quelle < 1080p): dort ist der Gewinn am größten
// (Cartoon-SD −34 %) und der Render billig. HD-Light bleibt bewusst leicht — auf dem
// Mac kostet CHH+DEC_L3 bei 1080p +0,21 ms (0,26 -> 0,47), auf dem A16 läge 1080p50-
// Live damit noch weiter über dem Budget. KUCKUCK_CHROMA_UP=bilinear|lanczos erzwingt.
// Die horizontale Stufe (Ausgabebreite × Chroma-Höhe) wird je Ziel gecacht; NULL =
// Allokation gescheitert -> bilinear.
static kk_tex *c_chh = NULL, *c_chh2 = NULL;   // DEC/DECLIN-Breite bzw. ArtCNN-2×-Breite
static bool kk_chroma_l3_an(bool hdLight) {
    const char *e = getenv("KUCKUCK_CHROMA_UP");
    if (e && e[0] == 'b') return false;
    if (e && e[0] == 'l') return true;
    return !hdLight;
}
static kk_tex *kk_chroma_h(kk_gpu *g, kk_tex *chroma, kk_tex **cache, int W, float cox) {
    int ch = kk_tex_h(chroma);
    if (*cache && (kk_tex_w(*cache) != W || kk_tex_h(*cache) != ch)) kk_tex_destroy(g, cache);
    if (!*cache) *cache = kk_tex_create(g, W, ch, KK_FMT_RG16, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
    if (!*cache) return NULL;
    struct { float cox; } P = { cox };
    kk_compute_args a = { .out = *cache, .in = { chroma }, .n_in = 1, .uniforms = &P, .uniforms_size = sizeof P };
    return kk_gpu_compute(g, CHH_MSL, "chh", &a) ? *cache : NULL;
}
static unsigned g_frame = 0;   // temporaler Grain-Index (Deband)

// yuv2rgb = 12 floats (9 Matrix row-major + 3 Offset) aus libplacebos pl_color_repr_decode
// (vom Hook übergeben — echte 601/709/2020-Matrix + Range). true = von kk_gpu gerendert.
bool kk_gpu_render(void *metal_device, void *cv_pixbuf, void *target_texture,
                   const float *yuv2rgb, const float *prim2disp,
                   void (*done)(void*), void *done_ud) {
    CVPixelBufferRef pb = (CVPixelBufferRef) cv_pixbuf;
    if (!pb || !target_texture) return false;
    OSType pf = CVPixelBufferGetPixelFormatType(pb);
    // Nur SDR-NV12 (8-bit biplanar). P010/HDR + andere -> Fallback.
    if (pf != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange &&
        pf != kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        return false;
    IOSurfaceRef surf = CVPixelBufferGetIOSurface(pb);
    if (!surf) return false;

    if (!g_kk) g_kk = kk_gpu_create(metal_device);  // geteiltes Device
    if (!g_kk) return false;
    kk_gpu *g = g_kk;

    // CVMetalTextureCache-Wrap (Pool-Buffer recyclen → Cache-Hit statt
    // newTextureWithDescriptor:iosurface: pro Frame); Fallback auf den rohen Wrap.
    kk_tex *luma   = kk_tex_wrap_pixbuf(g, pb, 0, KK_FMT_R8);
    kk_tex *chroma = kk_tex_wrap_pixbuf(g, pb, 1, KK_FMT_RG8);
    if (!luma)   luma   = kk_tex_wrap_iosurface(g, (void*)surf, 0, KK_FMT_R8,  KK_TEX_SAMPLE);
    if (!chroma) chroma = kk_tex_wrap_iosurface(g, (void*)surf, 1, KK_FMT_RG8, KK_TEX_SAMPLE);
    kk_tex *tgt    = kk_tex_wrap_mtltexture(g, target_texture);
    if (!luma || !chroma || !tgt) {
        kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt);
        return false;
    }
    int W = kk_tex_w(luma), H = kk_tex_h(luma);
    int OW = kk_tex_w(tgt), OH = kk_tex_h(tgt);
    if (W<=0||H<=0||OW<=0||OH<=0) { kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); return false; }

    // Intermediates nur bei Dim-Wechsel (neu)allozieren — sonst über Frames wiederverwenden.
    if (W != c_W || H != c_H || OW != c_OW || OH != c_OH) {
        kk_tex_destroy(g,&c_dec); kk_tex_destroy(g,&c_deb); kk_tex_destroy(g,&c_lin);
        kk_tex_destroy(g,&c_tmpx); kk_tex_destroy(g,&c_liny); kk_tex_destroy(g,&c_out);
        c_dec  = kk_tex_create(g, W,  H,  KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        c_deb  = kk_tex_create(g, W,  H,  KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        c_lin  = kk_tex_create(g, W,  H,  KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        c_tmpx = kk_tex_create(g, OW, H,  KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        c_liny = kk_tex_create(g, OW, OH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        c_out  = kk_tex_create(g, OW, OH, KK_FMT_BGRA8,   KK_TEX_STORAGE, NULL);
        kk_tex_destroy(g,&c_alin); kk_tex_destroy(g,&c_atmpx);
        c_alin  = kk_tex_create(g, W*2, H*2, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL); // linearisierter Anime4K-2×
        c_atmpx = kk_tex_create(g, OW,  H*2, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL); // X-skaliert (2H)
        kk_tex_destroy(g,&c_srgb); kk_tex_destroy(g,&c_a2rgb);
        c_srgb  = kk_tex_create(g, OW, OH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);   // CAS-Input
        c_a2rgb = kk_tex_create(g, W*2, H*2, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL); // ArtCNN-2×-RGB
        kk_tex_destroy(g,&c_adeb);
        c_adeb  = kk_tex_create(g, W*2, H*2, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL); // ArtCNN-Deband-2×
        kk_tex_destroy(g,&c_dbl);   // lazy im Deblock-Block
        c_W=W; c_H=H; c_OW=OW; c_OH=OH;
    }
    if (!c_dec || !c_deb || !c_lin || !c_tmpx || !c_liny || !c_out) {
        kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); return false;
    }

    // Speicher: nur der aktive Pfad resident. SDR-Pfad → HDR-Caches frei; nicht-aktiven
    // CNN-Cache frei (idempotent, nil-safe). Hält die RAM-Baseline niedrig (kein Akkumulieren
    // über Content-/Pfad-Wechsel → Jetsam-Schutz).
    kk_gpu_hdr_release(g);
    { const char *gl = getenv("KUCKUCK_GLSL_SHADER");
      if (!(gl && strcasestr(gl, "anime4k"))) kk_gpu_anime4k_release(g);
      if (!(gl && strcasestr(gl, "artcnn")))  kk_gpu_artcnn_release(g); }

    // Decode -> encodete RGB (echte YUV->RGB-Matrix vom Hook, Fallback BT.709 limited).
    struct { float m[12]; float co[2]; } D;
    kk_chroma_offset(pb, D.co);
    bool dith = kk_dither_an();
    if (yuv2rgb) { for (int i=0;i<12;i++) D.m[i]=yuv2rgb[i]; }
    else { float f[12]={1.1643f,0.0f,1.7927f, 1.1643f,-0.2132f,-0.5329f, 1.1643f,2.1124f,0.0f, -0.9729f,0.3015f,-1.1334f};
           for (int i=0;i<12;i++) D.m[i]=f[i]; }
    // HD-Light früh entscheiden (s. Default-Scaler unten): im HD-Light entfällt
    // Deband (wie renderpl.57) — der Pass würde sonst umsonst laufen. Für die
    // CNN-Pfade (SD-Cartoons/Realfilm) bleibt Deband unabhängig davon aktiv.
    const char *glsl_early = getenv("KUCKUCK_GLSL_SHADER");
    const char *hdl = getenv("KUCKUCK_HD_LIGHT");
    bool hdLight = hdl ? (hdl[0]=='1') : (H >= 1080);
    bool cnnPath = glsl_early && (strcasestr(glsl_early, "anime4k") || strcasestr(glsl_early, "artcnn"));
    bool deblock = glsl_early && strcasestr(glsl_early, "deblock");
    // DEC+LIN-Fusion (HD-Light ohne CNN/Deblock): dort ist LIN der EINZIGE c_dec-
    // Konsument (Deband ist aus, CNN-Pfade laufen nicht) -> der Standalone-DEC
    // entfällt, declin liest luma/chroma direkt (spart den c_dec-Roundtrip).
    bool fusedDec = hdLight && !cnnPath && !deblock;
    bool l3 = kk_chroma_l3_an(hdLight);
    if (!fusedDec) {
        kk_tex *chh = l3 ? kk_chroma_h(g, chroma, &c_chh, W, D.co[0]) : NULL;
        kk_compute_args da = { .out=c_dec, .in={luma, chh ? chh : chroma}, .n_in=2, .linear={false,true}, .uniforms=&D, .uniforms_size=sizeof D };
        kk_gpu_compute(g, chh ? DEC_L3_MSL : DEC_MSL, "dec", &da);
    }
    // Deblock (gated ~deblock): H -> c_dbl, V -> zurück nach c_dec. Danach sehen
    // alle Konsumenten (Deband, LIN, CNNs) die entblockte Quelle.
    if (deblock) {
        if (!c_dbl) c_dbl = kk_tex_create(g, W, H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        if (c_dbl) {
            struct { uint32_t axis; } ax = { 0 };
            kk_compute_args dh = { .out=c_dbl, .in={c_dec}, .n_in=1, .uniforms=&ax, .uniforms_size=sizeof ax };
            kk_gpu_compute(g, DEBLOCK_MSL, "deblock", &dh);
            ax.axis = 1;
            kk_compute_args dv = { .out=c_dec, .in={c_dbl}, .n_in=1, .uniforms=&ax, .uniforms_size=sizeof ax };
            kk_gpu_compute(g, DEBLOCK_MSL, "deblock", &dv);
        }
    }

    // Deband (KUCKUCK_DEBAND off|mild|strong) auf der encodeten Quelle.
    const char *dbenv = getenv("KUCKUCK_DEBAND");
    kk_tex *src_lin = c_dec;   // ohne Deband: linearize liest c_dec
    if (dbenv && (dbenv[0]=='m' || dbenv[0]=='s') && (!hdLight || cnnPath)) {
        struct { float radius, threshold, grain; uint32_t iters, index; } db;
        db.radius = 16.0f; db.index = (g_frame++);
        if (dbenv[0]=='s') { db.iters=2; db.threshold=4.0f/1000.0f; db.grain=1.0f/1000.0f; }  // strong
        else               { db.iters=1; db.threshold=3.0f/1000.0f; db.grain=1.0f/1000.0f; }  // mild
        kk_compute_args dba = { .out=c_deb, .in={c_dec}, .n_in=1, .linear={true}, .uniforms=&db, .uniforms_size=sizeof db };
        kk_gpu_compute(g, DEBAND_MSL, "deband", &dba);
        src_lin = c_deb;
    }

    // BT.1886 a/b + Primaries-Matrix (vom Hook; identity-Fallback bei 709/NULL).
    struct { float a, b; float m[9]; float o; } L = { 0.8704f, 0.0595f, {1,0,0, 0,1,0, 0,0,1}, 0.0f };
    if (prim2disp) for (int i=0;i<9;i++) L.m[i]=prim2disp[i];
    kk_lin_params(&L.a, &L.o);

    // Anime4K-Cartoon-Upscaler (gated via KUCKUCK_GLSL_SHADER~anime4k). Eigener Post-Scale
    // (2×-Output -> Ziel). Bei Fehler (Weights/Alloc) Fallback auf den Lanczos-Pfad unten.
    const char *glsl = getenv("KUCKUCK_GLSL_SHADER");
    if (glsl && strcasestr(glsl, "anime4k") && c_alin && c_atmpx) {
        const char *res = getenv("KUCKUCK_KK_CAPTURE_SHADERS");
        char wp[1200]; snprintf(wp, sizeof wp, "%s/anime4k_a_m.weights", res ? res : ".");
        kk_tex *a = kk_gpu_anime4k(g, src_lin, wp);   // 2W×2H encodete RGB
        if (a) {
            kk_compute_args alz = { .out=c_alin, .in={a}, .n_in=1, .uniforms=&L, .uniforms_size=sizeof L };
            kk_gpu_compute(g, LIN_MSL, "lin", &alz);                    // linearize (2W×2H)
            kk_lanczos_p apx = kk_lanczos_params((float)OW/(W*2), 0), apy = kk_lanczos_params((float)OH/(H*2), 1);
            kk_compute_args axa = { .out=c_atmpx, .in={c_alin}, .n_in=1, .uniforms=&apx, .uniforms_size=sizeof apx };
            kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &axa);            // X: 2W -> OW
            kk_compute_args aya = { .out=c_liny, .in={c_atmpx}, .n_in=1, .uniforms=&apy, .uniforms_size=sizeof apy };
            kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &aya);            // Y: 2H -> OH
            bool dw = kk_tex_can_write(tgt);   // Target ShaderWrite-fähig -> Blit sparen
            kk_compute_args ala = { .out=dw?tgt:c_out, .in={c_liny}, .n_in=1 };
            kk_gpu_compute(g, dith ? DELIN_D_MSL : DELIN_MSL, "delin", &ala);
            if (!dw) kk_gpu_blit(g, c_out, tgt);
            return kk_finish_or_submit(g, &luma, &chroma, &tgt, done, done_ud);
        }
    }

    // ArtCNN-Realfilm-SD-Luma-Upscaler (gated ~artcnn): luma -> 2×-Luma -> DEC(+chroma) ->
    // 2× encodete RGB -> geteilter 2×-Post (linearize -> Lanczos-to-Ziel -> sRGB).
    if (glsl && strcasestr(glsl, "artcnn") && c_a2rgb && c_alin && c_atmpx) {
        const char *res = getenv("KUCKUCK_KK_CAPTURE_SHADERS");
        char wp[1200]; snprintf(wp, sizeof wp, "%s/artcnn_c4f16.weights", res ? res : ".");
        kk_tex *luma2 = kk_gpu_artcnn(g, luma, wp);   // 2W×2H Luma (.r)
        if (luma2) {
            kk_tex *chh2 = l3 ? kk_chroma_h(g, chroma, &c_chh2, kk_tex_w(luma2), D.co[0]) : NULL;
            kk_compute_args ad = { .out=c_a2rgb, .in={luma2, chh2 ? chh2 : chroma}, .n_in=2, .linear={false,true}, .uniforms=&D, .uniforms_size=sizeof D };
            kk_gpu_compute(g, chh2 ? DEC_L3_MSL : DEC_MSL, "dec", &ad);   // 2×-Luma + chroma -> encodete RGB
            kk_tex *asrc = c_a2rgb;                    // Deband-mild (Realfilm) auf der 2×-Quelle
            if (dbenv && (dbenv[0]=='m' || dbenv[0]=='s') && c_adeb) {
                struct { float radius, threshold, grain; uint32_t iters, index; } db;
                db.radius=16.0f; db.index=g_frame;
                if (dbenv[0]=='s') { db.iters=2; db.threshold=4.0f/1000.0f; db.grain=1.0f/1000.0f; }
                else               { db.iters=1; db.threshold=3.0f/1000.0f; db.grain=1.0f/1000.0f; }
                kk_compute_args adb = { .out=c_adeb, .in={c_a2rgb}, .n_in=1, .linear={true}, .uniforms=&db, .uniforms_size=sizeof db };
                kk_gpu_compute(g, DEBAND_MSL, "deband", &adb);
                asrc = c_adeb;
            }
            kk_compute_args alz = { .out=c_alin, .in={asrc}, .n_in=1, .uniforms=&L, .uniforms_size=sizeof L };
            kk_gpu_compute(g, LIN_MSL, "lin", &alz);
            kk_lanczos_p apx = kk_lanczos_params((float)OW/(W*2), 0), apy = kk_lanczos_params((float)OH/(H*2), 1);
            kk_compute_args axa = { .out=c_atmpx, .in={c_alin}, .n_in=1, .uniforms=&apx, .uniforms_size=sizeof apx };
            kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &axa);
            kk_compute_args aya = { .out=c_liny, .in={c_atmpx}, .n_in=1, .uniforms=&apy, .uniforms_size=sizeof apy };
            kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &aya);
            bool dw = kk_tex_can_write(tgt);   // Target ShaderWrite-fähig -> Blit sparen
            kk_compute_args ala = { .out=dw?tgt:c_out, .in={c_liny}, .n_in=1 };
            kk_gpu_compute(g, dith ? DELIN_D_MSL : DELIN_MSL, "delin", &ala);
            if (!dw) kk_gpu_blit(g, c_out, tgt);
            return kk_finish_or_submit(g, &luma, &chroma, &tgt, done, done_ud);
        }
    }

    // HD-Light (Port von renderpl.57, war nie nach kk_gpu portiert — gemessen 22ms avg
    // @1080p, über dem 60Hz-Budget): ab src_h>=1080 separabler Lanczos (2×8 Taps) statt
    // EWA-polar (~49 Taps/Output-Pixel) + Sigmoid/Deband aus. Bei ~1,1-1,3× Upscale ist
    // EWA vs. separabel visuell belegt gleichwertig (IQ-Harness 06/2026); CAS bleibt.
    // Env KUCKUCK_HD_LIGHT: "0"=nie, "1"=immer, unset=auto (H>=1080).
    bool dw = kk_tex_can_write(tgt);     // Target ShaderWrite-fähig -> direkt rein, Blit sparen
    kk_tex *fin = dw ? tgt : c_out;
    bool cas = glsl && strcasestr(glsl, "cas") && c_srgb;
    kk_tex *dout = cas ? c_srgb : fin;

    if (hdLight && c_tmpx) {
        // HD: DECLIN (fusioniert) -> Lanczos X -> Lanczos Y -> Delin[+CAS fusioniert].
        if (fusedDec) {
            struct { float d[12]; float a, b; float m[9]; float o; float co[2]; } DL2;
            memcpy(DL2.d, D.m, sizeof DL2.d); DL2.a = L.a; DL2.b = L.b; memcpy(DL2.m, L.m, sizeof DL2.m);
            DL2.o = L.o; DL2.co[0] = D.co[0]; DL2.co[1] = D.co[1];
            kk_tex *chh = l3 ? kk_chroma_h(g, chroma, &c_chh, W, D.co[0]) : NULL;
            kk_compute_args la0 = { .out=c_lin, .in={luma, chh ? chh : chroma}, .n_in=2, .linear={false,true}, .uniforms=&DL2, .uniforms_size=sizeof DL2 };
            kk_gpu_compute(g, chh ? DECLIN_L3_MSL : DECLIN_MSL, "declin", &la0);
        } else {   // CNN-Gate an, aber CNN-Pfad oben gescheitert -> c_dec existiert
            kk_compute_args la0 = { .out=c_lin, .in={c_dec}, .n_in=1, .uniforms=&L, .uniforms_size=sizeof L };
            kk_gpu_compute(g, LIN_MSL, "lin", &la0);
        }
        kk_lanczos_p px = kk_lanczos_params((float)OW/W, 0), py = kk_lanczos_params((float)OH/H, 1);
        kk_compute_args xa = { .out=c_tmpx, .in={c_lin}, .n_in=1, .uniforms=&px, .uniforms_size=sizeof px };
        kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &xa);
        kk_compute_args ya = { .out=c_liny, .in={c_tmpx}, .n_in=1, .uniforms=&py, .uniforms_size=sizeof py };
        kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &ya);
        // Delin+CAS in EINEM Pass direkt -> fin (c_srgb-Roundtrip entfällt).
        //
        // ⚠️ KUCKUCK_NO_FUSION=1 fährt stattdessen DELIN -> CAS getrennt, mit
        // c_srgb als Zwischenstufe (der Zustand vor renderpl.70). Nur zum MESSEN:
        // die Fusion wurde 2026-07 am Mac verglichen (dort 9 % besser) und nie auf
        // einem Gerät. Die Pass-Zeitmessung vom 2026-09-02 legt nahe, dass der
        // Tausch „ALU gegen Bandbreite" dort anders ausgeht — delincas war mit
        // 2,567 ms der teuerste Pass, weil es je Pixel NEUN pow() rechnet (sRGB
        // pro CAS-Abtastpunkt) statt einem.
        const char *nofus = getenv("KUCKUCK_NO_FUSION");
        if (cas && nofus && nofus[0] == '1') {
            kk_compute_args ld = { .out=c_srgb, .in={c_liny}, .n_in=1 };
            kk_gpu_compute(g, DELIN_MSL, "delin", &ld);
            kk_compute_args lc = { .out=fin, .in={c_srgb}, .n_in=1 };
            kk_gpu_compute(g, dith ? CAS_D_MSL : CAS_MSL, "cas", &lc);
        } else {
            kk_compute_args la = { .out=fin, .in={c_liny}, .n_in=1 };
            const char *k = cas ? (dith ? DELINCAS_D_MSL : DELINCAS_MSL) : (dith ? DELIN_D_MSL : DELIN_MSL);
            kk_gpu_compute(g, k, cas ? "delincas" : "delin", &la);
        }
        if (!dw) kk_gpu_blit(g, c_out, tgt);
        return kk_finish_or_submit(g, &luma, &chroma, &tgt, done, done_ud);
    } else {
        // SD-Qualitätspfad: [Deband] -> LIN[+SIG fusioniert] -> EWA-lanczossharp ->
        // [Unsig+]Delin (= libplacebos pl_render_high_quality-Default).
        bool up = (OW > W);   // Sigmoid nur bei Upscale (wie libplacebo)
        kk_compute_args la0 = { .out=c_lin, .in={src_lin}, .n_in=1, .uniforms=&L, .uniforms_size=sizeof L };
        kk_gpu_compute(g, up ? LINSIG_MSL : LIN_MSL, up ? "linsig" : "lin", &la0);
        struct { float scale; uint32_t lutn; float radius; float lut[64]; } ew;
        ew.scale = (float)OW/W; ew.lutn = 64; ew.radius = KK_EWA_RADIUS;
        for (int i=0;i<64;i++) ew.lut[i]=KK_EWA_LUT[i];
        kk_compute_args ea = { .out=c_liny, .in={c_lin}, .n_in=1, .uniforms=&ew, .uniforms_size=sizeof ew };
        kk_gpu_compute(g, EWA_MSL, "ewa", &ea);
        kk_compute_args la = { .out=dout, .in={c_liny}, .n_in=1 };
        bool d8 = dith && !cas;   // mit CAS geht es erst in die Float-Stufe c_srgb
        kk_gpu_compute(g, up ? (d8 ? DELINU_D_MSL : DELINU_MSL) : (d8 ? DELIN_D_MSL : DELIN_MSL),
                       up ? "delinu" : "delin", &la);
    }
    // CAS-Sharpen (HD/Tuner, gated ~cas): encodete Ausgabe -> CAS -> Target.
    if (cas) {
        kk_compute_args ca = { .out=fin, .in={c_srgb}, .n_in=1 };
        kk_gpu_compute(g, dith ? CAS_D_MSL : CAS_MSL, "cas", &ca);
    }
    if (!dw) kk_gpu_blit(g, c_out, tgt); // nur falls Target nicht direkt beschreibbar
    return kk_finish_or_submit(g, &luma, &chroma, &tgt, done, done_ud);
}

// HDR-Render (P010 -> PQ/2020): MKPQ(2020-10bit-Decode+PQ-EOTF) -> EWA -> CMHDR
// (IPT-Tonemap zum EDR-Peak + Chroma-Hull -> PQ-Output) -> Blit. hp vom Hook (libplacebo).
static kk_tex *h_pq = NULL, *h_ewa = NULL, *h_out = NULL, *h_tmpx = NULL;
static int h_W = 0, h_H = 0, h_OW = 0, h_OH = 0;

bool kk_gpu_render_hdr(void *metal_device, void *cv_pixbuf, void *target_texture,
                       const kk_hdr_params *hp,
                       void (*done)(void*), void *done_ud) {
    CVPixelBufferRef pb = (CVPixelBufferRef) cv_pixbuf;
    if (!pb || !target_texture || !hp) return false;
    OSType pf = CVPixelBufferGetPixelFormatType(pb);
    if (pf != kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange &&
        pf != kCVPixelFormatType_420YpCbCr10BiPlanarFullRange)
        return false;
    IOSurfaceRef surf = CVPixelBufferGetIOSurface(pb);
    if (!surf) return false;
    if (!g_kk) g_kk = kk_gpu_create(metal_device);
    if (!g_kk) return false;
    kk_gpu *g = g_kk;

    // CVMetalTextureCache-Wrap wie im SDR-Pfad; Fallback auf den rohen Wrap.
    kk_tex *luma   = kk_tex_wrap_pixbuf(g, pb, 0, KK_FMT_R16);
    kk_tex *chroma = kk_tex_wrap_pixbuf(g, pb, 1, KK_FMT_RG16);
    if (!luma)   luma   = kk_tex_wrap_iosurface(g, (void*)surf, 0, KK_FMT_R16,  KK_TEX_SAMPLE);
    if (!chroma) chroma = kk_tex_wrap_iosurface(g, (void*)surf, 1, KK_FMT_RG16, KK_TEX_SAMPLE);
    kk_tex *tgt    = kk_tex_wrap_mtltexture(g, target_texture);
    if (!luma || !chroma || !tgt) {
        kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); return false;
    }
    int W = kk_tex_w(luma), H = kk_tex_h(luma), OW = kk_tex_w(tgt), OH = kk_tex_h(tgt);
    if (W<=0||H<=0||OW<=0||OH<=0) { kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); return false; }

    if (W != h_W || H != h_H || OW != h_OW || OH != h_OH) {
        kk_tex_destroy(g,&h_pq); kk_tex_destroy(g,&h_ewa); kk_tex_destroy(g,&h_out); kk_tex_destroy(g,&h_tmpx);
        h_pq  = kk_tex_create(g, W,  H,  KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        h_ewa = kk_tex_create(g, OW, OH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL);
        h_out = kk_tex_create(g, OW, OH, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL); // PQ-Float
        h_tmpx = kk_tex_create(g, OW, H, KK_FMT_RGBA16F, KK_TEX_SAMPLE|KK_TEX_STORAGE, NULL); // HD-Light Lanczos-X
        h_W=W; h_H=H; h_OW=OW; h_OH=OH;
    }
    if (!h_pq || !h_ewa || !h_out) {
        kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); return false;
    }
    // Speicher: HDR-Pfad nutzt nur h_* → SDR- + CNN-Caches freigeben (idempotent).
    kk_gpu_sdr_release(g); kk_gpu_anime4k_release(g); kk_gpu_artcnn_release(g);

    struct { float co[2]; } K; kk_chroma_offset(pb, K.co);
    kk_compute_args mk = { .out=h_pq, .in={luma,chroma}, .n_in=2, .linear={false,true}, .uniforms=&K, .uniforms_size=sizeof K };
    kk_gpu_compute(g, MKPQ_MSL, "mk", &mk);                           // P010 -> linear 2020 (10000-norm)
    // HD-Light auch für HDR (renderpl.69): 1440p-HDR ist am iPhone ein DOWNSCALE
    // (~0,84x) -> die EWA-Box wächst auf ~9x9=81 Taps = gemessen ~33ms avg (2x über
    // dem 60Hz-Budget, jeder 2. Frame gedroppt). Separabler band-limitierter Lanczos
    // (2x ~8 Taps) wie im SDR-HD-Pfad; gleiche Env-Gate-Semantik.
    const char *hdl = getenv("KUCKUCK_HD_LIGHT");
    bool hdLight = hdl ? (hdl[0]=='1') : (H >= 1080);
    if (hdLight && h_tmpx) {
        kk_lanczos_p px = kk_lanczos_params((float)OW/W, 0), py = kk_lanczos_params((float)OH/H, 1);
        kk_compute_args xa = { .out=h_tmpx, .in={h_pq}, .n_in=1, .uniforms=&px, .uniforms_size=sizeof px };
        kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &xa);
        kk_compute_args ya = { .out=h_ewa, .in={h_tmpx}, .n_in=1, .uniforms=&py, .uniforms_size=sizeof py };
        kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &ya);
    } else {
        struct { float scale; uint32_t lutn; float radius; float lut[64]; } ew;
        ew.scale=(float)OW/W; ew.lutn=64; ew.radius=KK_EWA_RADIUS; for(int i=0;i<64;i++) ew.lut[i]=KK_EWA_LUT[i];
        kk_compute_args ea = { .out=h_ewa, .in={h_pq}, .n_in=1, .uniforms=&ew, .uniforms_size=sizeof ew };
        kk_gpu_compute(g, EWA_MSL, "ewa", &ea);                       // EWA-Scale in Linear
    }
    bool dw = kk_tex_can_write(tgt);     // Target ShaderWrite-fähig -> direkt rein, Blit sparen
    kk_compute_args ca = { .out=dw?tgt:h_out, .in={h_ewa}, .n_in=1, .uniforms=hp, .uniforms_size=sizeof(kk_hdr_params) };
    kk_gpu_compute(g, CMHDR_MSL, "cmh", &ca);                         // IPT-Tonemap -> PQ/2020
    if (!dw) kk_gpu_blit(g, h_out, tgt);
    return kk_finish_or_submit(g, &luma, &chroma, &tgt, done, done_ud);
}

// --- Cache-Freigabe (Speicher / Jetsam-Schutz) ---
// SDR-Pfad-Caches (common + pfad-spezifisch). Reset c_W → lazy Re-Alloc bei nächstem SDR-Frame.
static void kk_gpu_sdr_release(kk_gpu *g) {
    kk_tex_destroy(g,&c_dec); kk_tex_destroy(g,&c_deb); kk_tex_destroy(g,&c_lin);
    kk_tex_destroy(g,&c_tmpx); kk_tex_destroy(g,&c_liny); kk_tex_destroy(g,&c_out);
    kk_tex_destroy(g,&c_alin); kk_tex_destroy(g,&c_atmpx); kk_tex_destroy(g,&c_srgb);
    kk_tex_destroy(g,&c_a2rgb); kk_tex_destroy(g,&c_dbl);
    kk_tex_destroy(g,&c_chh); kk_tex_destroy(g,&c_chh2);
    c_W = c_H = c_OW = c_OH = 0;
}
// HDR-Pfad-Caches. Reset h_W → lazy Re-Alloc bei nächstem HDR-Frame.
static void kk_gpu_hdr_release(kk_gpu *g) {
    kk_tex_destroy(g,&h_pq); kk_tex_destroy(g,&h_ewa); kk_tex_destroy(g,&h_out); kk_tex_destroy(g,&h_tmpx);
    h_W = h_H = h_OW = h_OH = 0;
}
// Alle kk_gpu-Caches freigeben (Teardown / Player-Close): SDR + HDR + beide CNN. g_kk bleibt.
void kk_gpu_release_all(void) {
    if (!g_kk) return;
    kk_gpu_sdr_release(g_kk); kk_gpu_hdr_release(g_kk);
    kk_gpu_anime4k_release(g_kk); kk_gpu_artcnn_release(g_kk);
}

// PSO-Prewarm (gegen Erst-Frame-Hitch): alle statischen Kernel einmal kompilieren
// (nur Compile+Cache, kein Dispatch). Vom App-Attach auf der Render-Queue gerufen —
// dieselbe Queue wie der erste Render → kein Race auf den PSO-Cache. CNN-Shader
// kompilieren weiterhin lazy in ihrem gated Pfad (groß + selten).
void kk_gpu_prewarm(void *metal_device) {
    if (!g_kk) g_kk = kk_gpu_create(metal_device);
    if (!g_kk) return;
    kk_gpu *g = g_kk;
    kk_gpu_compile(g, DEC_MSL,    "dec");
    kk_gpu_compile(g, DECLIN_MSL, "declin");
    kk_gpu_compile(g, DECLIN_L3_MSL, "declin");
    kk_gpu_compile(g, DEC_L3_MSL, "dec");
    kk_gpu_compile(g, CHH_MSL, "chh");
    kk_gpu_compile(g, DELINCAS_MSL, "delincas");
    kk_gpu_compile(g, DELINCAS_D_MSL, "delincas");
    kk_gpu_compile(g, DEBAND_MSL, "deband");
    kk_gpu_compile(g, DEBLOCK_MSL, "deblock");
    kk_gpu_compile(g, LIN_MSL,    "lin");
    kk_gpu_compile(g, LINSIG_MSL, "linsig");
    kk_gpu_compile(g, LANCZOS_MSL,"lanczos");
    kk_gpu_compile(g, DELIN_MSL,  "delin");
    kk_gpu_compile(g, DELINU_MSL, "delinu");
    kk_gpu_compile(g, CAS_MSL,    "cas");
    kk_gpu_compile(g, DELIN_D_MSL,  "delin");
    kk_gpu_compile(g, DELINU_D_MSL, "delinu");
    kk_gpu_compile(g, CAS_D_MSL,    "cas");
    kk_gpu_compile(g, EWA_MSL,    "ewa");
    kk_gpu_compile(g, MKPQ_MSL,   "mk");
    kk_gpu_compile(g, CMHDR_MSL,  "cmh");
}


// ---------------------------------------------------------------------------
// Deblock VOR VT-SR: NV12 (Decoder-Buffer) -> NV12 (IOSurface-Pool-Buffer der App),
// Luma über den separablen Bilateral, Chroma per Blit. Synchron (finish) — läuft
// auf der srQueue der App unmittelbar vor dem ANE-Aufruf.
// ⚠️ EIGENER kk_gpu-Kontext: kk_gpu bündelt alle Dispatches eines Kontexts in
// EINEN Command-Buffer (g->cb/g->enc). Der Render-Kontext g_kk wird gleichzeitig
// von der renderQueue befüllt — ein geteilter Kontext wäre ein Encoder-Race.
// Zwei Kontexte auf demselben MTLDevice/-Queue sind dagegen erlaubt.
static kk_gpu *g_dbl = NULL;
static kk_tex *c_dblY = NULL;
static int c_dblW = 0, c_dblH = 0;
bool kk_gpu_deblock_nv12(void *metal_device, void *src_pb, void *dst_pb) {
    CVPixelBufferRef s = (CVPixelBufferRef) src_pb, d = (CVPixelBufferRef) dst_pb;
    if (!s || !d) return false;
    if (!g_dbl) g_dbl = kk_gpu_create(metal_device);
    if (!g_dbl) return false;
    kk_gpu *g = g_dbl;
    int W = (int) CVPixelBufferGetWidthOfPlane(s, 0), H = (int) CVPixelBufferGetHeightOfPlane(s, 0);
    if (W <= 0 || H <= 0 || W != (int) CVPixelBufferGetWidthOfPlane(d, 0) || H != (int) CVPixelBufferGetHeightOfPlane(d, 0)) return false;
    IOSurfaceRef ssurf = CVPixelBufferGetIOSurface(s), dsurf = CVPixelBufferGetIOSurface(d);
    if (!dsurf) return false;
    kk_tex *sy = kk_tex_wrap_pixbuf(g, s, 0, KK_FMT_R8);
    kk_tex *sc = kk_tex_wrap_pixbuf(g, s, 1, KK_FMT_RG8);
    if (!sy && ssurf) sy = kk_tex_wrap_iosurface(g, (void*) ssurf, 0, KK_FMT_R8,  KK_TEX_SAMPLE);
    if (!sc && ssurf) sc = kk_tex_wrap_iosurface(g, (void*) ssurf, 1, KK_FMT_RG8, KK_TEX_SAMPLE);
    kk_tex *dy = kk_tex_wrap_iosurface(g, (void*) dsurf, 0, KK_FMT_R8,  KK_TEX_STORAGE | KK_TEX_SAMPLE);
    kk_tex *dc = kk_tex_wrap_iosurface(g, (void*) dsurf, 1, KK_FMT_RG8, KK_TEX_STORAGE | KK_TEX_SAMPLE);
    bool ok = false;
    if (sy && sc && dy && dc) {
        if (!c_dblY || c_dblW != W || c_dblH != H) {
            if (c_dblY) kk_tex_destroy(g, &c_dblY);
            c_dblY = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE | KK_TEX_STORAGE, NULL);
            c_dblW = W; c_dblH = H;
        }
        if (c_dblY) {
            struct { uint32_t axis; } ax = { 0 };
            kk_compute_args dh = { .out=c_dblY, .in={sy}, .n_in=1, .uniforms=&ax, .uniforms_size=sizeof ax };
            ok = kk_gpu_compute(g, DEBLOCK_Y_MSL, "deblock_y", &dh);
            ax.axis = 1;
            kk_compute_args dv = { .out=dy, .in={c_dblY}, .n_in=1, .uniforms=&ax, .uniforms_size=sizeof ax };
            ok = ok && kk_gpu_compute(g, DEBLOCK_Y_MSL, "deblock_y", &dv);
            if (ok) kk_gpu_blit(g, sc, dc);
            kk_gpu_finish(g);
        }
    }
    kk_tex_destroy(g, &sy); kk_tex_destroy(g, &sc); kk_tex_destroy(g, &dy); kk_tex_destroy(g, &dc);
    return ok;
}

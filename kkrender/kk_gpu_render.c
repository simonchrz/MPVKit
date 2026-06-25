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
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// DEC: YUV -> encodete RGB (exakte Matrix via pl_color_repr_decode, 9+3). KEIN linearize
// (Deband sitzt auf der encodeten Quelle, wie libplacebos source-deband).
static const char *DEC_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct D{float m[12];};\n"
"kernel void dec(texture2d<float> luma [[texture(0)]], texture2d<float> chroma [[texture(1)]],\n"
"  texture2d<float,access::write> dst [[texture(2)]], sampler near [[sampler(0)]], sampler lin [[sampler(1)]],\n"
"  constant D& d [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  float2 uv=(float2(id)+0.5)/float2(w,h); float Y=luma.read(id).r; float2 C=chroma.sample(lin,uv).rg;\n"
"  float3 v=float3(Y,C.r,C.g);\n"
"  float3 rgb=float3(d.m[0]*v.x+d.m[1]*v.y+d.m[2]*v.z, d.m[3]*v.x+d.m[4]*v.y+d.m[5]*v.z, d.m[6]*v.x+d.m[7]*v.y+d.m[8]*v.z)+float3(d.m[9],d.m[10],d.m[11]);\n"
"  dst.write(float4(rgb,1.0),id);}\n";
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
// LIN: BT.1886 a*pow(c+b,2.4) (encoded -> linear).
static const char *LIN_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct L{float a,b;};\n"
"kernel void lin(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant L& l [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  float3 c=max(src.read(id).rgb,0.0); dst.write(float4(l.a*pow(c+l.b,float3(2.4)),1.0),id);}\n";
static const char *LANCZOS_MSL =
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{float scale;uint axis;};\n"
"static inline float sinc(float x){if(x==0.0)return 1.0;x*=M_PI_F;return sin(x)/x;}\n"
"static inline float l3(float x){x=abs(x);if(x>=3.0)return 0.0;return sinc(x)*sinc(x/3.0);}\n"
"kernel void lanczos(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint W=dst.get_width(),H=dst.get_height(); if(id.x>=W||id.y>=H)return;\n"
"  int sw=int(src.get_width()),sh=int(src.get_height()); float coord=(p.axis==0u?float(id.x):float(id.y));\n"
"  float s=(coord+0.5)/p.scale-0.5; int base=int(floor(s)); float4 acc=float4(0.0); float wsum=0.0;\n"
"  for(int t=-3;t<=4;t++){ int tap=base+t; float w=l3(s-float(tap));\n"
"    int cx=(p.axis==0u)?clamp(tap,0,sw-1):int(id.x); int cy=(p.axis==1u)?clamp(tap,0,sh-1):int(id.y);\n"
"    acc+=w*src.read(uint2(cx,cy)); wsum+=w; } dst.write(acc/wsum,id);}\n";
static const char *DELIN_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"static inline float srgb(float c){ return c<=0.0031308 ? 12.92*c : 1.055*pow(c,1.0/2.4)-0.055; }\n"
"kernel void delin(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  float3 c=clamp(src.read(id).rgb,0.0,1.0); dst.write(float4(srgb(c.r),srgb(c.g),srgb(c.b),1.0),id);}\n";

static kk_gpu *g_kk = NULL;  // lazy, teilt libplacebos Device
// Gecachte Intermediates (über Frames wiederverwendet — KEIN Per-Frame-Alloc, sonst
// Jetsam-OOM durch Allok-Churn in Ziel-Auflösung). Re-create nur bei Dim-Wechsel.
static kk_tex *c_dec = NULL, *c_deb = NULL, *c_lin = NULL, *c_tmpx = NULL, *c_liny = NULL, *c_out = NULL;
static int c_W = 0, c_H = 0, c_OW = 0, c_OH = 0;
static unsigned g_frame = 0;   // temporaler Grain-Index (Deband)

// yuv2rgb = 12 floats (9 Matrix row-major + 3 Offset) aus libplacebos pl_color_repr_decode
// (vom Hook übergeben — echte 601/709/2020-Matrix + Range). true = von kk_gpu gerendert.
bool kk_gpu_render(void *metal_device, void *cv_pixbuf, void *target_texture,
                   const float *yuv2rgb) {
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

    kk_tex *luma   = kk_tex_wrap_iosurface(g, (void*)surf, 0, KK_FMT_R8,  KK_TEX_SAMPLE);
    kk_tex *chroma = kk_tex_wrap_iosurface(g, (void*)surf, 1, KK_FMT_RG8, KK_TEX_SAMPLE);
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
        c_W=W; c_H=H; c_OW=OW; c_OH=OH;
    }
    if (!c_dec || !c_deb || !c_lin || !c_tmpx || !c_liny || !c_out) {
        kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); return false;
    }

    // Decode -> encodete RGB (echte YUV->RGB-Matrix vom Hook, Fallback BT.709 limited).
    struct { float m[12]; } D;
    if (yuv2rgb) { for (int i=0;i<12;i++) D.m[i]=yuv2rgb[i]; }
    else { float f[12]={1.1643f,0.0f,1.7927f, 1.1643f,-0.2132f,-0.5329f, 1.1643f,2.1124f,0.0f, -0.9729f,0.3015f,-1.1334f};
           for (int i=0;i<12;i++) D.m[i]=f[i]; }
    kk_compute_args da = { .out=c_dec, .in={luma,chroma}, .n_in=2, .linear={false,true}, .uniforms=&D, .uniforms_size=sizeof D };
    kk_gpu_compute(g, DEC_MSL, "dec", &da);

    // Deband (KUCKUCK_DEBAND off|mild|strong) auf der encodeten Quelle.
    const char *dbenv = getenv("KUCKUCK_DEBAND");
    kk_tex *src_lin = c_dec;   // ohne Deband: linearize liest c_dec
    if (dbenv && (dbenv[0]=='m' || dbenv[0]=='s')) {
        struct { float radius, threshold, grain; uint32_t iters, index; } db;
        db.radius = 16.0f; db.index = (g_frame++);
        if (dbenv[0]=='s') { db.iters=2; db.threshold=4.0f/1000.0f; db.grain=1.0f/1000.0f; }  // strong
        else               { db.iters=1; db.threshold=3.0f/1000.0f; db.grain=1.0f/1000.0f; }  // mild
        kk_compute_args dba = { .out=c_deb, .in={c_dec}, .n_in=1, .linear={true}, .uniforms=&db, .uniforms_size=sizeof db };
        kk_gpu_compute(g, DEBAND_MSL, "deband", &dba);
        src_lin = c_deb;
    }

    // Linearize -> Lanczos X/Y -> delinearize(sRGB) -> Blit ins Target.
    struct { float a, b; } L = { 0.8704f, 0.0595f };  // BT.1886, csp_min=0.001
    kk_compute_args la0 = { .out=c_lin, .in={src_lin}, .n_in=1, .uniforms=&L, .uniforms_size=sizeof L };
    kk_gpu_compute(g, LIN_MSL, "lin", &la0);
    struct { float scale; uint32_t axis; } px = { (float)OW/W, 0 }, py = { (float)OH/H, 1 };
    kk_compute_args xa = { .out=c_tmpx, .in={c_lin},  .n_in=1, .uniforms=&px, .uniforms_size=sizeof px };
    kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &xa);
    kk_compute_args ya = { .out=c_liny, .in={c_tmpx}, .n_in=1, .uniforms=&py, .uniforms_size=sizeof py };
    kk_gpu_compute(g, LANCZOS_MSL, "lanczos", &ya);
    kk_compute_args la = { .out=c_out, .in={c_liny}, .n_in=1 };
    kk_gpu_compute(g, DELIN_MSL, "delin", &la);
    kk_gpu_blit(g, c_out, tgt);          // eigener Output -> Display-Target
    kk_gpu_finish(g);

    kk_tex_destroy(g,&luma); kk_tex_destroy(g,&chroma); kk_tex_destroy(g,&tgt); // nur die billigen Wraps
    return true;
}

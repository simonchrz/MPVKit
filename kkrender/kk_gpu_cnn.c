// kk_gpu_cnn.c — Anime4K-Mode-A(M) als kk_gpu-Funktion (content-gated Cartoon-Upscaler).
// Adaptiert aus dem headless verifizierten kk_anime4k_full (maxerr 1 vs CPU): De-Ring
// (5×5-Max-Pool+Clamp) -> Restore-CNN (conv_rgb + 6×CReLU + 1×1-Combine+Residual) ->
// Upscale-CNN (conv_rgb + 6×CReLU + 1×1-Combine) -> D2S(2× + bilinear-Residual).
// Input = encodete RGB (RGBA16F), Output = 2×-encodete-RGB. Weights aus Bundle-Datei.
// ⚠️ ON-DEVICE-PERF: 20 Pässe/Frame (Direkt-Reads); bei Jank -> Threadgroup-Tiling.
#include "kk_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const char *STAT=
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{uint vert;};\n"
"static inline float lum(float4 c){return dot(float4(0.299,0.587,0.114,0.0),c);}\n"
"kernel void stat(texture2d<float> s [[texture(0)]], texture2d<float,access::write> d [[texture(1)]], constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){uint w=d.get_width(),h=d.get_height();if(id.x>=w||id.y>=h)return;int iw=int(w),ih=int(h);float m=0;\n"
" for(int i=0;i<5;i++){int2 o=p.vert?int2(0,i-2):int2(i-2,0);int2 q=clamp(int2(id)+o,int2(0),int2(iw-1,ih-1));float g=p.vert?s.read(uint2(q)).x:lum(s.read(uint2(q)));m=max(g,m);} d.write(float4(m,0,0,0),id);}\n";
static const char *CLAMPK=
"#include <metal_stdlib>\nusing namespace metal;\nstatic inline float lum(float4 c){return dot(float4(0.299,0.587,0.114,0.0),c);}\n"
"kernel void clmp(texture2d<float> rgb [[texture(0)]], texture2d<float> st [[texture(1)]], texture2d<float,access::write> d [[texture(2)]], uint2 id [[thread_position_in_grid]]){uint w=d.get_width(),h=d.get_height();if(id.x>=w||id.y>=h)return;float4 c=rgb.read(id);float cl=lum(c);float nl=min(cl,st.read(id).x);d.write(c-(cl-nl),id);}\n";
static const char *CONV_RGB=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float m[144];float b[4];};\n"
"kernel void crgb(texture2d<float> in0 [[texture(0)]], texture2d<float,access::write> d [[texture(1)]], constant W& wb [[buffer(0)]], uint2 id [[thread_position_in_grid]]){uint w=d.get_width(),h=d.get_height();if(id.x>=w||id.y>=h)return;int iw=int(w),ih=int(h);float4 r=float4(wb.b[0],wb.b[1],wb.b[2],wb.b[3]);\n"
" for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int2 p=clamp(int2(id)+int2(X-1,Y-1),int2(0),int2(iw-1,ih-1));float4 v=in0.read(uint2(p));int sp=Y*3+X;int b=sp*16;\n"
"  float4x4 M=float4x4(wb.m[b],wb.m[b+1],wb.m[b+2],wb.m[b+3],wb.m[b+4],wb.m[b+5],wb.m[b+6],wb.m[b+7],wb.m[b+8],wb.m[b+9],wb.m[b+10],wb.m[b+11],wb.m[b+12],wb.m[b+13],wb.m[b+14],wb.m[b+15]); r+=M*v;} d.write(r,id);}\n";
static const char *CONV_CRELU=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float m[288];float b[4];};\n"
"kernel void ccr(texture2d<float> in0 [[texture(0)]], texture2d<float,access::write> d [[texture(1)]], constant W& wb [[buffer(0)]], uint2 id [[thread_position_in_grid]]){uint w=d.get_width(),h=d.get_height();if(id.x>=w||id.y>=h)return;int iw=int(w),ih=int(h);float4 r=float4(wb.b[0],wb.b[1],wb.b[2],wb.b[3]);\n"
" for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int2 p=clamp(int2(id)+int2(X-1,Y-1),int2(0),int2(iw-1,ih-1));float4 v=in0.read(uint2(p));float4 g0=max(v,0.0),g1=max(-v,0.0);int sp=Y*3+X;\n"
"  for(int a=0;a<2;a++){int b=(a*9+sp)*16;float4x4 M=float4x4(wb.m[b],wb.m[b+1],wb.m[b+2],wb.m[b+3],wb.m[b+4],wb.m[b+5],wb.m[b+6],wb.m[b+7],wb.m[b+8],wb.m[b+9],wb.m[b+10],wb.m[b+11],wb.m[b+12],wb.m[b+13],wb.m[b+14],wb.m[b+15]); r+=M*(a==0?g0:g1);}} d.write(r,id);}\n";
static const char *COMBINE=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float m[224];float b[4];uint resid;};\n"
"kernel void comb(texture2d<float> c0 [[texture(0)]],texture2d<float> c1 [[texture(1)]],texture2d<float> c2 [[texture(2)]],texture2d<float> c3 [[texture(3)]],texture2d<float> c4 [[texture(4)]],texture2d<float> c5 [[texture(5)]],texture2d<float> c6 [[texture(6)]],texture2d<float> mn [[texture(7)]], texture2d<float,access::write> d [[texture(8)]], constant W& wb [[buffer(0)]], uint2 id [[thread_position_in_grid]]){uint w=d.get_width(),h=d.get_height();if(id.x>=w||id.y>=h)return;\n"
" float4 cv[7]={c0.read(id),c1.read(id),c2.read(id),c3.read(id),c4.read(id),c5.read(id),c6.read(id)};\n"
" float4 r=float4(wb.b[0],wb.b[1],wb.b[2],wb.b[3]);\n"
" for(int i=0;i<7;i++){ float4 g0=max(cv[i],0.0),g1=max(-cv[i],0.0);\n"
"   for(int a=0;a<2;a++){int gi=i*2+a;int b=gi*16;float4x4 M=float4x4(wb.m[b],wb.m[b+1],wb.m[b+2],wb.m[b+3],wb.m[b+4],wb.m[b+5],wb.m[b+6],wb.m[b+7],wb.m[b+8],wb.m[b+9],wb.m[b+10],wb.m[b+11],wb.m[b+12],wb.m[b+13],wb.m[b+14],wb.m[b+15]); r+=M*(a==0?g0:g1);}}\n"
" if(wb.resid!=0u) r+=mn.read(id); d.write(r,id);}\n";
static const char *D2S=
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void d2s(texture2d<float> cl [[texture(0)]], texture2d<float> mn [[texture(1)]], texture2d<float,access::write> d [[texture(2)]], sampler near [[sampler(0)]], sampler lin [[sampler(1)]], uint2 id [[thread_position_in_grid]]){uint w=d.get_width(),h=d.get_height();if(id.x>=w||id.y>=h)return;\n"
" uint2 t=id/2; uint comp=(id.y%2)*2+(id.x%2); float det=cl.read(t)[comp];\n"
" float2 uv=(float2(id)+0.5)/float2(w,h); float4 m=mn.sample(lin,uv); d.write(m+det,id);}\n";

static float *g_w = NULL;   // 4256 floats, lazy aus Bundle geladen
static kk_tex *s1=NULL,*s2=NULL,*dering=NULL,*cv[7]={0},*restored=NULL,*cv2[7]={0},*lastT=NULL,*a4kOut=NULL;
static int cW=0,cH=0;

// Anime4K auf encodeter RGB (in, W×H) -> 2×-RGB (W*2×H*2). NULL bei Fehler.
kk_tex *kk_gpu_anime4k(kk_gpu *g, kk_tex *in, const char *weights_path) {
    if (!g_w) {
        FILE *f = fopen(weights_path, "rb"); if (!f) return NULL;
        g_w = malloc(4256*4); if (fread(g_w,4,4256,f)!=4256){fclose(f);free(g_w);g_w=NULL;return NULL;} fclose(f);
    }
    int W = kk_tex_w(in), H = kk_tex_h(in), OW = W*2, OH = H*2;
    if (W != cW || H != cH) {
        kk_tex_destroy(g,&s1);kk_tex_destroy(g,&s2);kk_tex_destroy(g,&dering);kk_tex_destroy(g,&restored);kk_tex_destroy(g,&lastT);kk_tex_destroy(g,&a4kOut);
        for(int i=0;i<7;i++){kk_tex_destroy(g,&cv[i]);kk_tex_destroy(g,&cv2[i]);}
        s1=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        s2=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        dering=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        restored=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        lastT=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        a4kOut=kk_tex_create(g,OW,OH,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        for(int i=0;i<7;i++){cv[i]=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
                             cv2[i]=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);}
        cW=W; cH=H;
    }
    if (!a4kOut) return NULL;
    // Offsets (floats): Restore rgb@0(148), crelu@148+k*292, combine@1900(228);
    //                   Upscale rgb@2128(148), crelu@2276+k*292, combine@4028(228).
    int Rrgb=0,Rcr[6],Rcomb=1900,Urgb=2128,Ucr[6],Ucomb=4028;
    for(int k=0;k<6;k++){Rcr[k]=148+k*292;Ucr[k]=2276+k*292;}
    unsigned vH=0,vV=1;
    // De-Ring
    kk_gpu_compute(g,STAT,"stat",&(kk_compute_args){.out=s1,.in={in},.n_in=1,.uniforms=&vH,.uniforms_size=4});
    kk_gpu_compute(g,STAT,"stat",&(kk_compute_args){.out=s2,.in={s1},.n_in=1,.uniforms=&vV,.uniforms_size=4});
    kk_gpu_compute(g,CLAMPK,"clmp",&(kk_compute_args){.out=dering,.in={in,s2},.n_in=2});
    // Restore-CNN
    kk_gpu_compute(g,CONV_RGB,"crgb",&(kk_compute_args){.out=cv[0],.in={dering},.n_in=1,.uniforms=g_w+Rrgb,.uniforms_size=148*4});
    for(int k=0;k<6;k++) kk_gpu_compute(g,CONV_CRELU,"ccr",&(kk_compute_args){.out=cv[k+1],.in={cv[k]},.n_in=1,.uniforms=g_w+Rcr[k],.uniforms_size=292*4});
    struct {float m[224];float b[4];unsigned resid;} cb; memcpy(cb.m,g_w+Rcomb,224*4); memcpy(cb.b,g_w+Rcomb+224,16); cb.resid=1;
    kk_gpu_compute(g,COMBINE,"comb",&(kk_compute_args){.out=restored,.in={cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],dering},.n_in=8,.uniforms=&cb,.uniforms_size=sizeof cb});
    // Upscale-CNN
    kk_gpu_compute(g,CONV_RGB,"crgb",&(kk_compute_args){.out=cv2[0],.in={restored},.n_in=1,.uniforms=g_w+Urgb,.uniforms_size=148*4});
    for(int k=0;k<6;k++) kk_gpu_compute(g,CONV_CRELU,"ccr",&(kk_compute_args){.out=cv2[k+1],.in={cv2[k]},.n_in=1,.uniforms=g_w+Ucr[k],.uniforms_size=292*4});
    struct {float m[224];float b[4];unsigned resid;} cu; memcpy(cu.m,g_w+Ucomb,224*4); memcpy(cu.b,g_w+Ucomb+224,16); cu.resid=0;
    kk_gpu_compute(g,COMBINE,"comb",&(kk_compute_args){.out=lastT,.in={cv2[0],cv2[1],cv2[2],cv2[3],cv2[4],cv2[5],cv2[6],restored},.n_in=8,.uniforms=&cu,.uniforms_size=sizeof cu});
    // Depth-to-Space (2×) + bilinear-restored-Residual
    kk_gpu_compute(g,D2S,"d2s",&(kk_compute_args){.out=a4kOut,.in={lastT,restored},.n_in=2,.linear={false,true}});
    return a4kOut;
}

// ===== ArtCNN_C4F16 (Realfilm-SD-Luma-Upscaler, HOOK LUMA) =====
// 9-Pass-Kette aus headless kk_artcnn_full (maxerr 1 vs CPU): LUMA -> [P0 1ch-vec4-Conv,
// 16feat-2×2] -> [P1-5 16ch-mat4-Conv+ReLU]×5 -> [P6 Skip(conv2d_5+conv2d) 16->4] ->
// Depth-to-Space -> 2×-LUMA. Input = R8-Luma (W×H), Output = 2×-Luma (RGBA16F, .r).
static const char *A_K0=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float w[144];float b[16];};\n"
"kernel void ak0(texture2d<float> luma [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
" constant W& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint lw=luma.get_width(),lh=luma.get_height(); if(gid.x>=lw||gid.y>=lh)return;\n"
" float4 r[4]; for(int k=0;k<4;k++) r[k]=float4(wb.b[k*4],wb.b[k*4+1],wb.b[k*4+2],wb.b[k*4+3]);\n"
" for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 lp=clamp(int2(gid)+int2(X-1,Y-1),int2(0),int2(int(lw)-1,int(lh)-1));\n"
"   float v=luma.read(uint2(lp)).x; int sp=Y*3+X;\n"
"   for(int k=0;k<4;k++){ int b=(k*9+sp)*4; r[k]+=float4(wb.w[b],wb.w[b+1],wb.w[b+2],wb.w[b+3])*v; } }\n"
" uint2 o=gid*2; dst.write(r[0],o); dst.write(r[1],o+uint2(1,0)); dst.write(r[2],o+uint2(0,1)); dst.write(r[3],o+uint2(1,1));}\n";
static const char *A_K16=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float w[2304];float b[16];};\n"
"kernel void ak16(texture2d<float> in0 [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
" constant W& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint lw=dst.get_width()/2,lh=dst.get_height()/2; if(gid.x>=lw||gid.y>=lh)return;\n"
" int iw=int(in0.get_width()),ih=int(in0.get_height()); int2 sub[4]={int2(0,0),int2(1,0),int2(0,1),int2(1,1)};\n"
" float4 r[4]; for(int k=0;k<4;k++) r[k]=float4(wb.b[k*4],wb.b[k*4+1],wb.b[k*4+2],wb.b[k*4+3]);\n"
" for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 ip=clamp(int2((int(gid.x)+X-1)*2+sub[C].x,(int(gid.y)+Y-1)*2+sub[C].y),int2(0),int2(iw-1,ih-1));\n"
"   float4 v=in0.read(uint2(ip)); int sp=Y*3+X;\n"
"   for(int k=0;k<4;k++){ int b=((k*4+C)*9+sp)*16; float4x4 M=float4x4(wb.w[b],wb.w[b+1],wb.w[b+2],wb.w[b+3],wb.w[b+4],wb.w[b+5],wb.w[b+6],wb.w[b+7],wb.w[b+8],wb.w[b+9],wb.w[b+10],wb.w[b+11],wb.w[b+12],wb.w[b+13],wb.w[b+14],wb.w[b+15]); r[k]+=M*v; } }\n"
" uint2 o=gid*2; dst.write(max(r[0],0.0),o); dst.write(max(r[1],0.0),o+uint2(1,0)); dst.write(max(r[2],0.0),o+uint2(0,1)); dst.write(max(r[3],0.0),o+uint2(1,1));}\n";
static const char *A_K6=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float w[576];float b[4];};\n"
"kernel void ak6(texture2d<float> in5 [[texture(0)]], texture2d<float> inS [[texture(1)]], texture2d<float,access::write> dst [[texture(2)]],\n"
" constant W& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint lw=dst.get_width(),lh=dst.get_height(); if(gid.x>=lw||gid.y>=lh)return;\n"
" int iw=int(in5.get_width()),ih=int(in5.get_height()); int2 sub[4]={int2(0,0),int2(1,0),int2(0,1),int2(1,1)};\n"
" float4 r=float4(wb.b[0],wb.b[1],wb.b[2],wb.b[3]);\n"
" for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 ip=clamp(int2((int(gid.x)+X-1)*2+sub[C].x,(int(gid.y)+Y-1)*2+sub[C].y),int2(0),int2(iw-1,ih-1));\n"
"   float4 v=in5.read(uint2(ip))+inS.read(uint2(ip)); int sp=Y*3+X; int b=(C*9+sp)*16;\n"
"   float4x4 M=float4x4(wb.w[b],wb.w[b+1],wb.w[b+2],wb.w[b+3],wb.w[b+4],wb.w[b+5],wb.w[b+6],wb.w[b+7],wb.w[b+8],wb.w[b+9],wb.w[b+10],wb.w[b+11],wb.w[b+12],wb.w[b+13],wb.w[b+14],wb.w[b+15]); r+=M*v; }\n"
" dst.write(r,gid);}\n";
static const char *A_KD2S=
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void ad2s(texture2d<float> c6 [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]){\n"
" uint w=dst.get_width(),h=dst.get_height(); if(gid.x>=w||gid.y>=h)return;\n"
" uint2 t=gid/2; uint comp=(gid.y%2)*2+(gid.x%2); float4 v=c6.read(t); float l=clamp(v[comp],0.0,1.0); dst.write(float4(l,l,l,1.0),gid);}\n";

static float *g_aw = NULL;   // 12340 floats
static kk_tex *ac[6]={0}, *ac6=NULL, *acOut=NULL;
static int acW=0, acH=0;

// ArtCNN auf R8-Luma (W×H) -> 2×-Luma (RGBA16F, .r). NULL bei Fehler.
kk_tex *kk_gpu_artcnn(kk_gpu *g, kk_tex *luma, const char *weights_path) {
    if (!g_aw) {
        FILE *f=fopen(weights_path,"rb"); if(!f) return NULL;
        g_aw=malloc(12340*4); if(fread(g_aw,4,12340,f)!=12340){fclose(f);free(g_aw);g_aw=NULL;return NULL;} fclose(f);
    }
    int W=kk_tex_w(luma), H=kk_tex_h(luma), IW=W*2, IH=H*2;
    if (W!=acW || H!=acH) {
        for(int i=0;i<6;i++) kk_tex_destroy(g,&ac[i]);
        kk_tex_destroy(g,&ac6); kk_tex_destroy(g,&acOut);
        for(int i=0;i<6;i++) ac[i]=kk_tex_create(g,IW,IH,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        ac6=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        acOut=kk_tex_create(g,IW,IH,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        acW=W; acH=H;
    }
    if (!acOut) return NULL;
    int OFF[7]={0,160,2480,4800,7120,9440,11760};
    kk_gpu_compute(g,A_K0,"ak0",&(kk_compute_args){.out=ac[0],.in={luma},.n_in=1,.uniforms=g_aw+OFF[0],.uniforms_size=160*4});
    for(int p=1;p<=5;p++) kk_gpu_compute(g,A_K16,"ak16",&(kk_compute_args){.out=ac[p],.in={ac[p-1]},.n_in=1,.uniforms=g_aw+OFF[p],.uniforms_size=2320*4});
    kk_gpu_compute(g,A_K6,"ak6",&(kk_compute_args){.out=ac6,.in={ac[5],ac[0]},.n_in=2,.uniforms=g_aw+OFF[6],.uniforms_size=580*4});
    kk_gpu_compute(g,A_KD2S,"ad2s",&(kk_compute_args){.out=acOut,.in={ac6},.n_in=1});
    return acOut;
}

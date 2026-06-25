// kk_anime4k_full.m — VOLLE Anime4K-Mode-A(M)-Kette auf kk_gpu vs CPU-Ref.
// De-Ring (5×5-Max-Pool+Clamp) -> Restore-CNN (conv_rgb + 6×CReLU + 1×1-Combine+MAIN-Res)
// -> Upscale-CNN (conv_rgb + 6×CReLU + 1×1-Combine -> conv_last) -> D2S(2×)+MAIN-Residual.
// Alle Conv 1× MAIN-Res (kein 2×2-Packing). Weights gen_a4k_all.py (1:1 GLSL). Validiert
// die volle Assemblierung (Chaining + Combine-14-Input + Residuals + D2S). Grau-Input
// (Luma) -> RGB=grau, damit der BT.709-De-Ring-Trick + RGB-Conv sauber sind.
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "kk_gpu.h"
#define W 16
#define H 16
#define OW (W*2)
#define OH (H*2)
// ---- Kernels ----
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

static float* sl(float*a,int o){return a+o;}
#define CL(a,lo,hi) ((a)<(lo)?(lo):(a)>(hi)?(hi):(a))
static float lumc(float r,float gg,float bb){return 0.299f*r+0.587f*gg+0.114f*bb;}
static void convrgb(float*src3,float*dst4,float*w){ for(int y=0;y<H;y++)for(int x=0;x<W;x++){float r[4]={w[144],w[145],w[146],w[147]};
  for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int px=CL(x+X-1,0,W-1),py=CL(y+Y-1,0,H-1);float v[4]={src3[(py*W+px)*3],src3[(py*W+px)*3+1],src3[(py*W+px)*3+2],0};int sp=Y*3+X;int b=sp*16;
    for(int row=0;row<4;row++)for(int col=0;col<4;col++)r[row]+=w[b+col*4+row]*v[col];} for(int c=0;c<4;c++)dst4[(y*W+x)*4+c]=r[c];} }
static void convcrelu(float*src4,float*dst4,float*w){ for(int y=0;y<H;y++)for(int x=0;x<W;x++){float r[4]={w[288],w[289],w[290],w[291]};
  for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int px=CL(x+X-1,0,W-1),py=CL(y+Y-1,0,H-1);float v[4];for(int c=0;c<4;c++)v[c]=src4[(py*W+px)*4+c];int sp=Y*3+X;
    for(int a=0;a<2;a++){int b=(a*9+sp)*16;for(int row=0;row<4;row++)for(int col=0;col<4;col++){float gg=a==0?fmaxf(v[col],0):fmaxf(-v[col],0);r[row]+=w[b+col*4+row]*gg;}}} for(int c=0;c<4;c++)dst4[(y*W+x)*4+c]=r[c];} }
int main(void){
    @autoreleasepool {
        FILE *f=fopen("/tmp/a4k_all.bin","rb"); if(!f){fprintf(stderr,"bin\n");return 1;}
        float *A=malloc(4256*4); fread(A,4,4256,f); fclose(f);
        // Offsets: restore: rgb@0(148), crelu@148+ k*292 (k=0..5), combine@1900(228)
        //          upscale: rgb@2128(148), crelu@2276+k*292, combine@4028(228)
        int Rrgb=0, Rcr[6], Rcomb=1900, Urgb=2128, Ucr[6], Ucomb=4028;
        for(int k=0;k<6;k++){Rcr[k]=148+k*292; Ucr[k]=2276+k*292;}
        // Grau-Input (Luma-Muster) -> RGB=grau.
        unsigned char *in=malloc(W*H*4);
        for(int i=0;i<W*H;i++){unsigned char v=(unsigned char)((i*53+17)&255); in[i*4]=v;in[i*4+1]=v;in[i*4+2]=v;in[i*4+3]=255;}
        kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu\n");return 1;}
        kk_tex *mainT=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_SAMPLE,in);
        kk_tex *s1=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *s2=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *deringed=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *cv[7]; for(int i=0;i<7;i++) cv[i]=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *restored=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *cv2[7]; for(int i=0;i<7;i++) cv2[i]=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *last=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *out=kk_tex_create(g,OW,OH,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
        unsigned int vH=0,vV=1;
        // De-Ring
        kk_gpu_compute(g,STAT,"stat",&(kk_compute_args){.out=s1,.in={mainT},.n_in=1,.uniforms=&vH,.uniforms_size=4});
        kk_gpu_compute(g,STAT,"stat",&(kk_compute_args){.out=s2,.in={s1},.n_in=1,.uniforms=&vV,.uniforms_size=4});
        kk_gpu_compute(g,CLAMPK,"clmp",&(kk_compute_args){.out=deringed,.in={mainT,s2},.n_in=2});
        // Restore-CNN
        kk_gpu_compute(g,CONV_RGB,"crgb",&(kk_compute_args){.out=cv[0],.in={deringed},.n_in=1,.uniforms=sl(A,Rrgb),.uniforms_size=148*4});
        for(int k=0;k<6;k++) kk_gpu_compute(g,CONV_CRELU,"ccr",&(kk_compute_args){.out=cv[k+1],.in={cv[k]},.n_in=1,.uniforms=sl(A,Rcr[k]),.uniforms_size=292*4});
        struct {float m[224];float b[4];unsigned int resid;} cb; memcpy(cb.m,sl(A,Rcomb),224*4); memcpy(cb.b,sl(A,Rcomb)+224,16); cb.resid=1;
        kk_gpu_compute(g,COMBINE,"comb",&(kk_compute_args){.out=restored,.in={cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],deringed},.n_in=8,.uniforms=&cb,.uniforms_size=sizeof cb});
        // Upscale-CNN
        kk_gpu_compute(g,CONV_RGB,"crgb",&(kk_compute_args){.out=cv2[0],.in={restored},.n_in=1,.uniforms=sl(A,Urgb),.uniforms_size=148*4});
        for(int k=0;k<6;k++) kk_gpu_compute(g,CONV_CRELU,"ccr",&(kk_compute_args){.out=cv2[k+1],.in={cv2[k]},.n_in=1,.uniforms=sl(A,Ucr[k]),.uniforms_size=292*4});
        struct {float m[224];float b[4];unsigned int resid;} cu; memcpy(cu.m,sl(A,Ucomb),224*4); memcpy(cu.b,sl(A,Ucomb)+224,16); cu.resid=0;
        kk_gpu_compute(g,COMBINE,"comb",&(kk_compute_args){.out=last,.in={cv2[0],cv2[1],cv2[2],cv2[3],cv2[4],cv2[5],cv2[6],restored},.n_in=8,.uniforms=&cu,.uniforms_size=sizeof cu});
        kk_gpu_compute(g,D2S,"d2s",&(kk_compute_args){.out=out,.in={last,restored},.n_in=2,.linear={false,true}});
        kk_gpu_finish(g);
        unsigned char *mine=malloc(OW*OH*4); kk_tex_download(g,out,mine);

        // ===== CPU-Ref =====
        // De-Ring
        float *L=malloc(W*H*4),*h1=malloc(W*H*4),*h2=malloc(W*H*4);
        for(int i=0;i<W*H;i++) L[i]=lumc(in[i*4]/255.0f,in[i*4+1]/255.0f,in[i*4+2]/255.0f);
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){float m=0;for(int i=0;i<5;i++){int q=CL(x+i-2,0,W-1);m=fmaxf(m,L[y*W+q]);}h1[y*W+x]=m;}
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){float m=0;for(int i=0;i<5;i++){int q=CL(y+i-2,0,H-1);m=fmaxf(m,h1[q*W+x]);}h2[y*W+x]=m;}
        float *dr=malloc(W*H*3*sizeof(float));
        for(int i=0;i<W*H;i++){float cl=L[i];float nl=fminf(cl,h2[i]);float dd=cl-nl;for(int c=0;c<3;c++)dr[i*3+c]=in[i*4+c]/255.0f-dd;}
        // conv_rgb helper
        float *C[7]; for(int i=0;i<7;i++)C[i]=malloc(W*H*4*sizeof(float));
        convrgb(dr,C[0],sl(A,Rrgb)); for(int k=0;k<6;k++)convcrelu(C[k],C[k+1],sl(A,Rcr[k]));
        // restore combine (+MAIN residual = deringed rgb)
        float *rest=malloc(W*H*3*sizeof(float)); float*w=sl(A,Rcomb);
        for(int i=0;i<W*H;i++){float r[4]={w[224],w[225],w[226],w[227]};
          for(int s=0;s<7;s++)for(int a=0;a<2;a++){int gi=s*2+a;int b=gi*16;for(int row=0;row<4;row++)for(int col=0;col<4;col++){float gg=a==0?fmaxf(C[s][i*4+col],0):fmaxf(-C[s][i*4+col],0);r[row]+=w[b+col*4+row]*gg;}}
          for(int c=0;c<3;c++) rest[i*3+c]=r[c]+dr[i*3+c]; }
        // upscale CNN
        float *C2[7]; for(int i=0;i<7;i++)C2[i]=malloc(W*H*4*sizeof(float));
        convrgb(rest,C2[0],sl(A,Urgb)); for(int k=0;k<6;k++)convcrelu(C2[k],C2[k+1],sl(A,Ucr[k]));
        float *lastc=malloc(W*H*4*sizeof(float)); float*wu=sl(A,Ucomb);
        for(int i=0;i<W*H;i++){float r[4]={wu[224],wu[225],wu[226],wu[227]};
          for(int s=0;s<7;s++)for(int a=0;a<2;a++){int gi=s*2+a;int b=gi*16;for(int row=0;row<4;row++)for(int col=0;col<4;col++){float gg=a==0?fmaxf(C2[s][i*4+col],0):fmaxf(-C2[s][i*4+col],0);r[row]+=wu[b+col*4+row]*gg;}}
          for(int c=0;c<4;c++) lastc[i*4+c]=r[c]; }
        // D2S + bilinear MAIN(rest) residual
        int maxe=0;double sume=0;int n=0;
        for(int oy=0;oy<OH;oy++)for(int ox=0;ox<OW;ox++){int tx=ox/2,ty=oy/2;int comp=(oy%2)*2+(ox%2);float det=lastc[(ty*W+tx)*4+comp];
          // bilinear sample von rest (grau) an (ox+0.5)/OW
          float fx=(ox+0.5f)/OW*W-0.5f, fy=(oy+0.5f)/OH*H-0.5f; int x0=(int)floorf(fx),y0=(int)floorf(fy); float ax=fx-x0,ay=fy-y0;
          float mres=0; for(int c=0;c<3;c++){ // grau -> rest channel egal, nimm [0]
            int X0=CL(x0,0,W-1),X1=CL(x0+1,0,W-1),Y0=CL(y0,0,H-1),Y1=CL(y0+1,0,H-1);
            float v00=rest[(Y0*W+X0)*3+c],v10=rest[(Y0*W+X1)*3+c],v01=rest[(Y1*W+X0)*3+c],v11=rest[(Y1*W+X1)*3+c];
            float vv=v00*(1-ax)*(1-ay)+v10*ax*(1-ay)+v01*(1-ax)*ay+v11*ax*ay; if(c==0)mres=vv; }
          float val=CL(det+mres,0,1); int ref=(int)(val*255+0.5f);
          int got=mine[(oy*OW+ox)*4]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; }
        printf("VOLLE Anime4K-Kette (De-Ring+Restore+Upscale+D2S) kk_gpu vs CPU-Ref: maxerr=%d mean=%.2f LSB %s\n",maxe,sume/n,maxe<=4?"PASS":"FAIL");
        kk_gpu_destroy(&g);
        return maxe<=4?0:1;
    }
}

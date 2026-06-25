// kk_a4k_dering.m — Anime4K De-Ring-Kette auf kk_gpu vs CPU-Ref (letzter Anime4K-Pass-Typ).
// stat1: 5-Tap-H-Luma-Max -> stat2: 5-Tap-V-Max (= separabler 5×5-Max-Pool) -> clamp:
// out = RGB - (luma - min(luma, statmax))  (Luma auf lokales Max runter, BT.709-Trick).
// luma = dot((0.299,0.587,0.114), rgb). Beweist den De-Ring-Pass-Typ; damit ALLE
// Anime4K-Pass-Typen auf kk_gpu (De-Ring + RGB-Conv + CReLU-Conv + 1×1-Combine + D2S).
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "kk_gpu.h"
#define W 24
#define H 24
static const char *STAT_MSL=
"#include <metal_stdlib>\nusing namespace metal;\nstruct P{uint vert;};\n"
"static inline float luma(float4 c){return dot(float4(0.299,0.587,0.114,0.0),c);}\n"
"kernel void stat(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
" constant P& p [[buffer(0)]], uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
" int iw=int(w),ih=int(h); float gmax=0.0;\n"
" for(int i=0;i<5;i++){ int2 o = p.vert? int2(0,i-2):int2(i-2,0); int2 q=clamp(int2(id)+o,int2(0),int2(iw-1,ih-1));\n"
"   float g = p.vert? src.read(uint2(q)).x : luma(src.read(uint2(q))); gmax=max(g,gmax); }\n"
" dst.write(float4(gmax,0,0,0),id);}\n";
static const char *CLAMP_MSL=
"#include <metal_stdlib>\nusing namespace metal;\n"
"static inline float luma(float4 c){return dot(float4(0.299,0.587,0.114,0.0),c);}\n"
"kernel void clmp(texture2d<float> rgb [[texture(0)]], texture2d<float> stat [[texture(1)]], texture2d<float,access::write> dst [[texture(2)]],\n"
" uint2 id [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
" float4 c=rgb.read(id); float cl=luma(c); float nl=min(cl, stat.read(id).x);\n"
" dst.write(c-(cl-nl),id);}\n";
int main(void){
    @autoreleasepool {
        unsigned char *in=malloc(W*H*4);
        for(int i=0;i<W*H*4;i++) in[i]=(unsigned char)((i*47+11)%256);
        kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu\n");return 1;}
        kk_tex *it=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_SAMPLE,in);
        kk_tex *s1=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *s2=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *ot=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
        unsigned int vH=0,vV=1;
        kk_gpu_compute(g,STAT_MSL,"stat",&(kk_compute_args){.out=s1,.in={it},.n_in=1,.uniforms=&vH,.uniforms_size=4});
        kk_gpu_compute(g,STAT_MSL,"stat",&(kk_compute_args){.out=s2,.in={s1},.n_in=1,.uniforms=&vV,.uniforms_size=4});
        kk_gpu_compute(g,CLAMP_MSL,"clmp",&(kk_compute_args){.out=ot,.in={it,s2},.n_in=2});
        kk_gpu_finish(g);
        unsigned char *out=malloc(W*H*4); kk_tex_download(g,ot,out);
        // CPU-Ref
        float *L=malloc(W*H*sizeof(float));
        for(int i=0;i<W*H;i++) L[i]=0.299f*in[i*4]/255+0.587f*in[i*4+1]/255+0.114f*in[i*4+2]/255;
        float *h1=malloc(W*H*sizeof(float)),*h2=malloc(W*H*sizeof(float));
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){float m=0;for(int i=0;i<5;i++){int q=x+i-2;q=q<0?0:q>W-1?W-1:q;m=fmaxf(m,L[y*W+q]);}h1[y*W+x]=m;}
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){float m=0;for(int i=0;i<5;i++){int q=y+i-2;q=q<0?0:q>H-1?H-1:q;m=fmaxf(m,h1[q*W+x]);}h2[y*W+x]=m;}
        int maxe=0;double sume=0;int n=0;
        for(int i=0;i<W*H;i++){ float cl=L[i]; float nl=fminf(cl,h2[i]); float d=cl-nl;
            for(int c=0;c<3;c++){ float v=in[i*4+c]/255.0f - d; v=v<0?0:v>1?1:v; int ref=(int)(v*255+0.5f);
                int got=out[i*4+c]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; } }
        printf("Anime4K De-Ring (5x5-Max-Pool + Clamp) vs CPU-Ref: maxerr=%d mean=%.2f LSB %s\n",maxe,sume/n,maxe<=2?"PASS":"FAIL");
        kk_gpu_destroy(&g);
        return maxe<=2?0:1;
    }
}

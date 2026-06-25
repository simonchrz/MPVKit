// kk_a4k_conv.m — Anime4K CReLU-Conv (conv2d_1_tf, 4ch->4ch) auf kk_gpu vs CPU-Ref.
// Anime4Ks Kern-Pass-Typ: fragment-style 3×3-Conv mit CReLU-Aktivierung
// (go_0=max(v,0), go_1=max(-v,0) -> 8 effektive Inputs aus 4ch), 18 mat4 + bias.
// Weights aus gen_a4k.py (1:1 aus anime4k_mode_a_m.glsl). Input signed [-1,1] als RGBA8
// encodiert (Conv-Outputs sind signed), Kernel decodet inline (v*2-1). Beweist: Anime4Ks
// Architektur (andere als ArtCNN) laeuft via demselben Generator+kk_gpu-Muster.
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "kk_gpu.h"
#define W 16
#define H 16
struct WB { float m[18*16]; float b[4]; };  // [act0:9 mat4][act1:9 mat4] + bias
static const char *CONV_MSL=
"#include <metal_stdlib>\nusing namespace metal;\nstruct WB{float m[288];float b[4];};\n"
"kernel void conv(texture2d<float> in0 [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
" constant WB& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint w=dst.get_width(),h=dst.get_height(); if(gid.x>=w||gid.y>=h)return;\n"
" int iw=int(w),ih=int(h); float4 r=float4(wb.b[0],wb.b[1],wb.b[2],wb.b[3]);\n"
" for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 p=clamp(int2(gid)+int2(X-1,Y-1),int2(0),int2(iw-1,ih-1));\n"
"   float4 v=in0.read(uint2(p))*2.0-1.0; float4 g0=max(v,0.0),g1=max(-v,0.0); int sp=Y*3+X;\n"
"   for(int act=0;act<2;act++){ int b=(act*9+sp)*16;\n"
"     float4x4 M=float4x4(wb.m[b],wb.m[b+1],wb.m[b+2],wb.m[b+3],wb.m[b+4],wb.m[b+5],wb.m[b+6],wb.m[b+7],wb.m[b+8],wb.m[b+9],wb.m[b+10],wb.m[b+11],wb.m[b+12],wb.m[b+13],wb.m[b+14],wb.m[b+15]);\n"
"     r += M*(act==0?g0:g1); } }\n"
" dst.write(clamp(r*0.25+0.5,0.0,1.0),gid);}\n";
int main(void){
    @autoreleasepool {
        FILE *f=fopen("/tmp/a4k_c1.bin","rb"); if(!f){fprintf(stderr,"bin\n");return 1;}
        struct WB wb; fread(wb.m,4,288,f); fread(wb.b,4,4,f); fclose(f);
        unsigned char *in=malloc(W*H*4);
        for(int i=0;i<W*H*4;i++) in[i]=(unsigned char)((i*61+23)%256);  // encodet signed [-1,1]
        kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu\n");return 1;}
        kk_tex *it=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_SAMPLE,in);
        kk_tex *ot=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
        kk_gpu_compute(g,CONV_MSL,"conv",&(kk_compute_args){.out=ot,.in={it},.n_in=1,.uniforms=&wb,.uniforms_size=sizeof wb});
        kk_gpu_finish(g);
        unsigned char *out=malloc(W*H*4); kk_tex_download(g,ot,out);
        int maxe=0; double sume=0; int n=0;
        for(int gy=0;gy<H;gy++)for(int gx=0;gx<W;gx++){ float r[4]={wb.b[0],wb.b[1],wb.b[2],wb.b[3]};
            for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int px=gx+X-1,py=gy+Y-1; px=px<0?0:px>W-1?W-1:px; py=py<0?0:py>H-1?H-1:py;
                float v[4],g0[4],g1[4]; for(int c=0;c<4;c++){v[c]=in[(py*W+px)*4+c]/255.0f*2-1; g0[c]=v[c]>0?v[c]:0; g1[c]=-v[c]>0?-v[c]:0;}
                int sp=Y*3+X; for(int act=0;act<2;act++){int b=(act*9+sp)*16; float*gg=act==0?g0:g1;
                    for(int row=0;row<4;row++)for(int col=0;col<4;col++) r[row]+=wb.m[b+col*4+row]*gg[col]; } }
            for(int c=0;c<4;c++){ float val=r[c]*0.25f+0.5f; val=val<0?0:val>1?1:val; int ref=(int)(val*255+0.5f);
                int got=out[(gy*W+gx)*4+c]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; } }
        printf("Anime4K CReLU-Conv (conv2d_1_tf, 4->4) vs CPU-Ref: maxerr=%d mean=%.2f LSB %s\n",maxe,sume/n,maxe<=2?"PASS":"FAIL");
        kk_gpu_destroy(&g);
        return maxe<=2?0:1;
    }
}

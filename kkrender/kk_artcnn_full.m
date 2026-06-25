// kk_artcnn_full.m — VOLLE 9-Pass-ArtCNN-Kette auf kk_gpu. Weights aus /tmp/artcnn_all.bin
// (gen_all.py, 1:1 aus der GLSL). Kette: LUMA(r8) -> [P0 1ch-vec4-Conv 2×] -> conv2d ->
// [P1-5 16ch-mat4-Conv+ReLU] ×5 -> conv2d_5 -> [P6 Skip conv2d_5+conv2d, 16->4, 1×] ->
// conv2d_6 -> [Depth-to-Space Pixel-Shuffle] -> 2×-LUMA. Validiert gegen CPU-Ref der
// vollen Kette (per-Pass-Conv schon einzeln vs libplacebo-Weights = maxerr 0; hier die
// ASSEMBLIERUNG: Chaining + 2×2-Packing + Skip + D2S). Intermediates RGBA16F.
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "kk_gpu.h"
#define LW 16
#define LH 16
#define IW (LW*2)
#define IH (LH*2)

// P0: 1-Kanal-vec4-Conv, LUMA->16feat(2×2), kein ReLU. w[4*9*4]+bias[16].
static const char *K0=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float w[144];float b[16];};\n"
"kernel void k0(texture2d<float> luma [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
" constant W& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint lw=luma.get_width(),lh=luma.get_height(); if(gid.x>=lw||gid.y>=lh)return;\n"
" float4 r[4]; for(int k=0;k<4;k++) r[k]=float4(wb.b[k*4],wb.b[k*4+1],wb.b[k*4+2],wb.b[k*4+3]);\n"
" for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 lp=clamp(int2(gid)+int2(X-1,Y-1),int2(0),int2(int(lw)-1,int(lh)-1));\n"
"   float v=luma.read(uint2(lp)).x; int sp=Y*3+X;\n"
"   for(int k=0;k<4;k++){ int b=(k*9+sp)*4; r[k]+=float4(wb.w[b],wb.w[b+1],wb.w[b+2],wb.w[b+3])*v; } }\n"
" uint2 o=gid*2; dst.write(r[0],o); dst.write(r[1],o+uint2(1,0)); dst.write(r[2],o+uint2(0,1)); dst.write(r[3],o+uint2(1,1));}\n";
// P1-5: 16-Kanal-mat4-Conv, +ReLU, 2×2. w[2304]+bias[16].
static const char *K16=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float w[2304];float b[16];};\n"
"kernel void k16(texture2d<float> in0 [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
" constant W& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint lw=dst.get_width()/2,lh=dst.get_height()/2; if(gid.x>=lw||gid.y>=lh)return;\n"
" int iw=int(in0.get_width()),ih=int(in0.get_height()); int2 sub[4]={int2(0,0),int2(1,0),int2(0,1),int2(1,1)};\n"
" float4 r[4]; for(int k=0;k<4;k++) r[k]=float4(wb.b[k*4],wb.b[k*4+1],wb.b[k*4+2],wb.b[k*4+3]);\n"
" for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 ip=clamp(int2((int(gid.x)+X-1)*2+sub[C].x,(int(gid.y)+Y-1)*2+sub[C].y),int2(0),int2(iw-1,ih-1));\n"
"   float4 v=in0.read(uint2(ip)); int sp=Y*3+X;\n"
"   for(int k=0;k<4;k++){ int b=((k*4+C)*9+sp)*16; float4x4 M=float4x4(wb.w[b],wb.w[b+1],wb.w[b+2],wb.w[b+3],wb.w[b+4],wb.w[b+5],wb.w[b+6],wb.w[b+7],wb.w[b+8],wb.w[b+9],wb.w[b+10],wb.w[b+11],wb.w[b+12],wb.w[b+13],wb.w[b+14],wb.w[b+15]); r[k]+=M*v; } }\n"
" uint2 o=gid*2; dst.write(max(r[0],0.0),o); dst.write(max(r[1],0.0),o+uint2(1,0)); dst.write(max(r[2],0.0),o+uint2(0,1)); dst.write(max(r[3],0.0),o+uint2(1,1));}\n";
// P6: Skip (conv2d_5 + conv2d) -> 16->4, 1×, kein ReLU. w[576]+bias[4].
static const char *K6=
"#include <metal_stdlib>\nusing namespace metal;\nstruct W{float w[576];float b[4];};\n"
"kernel void k6(texture2d<float> in5 [[texture(0)]], texture2d<float> inS [[texture(1)]], texture2d<float,access::write> dst [[texture(2)]],\n"
" constant W& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){ uint lw=dst.get_width(),lh=dst.get_height(); if(gid.x>=lw||gid.y>=lh)return;\n"
" int iw=int(in5.get_width()),ih=int(in5.get_height()); int2 sub[4]={int2(0,0),int2(1,0),int2(0,1),int2(1,1)};\n"
" float4 r=float4(wb.b[0],wb.b[1],wb.b[2],wb.b[3]);\n"
" for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){ int2 ip=clamp(int2((int(gid.x)+X-1)*2+sub[C].x,(int(gid.y)+Y-1)*2+sub[C].y),int2(0),int2(iw-1,ih-1));\n"
"   float4 v=in5.read(uint2(ip))+inS.read(uint2(ip)); int sp=Y*3+X; int b=(C*9+sp)*16;\n"
"   float4x4 M=float4x4(wb.w[b],wb.w[b+1],wb.w[b+2],wb.w[b+3],wb.w[b+4],wb.w[b+5],wb.w[b+6],wb.w[b+7],wb.w[b+8],wb.w[b+9],wb.w[b+10],wb.w[b+11],wb.w[b+12],wb.w[b+13],wb.w[b+14],wb.w[b+15]); r+=M*v; }\n"
" dst.write(r,gid);}\n";
// Depth-to-Space: conv2d_6(4feat) -> 2×-LUMA. out[2x+dx,2y+dy]=c6[x,y][dy*2+dx].
static const char *KD2S=
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void d2s(texture2d<float> c6 [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]], uint2 gid [[thread_position_in_grid]]){\n"
" uint w=dst.get_width(),h=dst.get_height(); if(gid.x>=w||gid.y>=h)return;\n"
" uint2 t=gid/2; uint comp=(gid.y%2)*2+(gid.x%2); float4 v=c6.read(t); float l=clamp(v[comp],0.0,1.0); dst.write(float4(l,l,l,1.0),gid);}\n";

static float* slice(float*all,int off){return all+off;}
int main(void){
    @autoreleasepool {
        FILE *f=fopen("/tmp/artcnn_all.bin","rb"); if(!f){fprintf(stderr,"bin fehlt\n");return 1;}
        float *W=malloc(12340*4); fread(W,4,12340,f); fclose(f);
        int OFF[7]={0,160,2480,4800,7120,9440,11760}; // P0,P1..5,P6
        unsigned char *luma=malloc(LW*LH);
        for(int i=0;i<LW*LH;i++) luma[i]=(unsigned char)((i*53+17)&255);
        kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu\n");return 1;}
        kk_tex *lt=kk_tex_create(g,LW,LH,KK_FMT_R8,KK_TEX_SAMPLE,luma);
        kk_tex *c[7]; for(int i=0;i<6;i++) c[i]=kk_tex_create(g,IW,IH,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL); // conv2d..conv2d_5
        kk_tex *c6=kk_tex_create(g,LW,LH,KK_FMT_RGBA16F,KK_TEX_SAMPLE|KK_TEX_STORAGE,NULL);
        kk_tex *out=kk_tex_create(g,IW,IH,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
        // P0: luma -> c[0] (conv2d)
        kk_gpu_compute(g,K0,"k0",&(kk_compute_args){.out=c[0],.in={lt},.n_in=1,.uniforms=slice(W,OFF[0]),.uniforms_size=160*4});
        // P1-5: c[i] -> c[i+1]
        for(int p=1;p<=5;p++) kk_gpu_compute(g,K16,"k16",&(kk_compute_args){.out=c[p],.in={c[p-1]},.n_in=1,.uniforms=slice(W,OFF[p]),.uniforms_size=2320*4});
        // P6: Skip c[5](conv2d_5) + c[0](conv2d) -> c6
        kk_gpu_compute(g,K6,"k6",&(kk_compute_args){.out=c6,.in={c[5],c[0]},.n_in=2,.uniforms=slice(W,OFF[6]),.uniforms_size=580*4});
        // D2S: c6 -> out (2×)
        kk_gpu_compute(g,KD2S,"d2s",&(kk_compute_args){.out=out,.in={c6},.n_in=1});
        kk_gpu_finish(g);
        unsigned char *mine=malloc(IW*IH*4); kk_tex_download(g,out,mine);

        // ===== CPU-Referenz der vollen Kette =====
        int sub[4][2]={{0,0},{1,0},{0,1},{1,1}};
        // P0
        float *cf[6]; for(int i=0;i<6;i++) cf[i]=malloc(IW*IH*4*sizeof(float));
        for(int gy=0;gy<LH;gy++)for(int gx=0;gx<LW;gx++){ float r[4][4]; float*w=slice(W,OFF[0]);
          for(int k=0;k<4;k++)for(int cc=0;cc<4;cc++)r[k][cc]=w[144+k*4+cc];
          for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int lx=gx+X-1,ly=gy+Y-1; lx=lx<0?0:lx>LW-1?LW-1:lx; ly=ly<0?0:ly>LH-1?LH-1:ly;
            float v=luma[ly*LW+lx]/255.0f; int sp=Y*3+X; for(int k=0;k<4;k++)for(int cc=0;cc<4;cc++)r[k][cc]+=w[(k*9+sp)*4+cc]*v; }
          for(int k=0;k<4;k++){int ox=gx*2+sub[k][0],oy=gy*2+sub[k][1]; for(int cc=0;cc<4;cc++) cf[0][(oy*IW+ox)*4+cc]=r[k][cc]; } }
        // P1-5
        for(int p=1;p<=5;p++){ float*w=slice(W,OFF[p]); float*in=cf[p-1],*ou=cf[p];
          for(int gy=0;gy<LH;gy++)for(int gx=0;gx<LW;gx++){ float r[4][4]; for(int k=0;k<4;k++)for(int cc=0;cc<4;cc++)r[k][cc]=w[2304+k*4+cc];
            for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int ix=(gx+X-1)*2+sub[C][0],iy=(gy+Y-1)*2+sub[C][1]; ix=ix<0?0:ix>IW-1?IW-1:ix; iy=iy<0?0:iy>IH-1?IH-1:iy;
              float vv[4]; for(int cc=0;cc<4;cc++)vv[cc]=in[(iy*IW+ix)*4+cc]; int sp=Y*3+X;
              for(int k=0;k<4;k++){int b=((k*4+C)*9+sp)*16; for(int row=0;row<4;row++)for(int col=0;col<4;col++)r[k][row]+=w[b+col*4+row]*vv[col]; } }
            for(int k=0;k<4;k++){int ox=gx*2+sub[k][0],oy=gy*2+sub[k][1]; for(int cc=0;cc<4;cc++){float val=r[k][cc]; ou[(oy*IW+ox)*4+cc]=val<0?0:val; } } } }
        // P6: c6cpu (16×16×4)
        float *c6cpu=malloc(LW*LH*4*sizeof(float)); float*w6=slice(W,OFF[6]);
        for(int gy=0;gy<LH;gy++)for(int gx=0;gx<LW;gx++){ float r[4]; for(int cc=0;cc<4;cc++)r[cc]=w6[576+cc];
          for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){int ix=(gx+X-1)*2+sub[C][0],iy=(gy+Y-1)*2+sub[C][1]; ix=ix<0?0:ix>IW-1?IW-1:ix; iy=iy<0?0:iy>IH-1?IH-1:iy;
            float vv[4]; for(int cc=0;cc<4;cc++)vv[cc]=cf[5][(iy*IW+ix)*4+cc]+cf[0][(iy*IW+ix)*4+cc]; int sp=Y*3+X; int b=(C*9+sp)*16;
            for(int row=0;row<4;row++)for(int col=0;col<4;col++)r[row]+=w6[b+col*4+row]*vv[col]; }
          for(int cc=0;cc<4;cc++) c6cpu[(gy*LW+gx)*4+cc]=r[cc]; }
        // D2S + compare
        int maxe=0; double sume=0; int n=0;
        for(int oy=0;oy<IH;oy++)for(int ox=0;ox<IW;ox++){ int tx=ox/2,ty=oy/2; int comp=(oy%2)*2+(ox%2);
          float v=c6cpu[(ty*LW+tx)*4+comp]; v=v<0?0:v>1?1:v; int ref=(int)(v*255+0.5f);
          int got=mine[(oy*IW+ox)*4]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; }
        printf("VOLLE 9-Pass-ArtCNN-Kette (kk_gpu) vs CPU-Ref: maxerr=%d mean=%.2f LSB %s\n",maxe,sume/n,maxe<=2?"PASS":"FAIL");
        kk_gpu_destroy(&g);
        return maxe<=2?0:1;
    }
}

// kk_tg_test.m — validiert das Threadgroup-Shared-Memory-Compute-Muster auf kk_gpu
// (die einzige neue Fähigkeit, die Multi-Pass-CNNs wie ArtCNN über das schon bewiesene
// Multi-Pass-Chaining hinaus brauchen): kooperatives 18×18-Tile-Load (16×16 Threads +
// 1px-Halo) → threadgroup_barrier → 3×3-Mittel aus Shared-Mem. Gegen CPU-3×3-Box-Ref.
// Beweist: kk_gpus dispatchThreads 16×16 + threadgroup-Arrays + Barrier funktionieren →
// die transpilierten CNN-Conv-Pässe (gl_WorkGroupSize-Tiling) laufen auf kk_gpu.
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "kk_gpu.h"
#define W 32
#define H 32

static const char *TG_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void blur(texture2d<float> src [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],\n"
"  uint2 tgid [[threadgroup_position_in_grid]]){\n"
"  threadgroup float3 tile[18*18];\n"
"  int sw=int(src.get_width()),sh=int(src.get_height());\n"
"  int2 base=int2(tgid)*16-1;                       // Tile-Ursprung (mit -1 Halo)\n"
"  uint li=lid.y*16u+lid.x;\n"
"  for(uint i=li;i<18u*18u;i+=256u){ int2 t=int2(int(i%18u),int(i/18u));\n"
"     int2 s=clamp(base+t,int2(0),int2(sw-1,sh-1)); tile[i]=src.read(uint2(s)).rgb; }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if(gid.x>=uint(sw)||gid.y>=uint(sh)) return;\n"
"  float3 sum=float3(0.0);\n"
"  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){ uint idx=uint((int(lid.y)+1+dy)*18+(int(lid.x)+1+dx)); sum+=tile[idx]; }\n"
"  dst.write(float4(sum/9.0,1.0),gid);}\n";

int main(void){
    kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu fail\n");return 1;}
    unsigned char *in=malloc(W*H*4);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){int i=(y*W+x)*4; in[i]=(x*255)/(W-1); in[i+1]=(y*255)/(H-1); in[i+2]=((x+y)&1)?220:40; in[i+3]=255;}
    kk_tex *src=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_SAMPLE,in);
    kk_tex *dst=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
    kk_gpu_compute(g,TG_MSL,"blur",&(kk_compute_args){.out=dst,.in={src},.n_in=1});
    kk_gpu_finish(g);
    unsigned char *out=malloc(W*H*4); kk_tex_download(g,dst,out);
    // CPU-Referenz: 3×3-Box, clamp-to-edge.
    int maxe=0; double sume=0; int n=0;
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)for(int c=0;c<3;c++){
        int s=0; for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){int sx=x+dx,sy=y+dy; sx=sx<0?0:sx>W-1?W-1:sx; sy=sy<0?0:sy>H-1?H-1:sy; s+=in[(sy*W+sx)*4+c];}
        int ref=(int)roundf(s/9.0f); int got=out[(y*W+x)*4+c]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; }
    printf("Threadgroup-Tile-Blur (Shared-Mem+Barrier) vs CPU-Box: maxerr=%d LSB  mean=%.2f LSB  %s\n",
           maxe,sume/n, maxe<=1?"PASS":"FAIL");
    kk_gpu_destroy(&g);
    return maxe<=1?0:1;
}

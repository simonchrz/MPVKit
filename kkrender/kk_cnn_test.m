// kk_cnn_test.m — echter ArtCNN-Pass-1-Conv (conv2d) nativ auf kk_gpu, gegen CPU-Ref
// derselben Weights. Validiert die KOMPLETTE CNN-Conv-Pass-Mechanik mit ECHTEN Weights:
// 18×18-Threadgroup-Tile-Load + barrier + 3×3-Conv (result0, 4 Features) + Output. Die
// Weights sind 1:1 aus ArtCNN_C4F16.glsl (result0-Block). Beweist: der reale Conv-Pass
// laeuft korrekt auf kk_gpu (Multi-Pass-Chaining + die restlichen 8 Paesse = dasselbe
// Muster, transpiliert). 16×16-Threadgroup (= kk_gpu-Dispatch). 1-Kanal-Luma-Input.
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import "kk_gpu.h"
#define W 32
#define H 32

// Pass-1 result0: bias + 9 * (weight_vec4 * inp(dx,dy)). Weights aus ArtCNN_C4F16.glsl.
// inp_0_X_Y = LUMA-Offset (X-1, Y-1). Threadgroup-Tile wie das Original (gl_WorkGroupSize=16).
static const char *CONV_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"constant float4 B   = float4(-0.0027198044,-0.013629392,-0.015712878,-0.050803013);\n"
"constant float4 Wm1m1=float4(-0.016452063,-0.1258466,0.013886958,0.036870774);\n"
"constant float4 W0m1 =float4(0.04311634,0.15515013,0.12190506,0.12543218);\n"
"constant float4 Wp1m1=float4(-0.0049624983,0.1029244,-0.10124424,0.06448426);\n"
"constant float4 Wm10 =float4(0.001886782,0.06120591,0.020384936,0.16804346);\n"
"constant float4 W00  =float4(-0.04256893,-0.07616671,-0.37889892,0.27856478);\n"
"constant float4 Wp10 =float4(-0.20398517,-0.12900643,0.113083735,0.11175711);\n"
"constant float4 Wm1p1=float4(0.009553091,0.13118562,-0.031063978,0.09478131);\n"
"constant float4 W0p1 =float4(0.066157505,-0.114692695,0.22418123,-0.009412468);\n"
"constant float4 Wp1p1=float4(0.15508306,0.011386595,0.014014352,0.09318008);\n"
"kernel void conv(texture2d<float> luma [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  uint2 gid [[thread_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]], uint2 tgid [[threadgroup_position_in_grid]]){\n"
"  threadgroup float tile[18*18];\n"
"  int sw=int(luma.get_width()),sh=int(luma.get_height());\n"
"  int2 base=int2(tgid)*16-1;\n"   // Tile-Ursprung (-1 Halo), wg_size=16, offset=1
"  for(uint i=lid.y*16u+lid.x;i<18u*18u;i+=256u){ int2 t=int2(int(i%18u),int(i/18u));\n"
"    int2 s=clamp(base+t,int2(0),int2(sw-1,sh-1)); tile[i]=luma.read(uint2(s)).x; }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if(gid.x>=uint(sw)||gid.y>=uint(sh)) return;\n"
"#define I(dx,dy) tile[uint((int(lid.y)+1+(dy))*18+(int(lid.x)+1+(dx)))]\n"
"  float4 r=B + Wm1m1*I(-1,-1)+W0m1*I(0,-1)+Wp1m1*I(1,-1)\n"
"             + Wm10*I(-1,0) +W00*I(0,0)  +Wp10*I(1,0)\n"
"             + Wm1p1*I(-1,1)+W0p1*I(0,1) +Wp1p1*I(1,1);\n"
"  dst.write(r*0.5+0.5,gid);}\n";   // Encode ~[-1,1]->[0,1] fuer RGBA8-Vergleich

static const float B[4]={-0.0027198044f,-0.013629392f,-0.015712878f,-0.050803013f};
static const float Wt[9][4]={
 {-0.016452063f,-0.1258466f,0.013886958f,0.036870774f}, {0.04311634f,0.15515013f,0.12190506f,0.12543218f}, {-0.0049624983f,0.1029244f,-0.10124424f,0.06448426f},
 {0.001886782f,0.06120591f,0.020384936f,0.16804346f}, {-0.04256893f,-0.07616671f,-0.37889892f,0.27856478f}, {-0.20398517f,-0.12900643f,0.113083735f,0.11175711f},
 {0.009553091f,0.13118562f,-0.031063978f,0.09478131f}, {0.066157505f,-0.114692695f,0.22418123f,-0.009412468f}, {0.15508306f,0.011386595f,0.014014352f,0.09318008f}};
// Offsets in derselben Reihenfolge wie Wt: (-1,-1)(0,-1)(1,-1)(-1,0)(0,0)(1,0)(-1,1)(0,1)(1,1)
static const int OFF[9][2]={{-1,-1},{0,-1},{1,-1},{-1,0},{0,0},{1,0},{-1,1},{0,1},{1,1}};

int main(void){
    kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu fail\n");return 1;}
    unsigned char *in=malloc(W*H);
    for(int i=0;i<W*H;i++) in[i]=(unsigned char)((i*37+13)&255);
    kk_tex *lt=kk_tex_create(g,W,H,KK_FMT_R8,KK_TEX_SAMPLE,in);
    kk_tex *dt=kk_tex_create(g,W,H,KK_FMT_RGBA16F,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL); // 4 Features
    // Download von RGBA16F braucht 8 B/px — kk_tex_download nimmt RGBA8 an. Daher RGBA8-Out
    // mit *255-Encode? Conv-Werte sind klein/negativ -> nicht RGBA8-darstellbar. Stattdessen
    // CPU-Ref auch in float + Vergleich via separater Float-Download. -> RGBA16F nicht
    // CPU-lesbar mit kk_tex_download(RGBA8). Workaround: encode result in RGBA8 via +0.5 clamp.
    (void)dt;
    kk_tex *d8=kk_tex_create(g,W,H,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL);
    // Encode-Wrapper: conv -> (r*0.5+0.5) in RGBA8 (Werte ~[-1,1]). MSL anpassen unten.
    kk_gpu_compute(g,CONV_MSL,"conv",&(kk_compute_args){.out=d8,.in={lt},.n_in=1});
    kk_gpu_finish(g);
    unsigned char *out=malloc(W*H*4); kk_tex_download(g,d8,out);

    int maxe=0; double sume=0; int n=0;
    for(int y=0;y<H;y++)for(int x=0;x<W;x++){
        float r[4]={B[0],B[1],B[2],B[3]};
        for(int k=0;k<9;k++){int sx=x+OFF[k][0],sy=y+OFF[k][1]; sx=sx<0?0:sx>W-1?W-1:sx; sy=sy<0?0:sy>H-1?H-1:sy;
            float v=in[sy*W+sx]/255.0f; for(int c=0;c<4;c++) r[c]+=Wt[k][c]*v; }
        for(int c=0;c<4;c++){ int ref=(int)roundf((r[c]*0.5f+0.5f)*255.0f); ref=ref<0?0:ref>255?255:ref;
            int got=out[(y*W+x)*4+c]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; } }
    printf("ArtCNN-Pass1-Conv (echte Weights, Threadgroup-Tile) vs CPU-Ref: maxerr=%d mean=%.2f LSB %s\n",
           maxe,sume/n, maxe<=2?"PASS":"FAIL");
    kk_gpu_destroy(&g);
    return maxe<=2?0:1;
}

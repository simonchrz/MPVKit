// kk_artcnn_p2.m — generischer Multi-Kanal-Conv (ArtCNN Pass 2, 16->16, mat4) auf kk_gpu
// vs CPU-Ref. Weights aus /tmp/artcnn_p2.bin (Generator gen_artcnn.py, 1:1 aus der GLSL).
// out_k[4]=bias_k + Σ_{C,X,Y} M4[k][C][Y*3+X] * inp_C_X_Y (mat4*vec4, column-major).
// inp_C_X_Y = input[((gx+X-1)*2+subX[C], (gy+Y-1)*2+subY[C])] (2×2-Feature-Packing).
// Output 2×2-Block (k: 0->(0,0)1->(1,0)2->(0,1)3->(1,1)) mit ReLU. Beweist den
// 16-Kanal-mat4-Conv (Pässe 2-6 = dasselbe, andere Weights) auf kk_gpu.
#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import "kk_gpu.h"
#define LW 16
#define LH 16
#define IW (LW*2)
#define IH (LH*2)
#define NW 2304   // 4*4*9*16
struct WB { float w[NW]; float bias[16]; };
static const char *CONV_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"struct WB{float w[2304];float bias[16];};\n"
"kernel void conv(texture2d<float> in0 [[texture(0)]], texture2d<float,access::write> dst [[texture(1)]],\n"
"  constant WB& wb [[buffer(0)]], uint2 gid [[thread_position_in_grid]]){\n"
"  uint LWv=dst.get_width()/2, LHv=dst.get_height()/2; if(gid.x>=LWv||gid.y>=LHv)return;\n"
"  int iw=int(in0.get_width()),ih=int(in0.get_height());\n"
"  int2 sub[4]={int2(0,0),int2(1,0),int2(0,1),int2(1,1)};\n"
"  float4 r[4]; for(int k=0;k<4;k++) r[k]=float4(wb.bias[k*4+0],wb.bias[k*4+1],wb.bias[k*4+2],wb.bias[k*4+3]);\n"
"  for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){\n"
"    int2 ip=int2((int(gid.x)+X-1)*2+sub[C].x, (int(gid.y)+Y-1)*2+sub[C].y);\n"
"    ip=clamp(ip,int2(0),int2(iw-1,ih-1)); float4 v=in0.read(uint2(ip));\n"
"    int sp=Y*3+X;\n"
"    for(int k=0;k<4;k++){ int base=((k*4+C)*9+sp)*16;\n"
"      float4x4 M=float4x4(wb.w[base+0],wb.w[base+1],wb.w[base+2],wb.w[base+3],\n"
"                          wb.w[base+4],wb.w[base+5],wb.w[base+6],wb.w[base+7],\n"
"                          wb.w[base+8],wb.w[base+9],wb.w[base+10],wb.w[base+11],\n"
"                          wb.w[base+12],wb.w[base+13],wb.w[base+14],wb.w[base+15]);\n"
"      r[k]+=M*v; } }\n"
"  uint2 o=gid*2;\n"
"  dst.write(max(r[0],0.0), o+uint2(0,0)); dst.write(max(r[1],0.0), o+uint2(1,0));\n"
"  dst.write(max(r[2],0.0), o+uint2(0,1)); dst.write(max(r[3],0.0), o+uint2(1,1));}\n";

int main(void){
    @autoreleasepool {
        FILE *f=fopen("/tmp/artcnn_p2.bin","rb"); if(!f){fprintf(stderr,"bin fehlt\n");return 1;}
        struct WB wb; fread(wb.w,4,NW,f); fread(wb.bias,4,16,f); fclose(f);
        // Synthetischer conv2d-Input (2×LUMA RGBA, Werte ~[0,1] wie nach ReLU/Conv).
        float *inf=malloc(IW*IH*4*sizeof(float));
        unsigned char *in8=malloc(IW*IH*4);
        for(int i=0;i<IW*IH*4;i++){ float v=((i*73+29)%211)/211.0f; inf[i]=v; in8[i]=(unsigned char)(v*255+0.5f); }
        kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu\n");return 1;}
        kk_tex *it=kk_tex_create(g,IW,IH,KK_FMT_RGBA8,KK_TEX_SAMPLE,in8);
        kk_tex *ot=kk_tex_create(g,IW,IH,KK_FMT_RGBA8,KK_TEX_STORAGE|KK_TEX_DOWNLOAD,NULL); // ReLU>=0, encode *0.5? Werte koennen >1
        // Conv-Outputs koennen >1 -> RGBA8 clippt. Fuer Vergleich: *0.25-Encode in MSL+CPU.
        kk_gpu_compute(g,CONV_MSL,"conv",&(kk_compute_args){.out=ot,.in={it},.n_in=1,.uniforms=&wb,.uniforms_size=sizeof wb});
        kk_gpu_finish(g);
        unsigned char *out=malloc(IW*IH*4); kk_tex_download(g,ot,out);
        // CPU-Ref (in8/255 als Input, gleiche Weights). Output 2×2-Packing.
        int sub[4][2]={{0,0},{1,0},{0,1},{1,1}};
        int maxe=0; double sume=0; int n=0;
        for(int gy=0;gy<LH;gy++)for(int gx=0;gx<LW;gx++){
            float r[4][4]; for(int k=0;k<4;k++)for(int c=0;c<4;c++) r[k][c]=wb.bias[k*4+c];
            for(int C=0;C<4;C++)for(int X=0;X<3;X++)for(int Y=0;Y<3;Y++){
                int ix=(gx+X-1)*2+sub[C][0], iy=(gy+Y-1)*2+sub[C][1];
                ix=ix<0?0:ix>IW-1?IW-1:ix; iy=iy<0?0:iy>IH-1?IH-1:iy;
                float v[4]; for(int c=0;c<4;c++) v[c]=in8[(iy*IW+ix)*4+c]/255.0f;
                int sp=Y*3+X;
                for(int k=0;k<4;k++){ int base=((k*4+C)*9+sp)*16;
                    for(int row=0;row<4;row++) for(int col=0;col<4;col++) r[k][row]+=wb.w[base+col*4+row]*v[col]; } }
            for(int k=0;k<4;k++){ int ox=gx*2+sub[k][0], oy=gy*2+sub[k][1];
                for(int c=0;c<4;c++){ float val=r[k][c]<0?0:r[k][c]; int ref=(int)(val*255+0.5f); ref=ref>255?255:ref;
                    int got=out[(oy*IW+ox)*4+c]; int e=abs(ref-got); if(e>maxe)maxe=e; sume+=e; n++; } } }
        printf("ArtCNN-Pass2 Multi-Kanal-Conv (16->16 mat4) vs CPU-Ref: maxerr=%d mean=%.2f LSB %s\n",
               maxe,sume/n, maxe<=2?"PASS":"FAIL");
        kk_gpu_destroy(&g);
        return maxe<=2?0:1;
    }
}

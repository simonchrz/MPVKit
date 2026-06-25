// kk_iosurface_test.m — Prod-Output-Pfad-Foundation: kk_gpu rendert zero-copy in eine
// IOSurface (= das Muster der Display-IOSurface der AVSampleBufferDisplayLayer).
// Erstellt eine RGBA8-IOSurface, wrappt sie WRITABLE (KK_TEX_STORAGE), schreibt ein
// bekanntes Muster per MSL, liest die IOSurface CPU-seitig zurück + verifiziert.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import "kk_gpu.h"
#define W 32
#define H 32

static const char *FILL_MSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void fill(texture2d<float,access::write> dst [[texture(0)]], uint2 id [[thread_position_in_grid]]){\n"
"  uint w=dst.get_width(),h=dst.get_height(); if(id.x>=w||id.y>=h)return;\n"
"  dst.write(float4(float(id.x)/float(w-1), float(id.y)/float(h-1), 0.5, 1.0), id);}\n";

int main(void){
    @autoreleasepool {
        // RGBA8-IOSurface (wie die Display-Surface).
        NSDictionary *props=@{ (id)kIOSurfaceWidth:@(W), (id)kIOSurfaceHeight:@(H),
            (id)kIOSurfaceBytesPerElement:@(4), (id)kIOSurfacePixelFormat:@((unsigned)'BGRA') };
        IOSurfaceRef surf=IOSurfaceCreate((CFDictionaryRef)props);
        if(!surf){fprintf(stderr,"IOSurfaceCreate fail\n");return 1;}

        kk_gpu *g=kk_gpu_create(NULL); if(!g){fprintf(stderr,"gpu fail\n");return 1;}
        // Output = die IOSurface, writable gewrappt (zero-copy).
        kk_tex *out=kk_tex_wrap_iosurface(g,(void*)surf,0,KK_FMT_BGRA8,KK_TEX_STORAGE);
        if(!out){fprintf(stderr,"wrap fail\n");return 1;}
        kk_gpu_compute(g,FILL_MSL,"fill",&(kk_compute_args){.out=out,.n_in=0});
        kk_gpu_finish(g);

        // IOSurface CPU-seitig zurücklesen.
        IOSurfaceLock(surf,kIOSurfaceLockReadOnly,NULL);
        unsigned char *base=IOSurfaceGetBaseAddress(surf);
        size_t bpr=IOSurfaceGetBytesPerRow(surf);
        int fails=0; // erwartet: BGRA, R~x-Rampe, G~y-Rampe, B~128
        for(int y=0;y<H;y++)for(int x=0;x<W;x++){ unsigned char *px=base+y*bpr+x*4;
            int B=px[0],Gc=px[1],R=px[2]; // BGRA
            int eR=R-(x*255)/(W-1), eG=Gc-(y*255)/(H-1), eB=B-128;
            if(abs(eR)>2||abs(eG)>2||abs(eB)>2) fails++; }
        unsigned char *p0=base; printf("IOSurface[0,0] BGRA=%d,%d,%d,%d  fails=%d  %s\n",
            p0[0],p0[1],p0[2],p0[3], fails, fails?"FAIL":"PASS (kk_gpu schreibt zero-copy in IOSurface)");
        IOSurfaceUnlock(surf,kIOSurfaceLockReadOnly,NULL);
        kk_gpu_destroy(&g);
        return fails?1:0;
    }
}

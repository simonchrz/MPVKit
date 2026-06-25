// kk_gpu.m — Metal-Impl der kk_gpu-Abstraktion (libplacebo-Ablösung, Etappe 1).
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#include "kk_gpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct kk_tex {
    id<MTLTexture> tex;
    int w, h;
    bool downloadable;
};

struct kk_gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> cb;        // aktueller (lazy), gebatcht bis finish
    id<MTLComputeCommandEncoder> enc;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *psoCache;
    id<MTLSamplerState> sampNearest, sampLinear;
};

static MTLPixelFormat mtl_fmt(kk_fmt f) {
    switch (f) {
        case KK_FMT_R8:      return MTLPixelFormatR8Unorm;
        case KK_FMT_RG8:     return MTLPixelFormatRG8Unorm;
        case KK_FMT_R16:     return MTLPixelFormatR16Unorm;
        case KK_FMT_RG16:    return MTLPixelFormatRG16Unorm;
        case KK_FMT_RGBA8:   return MTLPixelFormatRGBA8Unorm;
        case KK_FMT_BGRA8:   return MTLPixelFormatBGRA8Unorm;
        case KK_FMT_RGBA16F: return MTLPixelFormatRGBA16Float;
    }
    return MTLPixelFormatRGBA8Unorm;
}
static int fmt_bytes(kk_fmt f) {
    switch (f) {
        case KK_FMT_R8:      return 1;
        case KK_FMT_RG8:     return 2;
        case KK_FMT_R16:     return 2;
        case KK_FMT_RG16:    return 4;
        case KK_FMT_RGBA8:   return 4;
        case KK_FMT_BGRA8:   return 4;
        case KK_FMT_RGBA16F: return 8;
    }
    return 4;
}

kk_gpu *kk_gpu_create(void *mtl_device) {
    @autoreleasepool {
        struct kk_gpu *g = calloc(1, sizeof(*g));
        if (!g) return NULL;
        g->dev = mtl_device ? (__bridge id<MTLDevice>) mtl_device
                            : MTLCreateSystemDefaultDevice();
        if (!g->dev) { free(g); return NULL; }
        CFRetain((__bridge CFTypeRef) g->dev);
        g->queue = [g->dev newCommandQueue];
        CFRetain((__bridge CFTypeRef) g->queue);
        g->psoCache = [[NSMutableDictionary alloc] init];
        CFRetain((__bridge CFTypeRef) g->psoCache);
        MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
        sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        g->sampNearest = [g->dev newSamplerStateWithDescriptor:sd];
        CFRetain((__bridge CFTypeRef) g->sampNearest);
        sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
        g->sampLinear = [g->dev newSamplerStateWithDescriptor:sd];
        CFRetain((__bridge CFTypeRef) g->sampLinear);
        return g;
    }
}

void kk_gpu_destroy(kk_gpu **pgpu) {
    if (!pgpu || !*pgpu) return;
    struct kk_gpu *g = *pgpu;
    kk_gpu_finish(g);
    if (g->sampLinear)  CFRelease((__bridge CFTypeRef) g->sampLinear);
    if (g->sampNearest) CFRelease((__bridge CFTypeRef) g->sampNearest);
    if (g->psoCache)    CFRelease((__bridge CFTypeRef) g->psoCache);
    if (g->queue)       CFRelease((__bridge CFTypeRef) g->queue);
    if (g->dev)         CFRelease((__bridge CFTypeRef) g->dev);
    free(g);
    *pgpu = NULL;
}

void *kk_gpu_mtl_device(kk_gpu *g) { return (__bridge void *) g->dev; }

// Encoder/CB lazy holen (Batching: alle Passes in einen CB bis finish).
static id<MTLComputeCommandEncoder> get_enc(struct kk_gpu *g) {
    if (!g->cb) {
        g->cb = [g->queue commandBuffer];
        CFRetain((__bridge CFTypeRef) g->cb);
    }
    if (!g->enc) {
        g->enc = [g->cb computeCommandEncoder];
        CFRetain((__bridge CFTypeRef) g->enc);
    }
    return g->enc;
}

void kk_gpu_finish(kk_gpu *g) {
    @autoreleasepool {
        if (g->enc) { [g->enc endEncoding]; CFRelease((__bridge CFTypeRef) g->enc); g->enc = nil; }
        if (g->cb)  { [g->cb commit]; [g->cb waitUntilCompleted];
                      CFRelease((__bridge CFTypeRef) g->cb); g->cb = nil; }
    }
}

// Texture->Texture-Copy (z.B. eigener Render-Output -> Display-Target ohne ShaderWrite).
void kk_gpu_blit(kk_gpu *g, kk_tex *src, kk_tex *dst) {
    @autoreleasepool {
        if (!g->cb) { g->cb = [g->queue commandBuffer]; CFRetain((__bridge CFTypeRef) g->cb); }
        if (g->enc) { [g->enc endEncoding]; CFRelease((__bridge CFTypeRef) g->enc); g->enc = nil; }
        id<MTLBlitCommandEncoder> be = [g->cb blitCommandEncoder];
        int w = src->w < dst->w ? src->w : dst->w, h = src->h < dst->h ? src->h : dst->h;
        [be copyFromTexture:src->tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                 sourceSize:MTLSizeMake(w,h,1) toTexture:dst->tex destinationSlice:0
           destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
        [be endEncoding];
    }
}

kk_tex *kk_tex_create(kk_gpu *g, int w, int h, kk_fmt fmt, uint32_t usage,
                      const void *initial_data) {
    @autoreleasepool {
        struct kk_tex *t = calloc(1, sizeof(*t));
        if (!t) return NULL;
        t->w = w; t->h = h;
        MTLTextureDescriptor *d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:mtl_fmt(fmt) width:w height:h mipmapped:NO];
        MTLTextureUsage u = 0;
        if (usage & KK_TEX_SAMPLE)  u |= MTLTextureUsageShaderRead;
        if (usage & KK_TEX_STORAGE) u |= MTLTextureUsageShaderWrite;
        if (usage & KK_TEX_RENDER)  u |= MTLTextureUsageRenderTarget;
        d.usage = u;
        t->downloadable = (usage & KK_TEX_DOWNLOAD) != 0;
        d.storageMode = t->downloadable ? MTLStorageModeShared : MTLStorageModePrivate;
        t->tex = [g->dev newTextureWithDescriptor:d];
        if (!t->tex) { free(t); return NULL; }
        CFRetain((__bridge CFTypeRef) t->tex);
        if (initial_data) {
            [t->tex replaceRegion:MTLRegionMake2D(0, 0, w, h) mipmapLevel:0
                        withBytes:initial_data bytesPerRow:(NSUInteger) w * fmt_bytes(fmt)];
        }
        return t;
    }
}

kk_tex *kk_tex_create_3d(kk_gpu *g, int w, int h, int d, kk_fmt fmt,
                         const void *initial_data) {
    @autoreleasepool {
        struct kk_tex *t = calloc(1, sizeof(*t));
        if (!t) return NULL;
        t->w = w; t->h = h;
        MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
        desc.textureType = MTLTextureType3D;
        desc.pixelFormat = mtl_fmt(fmt);
        desc.width = w; desc.height = h; desc.depth = d;
        desc.usage = MTLTextureUsageShaderRead;
        desc.storageMode = MTLStorageModeShared;
        t->tex = [g->dev newTextureWithDescriptor:desc];
        if (!t->tex) { free(t); return NULL; }
        CFRetain((__bridge CFTypeRef) t->tex);
        if (initial_data) {
            NSUInteger bpr = (NSUInteger) w * fmt_bytes(fmt);
            [t->tex replaceRegion:MTLRegionMake3D(0, 0, 0, w, h, d) mipmapLevel:0 slice:0
                        withBytes:initial_data bytesPerRow:bpr bytesPerImage:bpr * h];
        }
        return t;
    }
}

kk_tex *kk_tex_wrap_mtltexture(kk_gpu *g, void *mtltexture) {
    @autoreleasepool {
        id<MTLTexture> mt = (__bridge id<MTLTexture>) mtltexture;
        if (!mt) return NULL;
        struct kk_tex *t = calloc(1, sizeof(*t));
        if (!t) return NULL;
        t->w = (int) mt.width; t->h = (int) mt.height;
        t->tex = mt;                       // existierende Textur (geteiltes Device, z.B. Display-Target)
        CFRetain((__bridge CFTypeRef) t->tex);
        return t;                          // Usage-Flags kommen von der Quell-Textur
    }
}

kk_tex *kk_tex_wrap_iosurface(kk_gpu *g, void *iosurface, int plane, kk_fmt fmt,
                              uint32_t usage) {
    @autoreleasepool {
        IOSurfaceRef surf = (IOSurfaceRef) iosurface;
        int w = (int) IOSurfaceGetWidthOfPlane(surf, plane);
        int h = (int) IOSurfaceGetHeightOfPlane(surf, plane);
        struct kk_tex *t = calloc(1, sizeof(*t));
        if (!t) return NULL;
        t->w = w; t->h = h;
        MTLTextureDescriptor *d = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:mtl_fmt(fmt) width:w height:h mipmapped:NO];
        MTLTextureUsage u = 0;
        if (usage & KK_TEX_SAMPLE)  u |= MTLTextureUsageShaderRead;
        if (usage & KK_TEX_STORAGE) u |= MTLTextureUsageShaderWrite;   // Output: in Display-IOSurface schreiben
        if (usage & KK_TEX_RENDER)  u |= MTLTextureUsageRenderTarget;
        d.usage = u ? u : MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModeShared;
        t->tex = [g->dev newTextureWithDescriptor:d iosurface:surf plane:plane];
        if (!t->tex) { free(t); return NULL; }
        CFRetain((__bridge CFTypeRef) t->tex);
        return t;
    }
}

void kk_tex_destroy(kk_gpu *g, kk_tex **ptex) {
    if (!ptex || !*ptex) return;
    struct kk_tex *t = *ptex;
    if (t->tex) CFRelease((__bridge CFTypeRef) t->tex);
    free(t);
    *ptex = NULL;
}
int kk_tex_w(const kk_tex *t) { return t->w; }
int kk_tex_h(const kk_tex *t) { return t->h; }
bool kk_tex_can_write(const kk_tex *t) {
    return t && t->tex && (t->tex.usage & MTLTextureUsageShaderWrite) != 0;
}

bool kk_tex_download(kk_gpu *g, kk_tex *t, void *dst) {
    if (!t->downloadable) return false;
    NSUInteger bpr = (NSUInteger) t->w * 4; // Annahme: Download-Targets sind RGBA8
    [t->tex getBytes:dst bytesPerRow:bpr fromRegion:MTLRegionMake2D(0, 0, t->w, t->h) mipmapLevel:0];
    return true;
}

// MSL -> MTLComputePipelineState, gecacht nach (source,entry).
static id<MTLComputePipelineState> get_pso(struct kk_gpu *g, const char *src, const char *entry) {
    NSString *key = [NSString stringWithFormat:@"%s|%s", entry, src];
    id<MTLComputePipelineState> pso = g->psoCache[key];
    if (pso) return pso;
    NSError *err = nil;
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                                              options:nil error:&err];
    if (!lib) { fprintf(stderr, "[kk_gpu] MSL compile failed: %s\n",
                        err.localizedDescription.UTF8String); return nil; }
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
    if (!fn) { fprintf(stderr, "[kk_gpu] entry '%s' not found\n", entry); return nil; }
    pso = [g->dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "[kk_gpu] PSO failed: %s\n", err.localizedDescription.UTF8String); return nil; }
    g->psoCache[key] = pso;
    return pso;
}

bool kk_gpu_compute(kk_gpu *g, const char *msl_source, const char *entry,
                    const kk_compute_args *a) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = get_pso(g, msl_source, entry);
        if (!pso || !a->out) return false;
        id<MTLComputeCommandEncoder> enc = get_enc(g);
        [enc setComputePipelineState:pso];
        // Eingänge: texture(0..n_in-1) + sampler(0)=nearest, sampler(1)=linear.
        bool anyLinear = false;
        for (int i = 0; i < a->n_in; i++) {
            [enc setTexture:a->in[i]->tex atIndex:i];
            if (a->linear[i]) anyLinear = true;
        }
        [enc setTexture:a->out->tex atIndex:a->n_in];   // Output (storage) hinter den Eingängen
        [enc setSamplerState:g->sampNearest atIndex:0];
        [enc setSamplerState:g->sampLinear atIndex:1];
        (void) anyLinear;
        if (a->uniforms && a->uniforms_size)
            [enc setBytes:a->uniforms length:a->uniforms_size atIndex:0];
        MTLSize grid = MTLSizeMake(a->out->w, a->out->h, 1);
        NSUInteger tgw = 16, tgh = 16;
        [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tgw, tgh, 1)];
        return true;
    }
}

#ifdef KK_GPU_TEST
// Selbsttest: ein Compute-Kernel skaliert R8-Input *0.5 in RGBA8-Output -> Download
// + Pixel-Check. Beweist: Device, Texturen, MSL-Compile/PSO, Dispatch, Download.
static const char *TEST_MSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void test_copy(texture2d<float> src [[texture(0)]],\n"
"                      texture2d<float, access::write> dst [[texture(1)]],\n"
"                      sampler near [[sampler(0)]],\n"
"                      uint2 id [[thread_position_in_grid]]) {\n"
"  if (id.x >= dst.get_width() || id.y >= dst.get_height()) return;\n"
"  float v = src.read(id).r * 0.5;\n"
"  dst.write(float4(v, v, v, 1.0), id);\n"
"}\n";

int main(void) {
    kk_gpu *g = kk_gpu_create(NULL);
    if (!g) { fprintf(stderr, "create failed\n"); return 1; }
    const int W = 64, H = 64;
    unsigned char *in = malloc(W * H);
    for (int i = 0; i < W * H; i++) in[i] = 200; // 200/255
    kk_tex *src = kk_tex_create(g, W, H, KK_FMT_R8, KK_TEX_SAMPLE, in);
    kk_tex *dst = kk_tex_create(g, W, H, KK_FMT_RGBA8, KK_TEX_STORAGE | KK_TEX_DOWNLOAD, NULL);
    if (!src || !dst) { fprintf(stderr, "tex failed\n"); return 1; }
    kk_compute_args a = { .out = dst, .in = { src }, .n_in = 1 };
    if (!kk_gpu_compute(g, TEST_MSL, "test_copy", &a)) { fprintf(stderr, "compute failed\n"); return 1; }
    kk_gpu_finish(g);
    unsigned char *out = malloc(W * H * 4);
    if (!kk_tex_download(g, dst, out)) { fprintf(stderr, "download failed\n"); return 1; }
    int got = out[0], want = 100; // 200*0.5
    printf("kk_gpu self-test: in=200 *0.5 -> out=%d (want~%d) %s\n",
           got, want, (got >= want - 1 && got <= want + 1) ? "PASS" : "FAIL");
    free(in); free(out);
    kk_tex_destroy(g, &src); kk_tex_destroy(g, &dst);
    kk_gpu_destroy(&g);
    return 0;
}
#endif

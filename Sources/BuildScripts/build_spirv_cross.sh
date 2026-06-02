#!/bin/bash
# Build SPIRV-Cross (Khronos) as static libs for all MPVKit-supported
# Apple platforms / archs. Places the result in dist/libspirv-cross/<platform>/thin/<arch>/
# so MPVKit's pkgConfigPath() helper picks it up automatically when libmpv configures.
#
# Triggered manually (no companion mpvkit/libspirv-cross-build repo exists);
# called once before `make build platform=ios` until upstream MPVKit ships an
# official spirv-cross prebuilt.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIST="$ROOT/dist/libspirv-cross"

# Source checkout. Validate CMakeLists.txt — a bare `[ -d ]` check passes for a
# half-finished / interrupted clone (dir exists but empty), which then fails the
# whole build at the cmake step. Re-clone if the checkout is missing OR broken.
SRC=/tmp/SPIRV-Cross-src
if [ ! -f "$SRC/CMakeLists.txt" ]; then
    rm -rf "$SRC"
    git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Cross.git "$SRC"
fi

# (platform, arch, sdk, target_triple_args, deployment_target)
build_one() {
    local PLATFORM=$1 ARCH=$2 SDK=$3 TARGET=$4 MIN=$5
    local SDKPATH=$(xcrun --sdk "$SDK" --show-sdk-path)
    local OUT="$DIST/$PLATFORM/thin/$ARCH"
    local BUILD="/tmp/spirv-cross-build-$PLATFORM-$ARCH"
    echo ">>> Building SPIRV-Cross for $PLATFORM/$ARCH"
    rm -rf "$BUILD" "$OUT"
    mkdir -p "$BUILD"
    cd "$BUILD"
    cmake -G "Unix Makefiles" \
        -DCMAKE_INSTALL_PREFIX="$OUT" \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$SDKPATH" \
        -DCMAKE_OSX_ARCHITECTURES="$ARCH" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET="$MIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPIRV_CROSS_SHARED=OFF \
        -DSPIRV_CROSS_STATIC=ON \
        -DSPIRV_CROSS_CLI=OFF \
        -DSPIRV_CROSS_ENABLE_TESTS=OFF \
        -DSPIRV_CROSS_ENABLE_GLSL=ON \
        -DSPIRV_CROSS_ENABLE_HLSL=OFF \
        -DSPIRV_CROSS_ENABLE_MSL=ON \
        -DSPIRV_CROSS_ENABLE_CPP=OFF \
        -DSPIRV_CROSS_ENABLE_REFLECT=ON \
        -DSPIRV_CROSS_ENABLE_C_API=ON \
        -DSPIRV_CROSS_ENABLE_UTIL=ON \
        -DSPIRV_CROSS_FORCE_PIC=ON \
        -DCMAKE_C_FLAGS="-isysroot $SDKPATH -arch $ARCH $TARGET" \
        -DCMAKE_CXX_FLAGS="-isysroot $SDKPATH -arch $ARCH $TARGET -stdlib=libc++" \
        "$SRC" > /dev/null
    make -j"$(sysctl -n hw.ncpu)" > /dev/null
    make install > /dev/null

    # mpv's meson looks for spirv-cross-c-shared.pc via pkg-config — write one.
    mkdir -p "$OUT/lib/pkgconfig"
    cat > "$OUT/lib/pkgconfig/spirv-cross-c-shared.pc" <<EOF
prefix=$OUT
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: spirv-cross-c
Description: C API for SPIRV-Cross (built from source by MPVKit-fork)
Version: 0.66.0
Libs: -L\${libdir} -lspirv-cross-c -lspirv-cross-glsl -lspirv-cross-msl -lspirv-cross-reflect -lspirv-cross-core -lspirv-cross-util -lc++
Cflags: -I\${includedir}
EOF
}

# iOS device
build_one ios            arm64  iphoneos        "-mios-version-min=14.0"                   14.0
# iOS simulator
build_one isimulator     arm64  iphonesimulator "-mios-simulator-version-min=14.0"         14.0
build_one isimulator     x86_64 iphonesimulator "-mios-simulator-version-min=14.0"         14.0

echo ">>> SPIRV-Cross built for all iOS platforms."
echo ">>> Install dirs:"
ls "$DIST"/*/thin/*/lib/libspirv-cross-c.a 2>/dev/null

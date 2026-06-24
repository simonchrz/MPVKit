#!/usr/bin/env bash
# Baut das standalone hybrid_render.c (kuckuck_hybrid_*) zu Libkkrender.xcframework
# (Device arm64 + Simulator arm64). Referenziert libplacebo-Symbole undefined →
# beim App-Link von Libplacebo.xcframework aufgelöst (wie Libcurl↔FFmpeg).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST="$(cd "$ROOT/.." && pwd)/dist"
OUT="$ROOT/out"; rm -rf "$OUT"; mkdir -p "$OUT/dev" "$OUT/sim"
CLANG="$(xcrun -f clang)"; LIBTOOL="$(xcrun -f libtool)"

build_slice() {  # build_slice <sdk> <min-flag> <plinc> <spvcdir> <outdir>
  local sdk="$1" min="$2" plinc="$3" spvc="$4" od="$5"
  local sysroot; sysroot="$(xcrun --sdk "$sdk" --show-sdk-path)"
  for src in hybrid_render kk_renderer; do
    "$CLANG" -c "$ROOT/$src.c" -o "$od/$src.o" \
      -arch arm64 -isysroot "$sysroot" "$min" ${KK_AOT:+-DKK_AOT} \
      -I"$plinc" -I"$ROOT" -Wall -O2 -fno-common
  done
  # hybrid_render.o + kk_renderer.o + spirv-cross (libplacebo-Metal braucht spvc_*
  # zur Laufzeit; früher in libmpv.a gemergt → jetzt hier, da libmpv wegfällt).
  # KK_AOT=1: kein Runtime-Compiler -> spirv-cross NICHT mergen (~Größen-Schnitt).
  if [ -n "${KK_AOT:-}" ]; then
    "$LIBTOOL" -static -o "$od/libkkrender.a" \
      "$od/hybrid_render.o" "$od/kk_renderer.o"
  else
    "$LIBTOOL" -static -o "$od/libkkrender.a" \
      "$od/hybrid_render.o" "$od/kk_renderer.o" "$spvc"/*.a
  fi
  echo "built $od/libkkrender.a (hybrid=$(nm "$od/libkkrender.a" 2>/dev/null | grep -c ' T _kuckuck_hybrid') spvc=$(nm "$od/libkkrender.a" 2>/dev/null | grep -c ' T _spvc_'))"
}

build_slice iphoneos        -miphoneos-version-min=14.0 \
  "$DIST/libplacebo/ios/thin/arm64/include"        "$DIST/libspirv-cross/ios/thin/arm64/lib"        "$OUT/dev"
build_slice iphonesimulator -mios-simulator-version-min=14.0 \
  "$DIST/libplacebo/isimulator/thin/arm64/include" "$DIST/libspirv-cross/isimulator/thin/arm64/lib" "$OUT/sim"

rm -rf "$OUT/Libkkrender.xcframework"
xcodebuild -create-xcframework \
  -library "$OUT/dev/libkkrender.a" -headers "$ROOT/include" \
  -library "$OUT/sim/libkkrender.a" -headers "$ROOT/include" \
  -output "$OUT/Libkkrender.xcframework"
echo "==> Libkkrender.xcframework gebaut:"; find "$OUT/Libkkrender.xcframework" -maxdepth 2

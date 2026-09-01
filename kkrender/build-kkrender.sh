#!/usr/bin/env bash
# Baut den standalone kk_gpu-Renderer (kuckuck_hybrid_* + kk_gpu_*) zu
# Libkkrender.xcframework (Device arm64 + Simulator arm64). KEINE libplacebo-/spirv-cross-
# Abhängigkeit mehr (libplacebo-Drop 2026-06-25): kk_gpu nutzt Metal direkt
# (newLibraryWithSource), die Color-Mathe ist nativ (kk_gpu_genparams). Self-contained.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/out"; rm -rf "$OUT"; mkdir -p "$OUT/dev" "$OUT/sim"
CLANG="$(xcrun -f clang)"; LIBTOOL="$(xcrun -f libtool)"

build_slice() {  # build_slice <sdk> <min-flag> <outdir>
  local sdk="$1" min="$2" od="$3"
  local sysroot; sysroot="$(xcrun --sdk "$sdk" --show-sdk-path)"
  for src in hybrid_render kk_gpu_render kk_gpu_cnn kk_gpu_genparams; do
    "$CLANG" -c "$ROOT/$src.c" -o "$od/$src.o" \
      -arch arm64 -isysroot "$sysroot" "$min" -I"$ROOT" -Wall -O2 -fno-common
  done
  # kk_gpu.m = eigene Metal-Foundation (ObjC, ARC).
  "$CLANG" -c "$ROOT/kk_gpu.m" -o "$od/kk_gpu.o" \
    -arch arm64 -isysroot "$sysroot" "$min" -fobjc-arc -I"$ROOT" -Wall -O2 -fno-common
  "$LIBTOOL" -static -o "$od/libkkrender.a" \
    "$od/hybrid_render.o" "$od/kk_gpu_render.o" "$od/kk_gpu_cnn.o" "$od/kk_gpu_genparams.o" "$od/kk_gpu.o"
  echo "built $od/libkkrender.a (hybrid=$(nm "$od/libkkrender.a" 2>/dev/null | grep -c ' T _kuckuck_hybrid') pl-undef=$(nm "$od/libkkrender.a" 2>/dev/null | grep -c ' U _pl_'))"
}

build_slice iphoneos        -miphoneos-version-min=14.0 "$OUT/dev"
build_slice iphonesimulator -mios-simulator-version-min=14.0 "$OUT/sim"

rm -rf "$OUT/Libkkrender.xcframework"
xcodebuild -create-xcframework \
  -library "$OUT/dev/libkkrender.a" -headers "$ROOT/include" \
  -library "$OUT/sim/libkkrender.a" -headers "$ROOT/include" \
  -output "$OUT/Libkkrender.xcframework"
echo "==> Libkkrender.xcframework gebaut:"; find "$OUT/Libkkrender.xcframework" -maxdepth 2

# Prüfstände auf dem Mac laufen lassen (macOS-SDK, echte GPU, CPU-Referenzen).
# ⚠️ Bewusst NACH dem Bau und im selben Skript: bis 2026-09-01 lagen die Prüfstände
# daneben und mussten von Hand übersetzt werden — entsprechend lief sie niemand,
# und `kk_scale_test` blieb monatelang grün, während es eine im Juli abgelöste
# sin()-Variante prüfte statt des ausgelieferten Kernels.
# Überspringen mit KK_SKIP_TESTS=1 (z.B. auf einem Rechner ohne Metal).
if [ "${KK_SKIP_TESTS:-0}" != "1" ]; then
  echo "==> Prüfstände:"
  "$ROOT/run-tests.sh"
fi

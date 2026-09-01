#!/usr/bin/env bash
# Baut und läuft die kk_gpu-Prüfstände auf dem MAC (macOS-SDK, echte Metal-GPU).
#
# ⚠️ Warum auf dem Mac und nicht auf dem Gerät: es geht um BILDMATHEMATIK, nicht um
# Gerätespezifisches. Jeder Test rechnet dieselbe Operation zusätzlich auf der CPU
# und vergleicht — eine Referenz, die auf jeder GPU gleich sein muss. Was NUR auf
# echter Hardware auffällt (Pacing, Thermik, Speicher), gehört nicht hierher,
# sondern in die on-device-Messung via `hybrid.log`.
#
# Aufruf: ./run-tests.sh [name ...]   (ohne Argument: alle)
# `build-kkrender.sh` ruft das am Ende selbst auf — ein Prüfstand, den niemand
# startet, schützt nichts.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP="${TMPDIR:-/tmp}/kkrender-tests"; mkdir -p "$TMP"
CLANG="$(xcrun -f clang)"
SDK="$(xcrun --sdk macosx --show-sdk-path)"

# Die Quellen, die JEDER Prüfstand braucht (Metal-Foundation + Farb-Mathe + CNN).
GEMEINSAM=("$ROOT/kk_gpu.m" "$ROOT/kk_gpu_genparams.c" "$ROOT/kk_gpu_cnn.c")
RAHMEN=(-framework Metal -framework Foundation -framework CoreVideo -framework IOSurface)

alle=(kk_tg_test kk_decode_test kk_iosurface_test kk_scale_test kk_cnn_test
      kk_fusion_test kk_lut_test kk_ewa_test kk_hdr_test kk_rest_test)
tests=("$@"); [ ${#tests[@]} -eq 0 ] && tests=("${alle[@]}")

fehler=0
for t in "${tests[@]}"; do
  if [ ! -f "$ROOT/$t.m" ]; then printf '%-18s ÜBERSPRUNGEN (fehlt)\n' "$t"; continue; fi
  # ⚠️ -Wall UND Warnungen anzeigen: ein weggefiltertes `warning:` hat beim Bau
  # dieser Prüfstände einen Doppelzeiger-Fehler verdeckt, der als Absturz endete.
  if ! "$CLANG" -fobjc-arc -O1 -Wall -isysroot "$SDK" -I"$ROOT" \
        "$ROOT/$t.m" "${GEMEINSAM[@]}" "${RAHMEN[@]}" -o "$TMP/$t" 2>"$TMP/$t.build"; then
    printf '%-18s BAUT NICHT\n' "$t"; grep -E "error:|warning:" "$TMP/$t.build" | sed 's/^/    /' | head -4
    fehler=$((fehler+1)); continue
  fi
  # ⚠️ `grep -c` gibt bei 0 Treffern die 0 aus UND meldet Exit 1 — mit `set -e`
  # und einer &&-Kette reisst das den ganzen Lauf ab. Darum if-Block + `|| true`.
  if grep -q "warning:" "$TMP/$t.build" 2>/dev/null; then
    printf '%-18s Warnungen beim Bau:\n' "$t"
    grep "warning:" "$TMP/$t.build" | sed 's/^/    /' | head -3
  fi
  if ausgabe="$("$TMP/$t" 2>&1)"; then
    printf '%-18s %s\n' "$t" "$(echo "$ausgabe" | tail -1)"
  else
    printf '%-18s FEHLGESCHLAGEN\n' "$t"; echo "$ausgabe" | sed 's/^/    /' | tail -5
    fehler=$((fehler+1))
  fi
done

if [ "$fehler" -gt 0 ]; then echo "==> $fehler Prüfstand/Prüfstände rot"; exit 1; fi
echo "==> alle Prüfstände grün"

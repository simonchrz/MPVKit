# Upstream-Prep: Metal-RA-Backend → mpv-master

Ziel: das `ra_metal`-Backend in **upstream-fähige, atomar-begründete Commits** zerlegen,
sauber **getrennt** von der Kuckuck-spezifischen Instrumentierung. Die Instrumentierung
bleibt als privater **Overlay-Patch** auf unserem Fork (load-bearing für unseren
Cold-Start, NICHT löschbar). Profitiert auch ohne PR: löst die dirty-tree-Wartungsschuld.

Branch: `upstream-prep` (off `ios-landscape-resize`).

## Boundary-Regel
- **KERN (upstreamable):** RA-Backend, in-memory `pipeline_cache` (cache_key-Compute +
  Lookup/Store), Blit-PSO, Textures, Renderpass, Init/Destroy, hwdec-vt, Vertex-Ring.
- **OVERLAY (Kuckuck, bleibt geforkt):** disk-`pso_sidecar` (Warmup-Persistenz),
  `pso_stats` (Hit-Rate-Counter), `cache_key`-**Logging** (`metal_log_cache_key`),
  `passgraph`-Stats, tex-upload-/shader-**Dumps**, + die `mpv_metal_*`-Custom-API.

Merksatz: **in-memory-Caching/Blit/RA = Kern; disk-Persistenz + Telemetrie + Diagnostik = Overlay.**

## Per-Datei-Einordnung (evidenzbasiert, dist/libmpv-master/video/out/metal/)

| Datei | Zeilen | Kern | Overlay zu extrahieren |
|---|---|---|---|
| `ra_metal_textures.m` | 1035 | fast ganz | nur tex-upload-Dump (Z.28, ios-metal.54) |
| `ra_metal_renderpass.m` | 1870 | Großteil + in-mem pipeline_cache | `metal_log_cache_key` (Z.854), pso_sidecar-Block (Z.1116-1138), `serialize_renderpass_params` (Z.203) |
| `ra_metal_init.m` | 755 | RA-Init/Destroy | PSO-Sidecar load/save + Warmup-Thread |
| `ra_metal.m` | 252 | wenig (RA-Glue) | Großteil: sidecar kick/save/flush + cache-key-Logging + passgraph-Stats |
| `ra_metal.h` | 213 | Header | evtl. overlay-decls |
| `include/mpv/render_mtl.h` | 189 | `mpv_metal_init_params`, `mpv_metal_drawable` | `mpv_metal_pso_stats(_get)`, `mpv_metal_pso_sidecar_flush`, `mpv_metal_passgraph_stats(_get)` |

## Ziel-Commit-Serie (Kern, atomar, mpv-Style — je mit what+why)
1. `render: add Metal render-param enum values` (= heutiges 0006, hat schon Subject ✓)
2. `vo_gpu/ra_metal: meson + backend registration` (0007 + 0009)
3. `vo_gpu/ra_metal: core RA backend skeleton` (0008 minus pso_sidecar/cache_key-Logging)
4. `vo_gpu/ra_metal: renderpass + in-memory pipeline cache` (renderpass-Kern)
5. `vo_gpu/ra_metal: texture upload + format-convert blit` (textures-Kern)
6. `vo_gpu/ra_metal: vertex ring allocator` (0010)
7. `vo_gpu/ra_metal: blit flip handling` (0011, hat Subject ✓)
8. `vo_gpu/ra_metal: VideoToolbox Metal hwdec` (0012)
9. iOS-surface-resize-Bits (0004/0005) — evtl. separater/optionaler PR-Teil
+ **Overlay-Patch** (bleibt im Fork, NICHT im PR): „kuckuck: PSO-sidecar warmup + stats + passgraph + cache-key logging" = alles aus der Overlay-Spalte oben, inkl. der `mpv_metal_*`-Custom-API.

## Offene Design-Fragen für den PR
- `mpv_metal_init_params`/`drawable`: passen die zu mpv's render-API-Konventionen, oder
  will Upstream das über `mpv_render_context_create` + params-array statt eigener structs?
- Braucht's einen `-Dmetal`-meson-Flag-Namen, der zu mpv's Konvention passt (`gl`/`vulkan`/`d3d11` → `metal`)? (0007 macht das schon — gegen Upstream-Naming prüfen.)
- Lizenz-Header in 0008 sagt „Copyright the mpv developers" — vor PR auf korrekte Attribution prüfen.

## Status / nächste Schritte
- [x] Separations-Landkarte (diese Datei) — evidenzbasiert verifiziert.
- [ ] **Session 2:** erste kohärente Extraktion — Overlay-Symbole + ihre decls + Custom-API
  zusammen rausziehen, sodass der Kern noch **baut** (nicht datei-partiell). Start: render_mtl.h-Split + ra_metal.m (am meisten Overlay) + die renderpass/init-Hooks. Danach `make build platform=ios` → muss grün sein (Kern ohne Overlay lauffähig), dann Overlay als separater Patch wieder drauf → A/B render-neutral verifizieren.
- [ ] Kern-Commit-Serie schreiben (Subjects + Bodies).
- [ ] Rebase auf frisches mpv-master (nicht den dirty-tree-Snapshot).
- [ ] Maintainer-Kontakt.

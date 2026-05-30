# Overlay-Extraktions-Spec (exakte Call-Sites)

Zeilen-Refs gegen den aktuellen `dist/libmpv-master/`-Tree (Stand upstream-prep).
„entfernen" = aus Kern raus, geht in den privaten Kuckuck-Overlay-Patch.
**Wichtig:** der in-memory `pipeline_cache` (cache_key-Compute + Lookup/Store in
renderpass.m ~843-853, 1112) BLEIBT Kern — nur Telemetrie/Persistenz/Diagnostik raus.

## Fertig (diese Session, in upstream-core/ gestaged)
- `include/mpv/render_mtl.h` — Kern (init_params + drawable). Overlay raus: pso_stats(_get), sidecar_flush, passgraph_stats(_get) (war Z.125-183).
- `video/out/metal/ra_metal.m` — Kern (Prototypes + ra_fns_metal-vtable). Overlay raus: PSO/PG-Counter-Globals + _get(), sidecar save/kick/flush/clear, cache-key-log (war Z.19-191).

## ⚠ NEUE BEFUNDE (S3-Grind) — renderpass.m ist VIEL schwerer als erst gemappt
- **4. Diagnostik-Schicht entdeckt: `metal_rp_logf` (ios-metal.52 „Runtime-instrumentation")**
  — Helper-Funktionen Z.104-170 (`metal_rp_log_file`, `metal_rp_logf`, `metal_rp_glsl_hash_buf`,
  `metal_rp_vartype_name`) + **31 Call-Sites** über die ganze Datei verstreut.
- Damit hat renderpass.m **~40 Entfernungs-Stellen** über 4 Layer (rp_logf + pso-stats +
  cache-key-log + passgraph + sidecar). Das ist der dominante Extraktions-Aufwand.
- **Lektion:** jeder Deep-Dive findet eine weitere Diagnostik-Schicht. Effort > erste Schätzung.

## ✅ FERTIG (gestaged in upstream-core/)
- render_mtl.h, ra_metal.m (S2) + **ra_metal_textures.m** (S3, 1035→889 Z., verifiziert
  null dangling refs: readback-diag un-weaved, logf-helpers + should_dump raus).

## ra_metal_renderpass.m — entfernen
- 86-91: extern-decls `mpv_metal_pso_lookups/hits/warmup_replays`, `mpv_metal_pg_total/raster/compute`
- 94, 97: extern-decls `metal_kick_pso_sidecar_save`, `metal_log_cache_key`
- 202: `metal_serialize_renderpass_params(...)` Definition (nur fürs Sidecar)
- 358: `atomic_fetch_add(&mpv_metal_pso_warmup_replays, 1)`
- 852: `atomic_fetch_add(&mpv_metal_pso_lookups, 1)`
- 854: `metal_log_cache_key(...)`-Call
- 885: `atomic_fetch_add(&mpv_metal_pso_hits, 1)`
- 1121-1138: pso_sidecar-Append-Block (`m->pso_sidecar`, serialize, `metal_kick_pso_sidecar_save`)
- 1379-1380, 1687-1688: passgraph-Counter (`pg_total` + `pg_raster`/`pg_compute`)
- **BLEIBT:** cache_key-Compute (843-850) + `pipeline_cache` Lookup/Store (853, 1112)

## ra_metal_init.m — entfernen
- 82-141: `metal_pso_sidecar_path`, `metal_load_pso_sidecar`, `metal_save_pso_sidecar`
- 143-162: `metal_kick_pso_warmup` + `metal_warmup_replay_entries`-extern
- 419-431: im Init: sidecar-load + `p->pso_sidecar=`, `warmup_group`-create, `metal_kick_pso_warmup`
- 625-655: `analyze_passgraph_and_reset` + pg-Counter + Call in commit_frame (655)
- 686-689: destroy — warmup_group-cleanup
- 717-731: destroy — `metal_clear_pso_sidecar_if` + `metal_save_pso_sidecar` + CFRelease(pso_sidecar)
- **BLEIBT:** RA-Init/Destroy-Kern (device/queue/formats/binary_archive ist separat zu prüfen — MTLBinaryArchive ist evtl. auch Overlay/Optional)

## ra_metal_textures.m — entfernen (⚠ STRUKTURELL VERWOBEN, kein Block-Delete)
- 28-50: tex-upload-Dump-Helper (`metal_tx_log_file` + `metal_tx_logf`)
- 293-295: `readback_n` + `use_separate_cb`-Decl (`rb_n<6 && dimensions==2 && ...`)
- **297-301: cb-SELECTION un-weaven** → `if(use_separate_cb){cb=[queue commandBuffer]}else{cb=ra_metal_cmd_buf(ra)}` ersetzen durch nur `id<MTLCommandBuffer> cb = ra_metal_cmd_buf(ra);` (else-Zweig = Kern-Pfad).
- 314-384: `if(use_separate_cb){...return true;}` Readback-Diag-Block (mit early-return!) löschen.
- 534-561: `should_dump`-Block + `call_count`-Decl in tex_upload.
- **BLEIBT:** tex_create/upload/download/buf_*/clear/blit + `metal_blit_pso`, der Upload-Blit (302-311) + Pool-Return-Handler (386-395) sind KERN.
- **Lektion:** die Diag ist NICHT additiv — sie hijackt die cb-Auswahl. Hand-Edit mit Flow-Verständnis nötig, kein sed/awk.

## ra_metal.h (struct ra_metal) — Overlay-Felder prüfen+extrahieren
- `pso_sidecar`, `warmup_group`, `warmup_cancel`, passgraph-pass-list → Overlay-Felder.
  (Kern-struct behält device/queue/cmd_buf/pipeline_cache/binary_archive/samplers/staging_pool.)

## Build-Reihenfolge (nächste Session)
1. ra_metal.h: Overlay-Felder in einen `#ifdef`-freien Kern-struct vs Overlay-Append trennen
   (oder: Felder bleiben im struct, nur die Nutzung raus — minimal-invasiver für den Build).
2. Die obigen Sites entfernen (alle zusammen, sonst undefined symbols).
3. `make build platform=ios` → muss grün sein (Kern ohne Overlay).
4. Overlay als EIN Patch zurück (= alle entfernten Hunks) → A/B render-neutral + PSO-Warmup
   wieder aktiv verifizieren (raw-TS-VOX-Auto-Launch).
5. Offene Frage: MTLBinaryArchive (PSO-Disk-Cache via Apple-API) — Kern oder Overlay?
   Wahrscheinlich Kern-optional (mpv-generisch nützlich), aber prüfen ob app-gekoppelt.

// swift-tools-version:5.9

import PackageDescription

let package = Package(
    name: "MPVKit",
    platforms: [.macOS(.v11), .iOS(.v14), .tvOS(.v14), .visionOS(.v1)],
    products: [
        .library(
            name: "MPVKit",
            targets: ["_MPVKit"]
        ),
        .library(
            name: "MPVKit-GPL",
            targets: ["_MPVKit-GPL"]
        ),
    ],
    targets: [
        .target(
            name: "_MPVKit",
            dependencies: [
                "Libmpv", "_FFmpeg", "Libuchardet", "Libbluray",
                .target(name: "Libluajit", condition: .when(platforms: [.macOS])),
            ],
            path: "Sources/_MPVKit",
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreAudio"),
            ]
        ),
        .target(
            name: "_FFmpeg",
            dependencies: [
                "Libavcodec", "Libavdevice", "Libavfilter", "Libavformat", "Libavutil", "Libswresample", "Libswscale",
                "Libssl", "Libcrypto", "Libass", "Libfreetype", "Libfribidi", "Libharfbuzz",
                "MoltenVK", "Libshaderc_combined", "lcms2", "Libplacebo", "Libdovi", "Libunibreak",
                "gmp", "nettle", "hogweed", "gnutls", "Libdav1d", "Libuavs3d",
                "Libngtcp2", "Libnghttp3", "Libngtcp2_crypto_gnutls", "Libcurl"   // HTTP/3 via libcurl (libcurl://)
            ],
            path: "Sources/_FFmpeg",
            linkerSettings: [
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("Metal"),
                .linkedFramework("VideoToolbox"),
                .linkedLibrary("bz2"),
                .linkedLibrary("iconv"),
                .linkedLibrary("expat"),
                .linkedLibrary("resolv"),
                .linkedLibrary("xml2"),
                .linkedLibrary("z"),
                .linkedLibrary("c++"),
            ]
        ),
        .target(
            name: "_MPVKit-GPL",
            dependencies: [
                "Libmpv-GPL", "_FFmpeg-GPL", "Libuchardet", "Libbluray",
                .target(name: "Libluajit", condition: .when(platforms: [.macOS])),
            ],
            path: "Sources/_MPVKit-GPL",
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreAudio"),
            ]
        ),
        .target(
            name: "_FFmpeg-GPL",
            dependencies: [
                "Libavcodec-GPL", "Libavdevice-GPL", "Libavfilter-GPL", "Libavformat-GPL", "Libavutil-GPL", "Libswresample-GPL", "Libswscale-GPL",
                "Libssl", "Libcrypto", "Libass", "Libfreetype", "Libfribidi", "Libharfbuzz",
                "MoltenVK", "Libshaderc_combined", "lcms2", "Libplacebo", "Libdovi", "Libunibreak",
                "Libsmbclient", "gmp", "nettle", "hogweed", "gnutls", "Libdav1d", "Libuavs3d",
                "Libngtcp2", "Libnghttp3", "Libngtcp2_crypto_gnutls", "Libcurl"   // HTTP/3 via libcurl (libcurl://)
            ],
            path: "Sources/_FFmpeg-GPL",
            linkerSettings: [
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("Metal"),
                .linkedFramework("VideoToolbox"),
                .linkedLibrary("bz2"),
                .linkedLibrary("iconv"),
                .linkedLibrary("expat"),
                .linkedLibrary("resolv"),
                .linkedLibrary("xml2"),
                .linkedLibrary("z"),
                .linkedLibrary("c++"),
            ]
        ),

        .binaryTarget(
            name: "Libmpv-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libmpv-GPL.xcframework.zip",
            checksum: "27519c054c4f73684ea50f5753c613d4151a56790c2e936ff39347be6d10ddb7"
        ),
        .binaryTarget(
            name: "Libavcodec-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavcodec-GPL.xcframework.zip",
            checksum: "9607ce804264794d626d86c3b8f03aeb76a96a28830cd405334e9d384f9ccf54"
        ),
        .binaryTarget(
            name: "Libavdevice-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavdevice-GPL.xcframework.zip",
            checksum: "c6ee8f0165a916ca73341513a2ea8170c6b5e67fa1b32450472b18e7c7d90bbe"
        ),
        .binaryTarget(
            name: "Libavformat-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavformat-GPL.xcframework.zip",
            checksum: "8aebce3ba30b282c6839f2368795ff5432e9824f1eca3c56a9bd68d38f2291f0"
        ),
        .binaryTarget(
            name: "Libavfilter-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavfilter-GPL.xcframework.zip",
            checksum: "b2ad6d7c11682c9452c1bf0f177f097362a47cb168a4a473d20b389860395833"
        ),
        .binaryTarget(
            name: "Libavutil-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavutil-GPL.xcframework.zip",
            checksum: "95f1c291e2bc5ec353fc2419cab71359bcf909a242e73eb9c5f64357f5f6c186"
        ),
        .binaryTarget(
            name: "Libswresample-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libswresample-GPL.xcframework.zip",
            checksum: "ac6184dc35e0cc31cfab86aaafc5763c35a9501929f136208ec8a05c6d2314cf"
        ),
        .binaryTarget(
            name: "Libswscale-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libswscale-GPL.xcframework.zip",
            checksum: "d13176eceb2e1aa47c703bd4b0bd549956d3543cbdcede1e96d3d1a27de671d7"
        ),
        //AUTO_GENERATE_TARGETS_BEGIN//

        .binaryTarget(
            name: "Libcrypto",
            url: "https://github.com/mpvkit/openssl-build/releases/download/3.3.5/Libcrypto.xcframework.zip",
            checksum: "593283be2a90f7fd66f6e6ed331b2f099cf403e0926fe3b4ac09a7062b793965"
        ),
        .binaryTarget(
            name: "Libssl",
            url: "https://github.com/mpvkit/openssl-build/releases/download/3.3.5/Libssl.xcframework.zip",
            checksum: "ff5ffd43d015d7285fd37e4a3145b25cbd8d2842740bd629a711c299a20e226a"
        ),

        .binaryTarget(
            name: "gmp",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/gmp.xcframework.zip",
            checksum: "ad33c7a08f4cdcb9924c8f0e6d9a054dad33d7794b97667bf8b6fb2b236ae585"
        ),

        .binaryTarget(
            name: "nettle",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/nettle.xcframework.zip",
            checksum: "0fdf3ebf8bd7b8bc8eee837cf27261cb4c52ae520b6576a2f468656aa1691e02"
        ),
        .binaryTarget(
            name: "hogweed",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/hogweed.xcframework.zip",
            checksum: "25727c9fa67287fa0a4f4722f88bb8be669b23cd7e837e2d00870eb8a25d3f27"
        ),

        .binaryTarget(
            name: "gnutls",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/gnutls.xcframework.zip",
            checksum: "3dbec5809339189bf9679e218c6cff387ebf8fb72745927835afc2678f5c9f4d"
        ),

        .binaryTarget(
            name: "Libunibreak",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libunibreak.xcframework.zip",
            checksum: "001087c0e927ae00f604422b539898b81eb77230ea7700597b70393cd51e946c"
        ),

        .binaryTarget(
            name: "Libfreetype",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libfreetype.xcframework.zip",
            checksum: "f2840aba1ce35e51c0595557eee82c908dac8e32108ecc0661301c06061e051c"
        ),

        .binaryTarget(
            name: "Libfribidi",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libfribidi.xcframework.zip",
            checksum: "4a55513792ef7a17893875f74cc84c56f3657e8768c07a7a96f563a11dc4b743"
        ),

        .binaryTarget(
            name: "Libharfbuzz",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libharfbuzz.xcframework.zip",
            checksum: "91558d8497d9d97bc11eeef8b744d104315893bfee8f17483d8002e14565f84b"
        ),

        .binaryTarget(
            name: "Libass",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libass.xcframework.zip",
            checksum: "1e41f5a69c74f6c6407aab84a65ccd0b34e73fa44465f488f99bf22bd61b070d"
        ),

        .binaryTarget(
            name: "Libsmbclient",
            url: "https://github.com/mpvkit/libsmbclient-build/releases/download/4.15.13-2512/Libsmbclient.xcframework.zip",
            checksum: "3a53375fab11bc888cc553664ea5dd902208d04f0cc21ec746302bf356246b6f"
        ),

        .binaryTarget(
            name: "Libbluray",
            url: "https://github.com/mpvkit/libbluray-build/releases/download/1.4.0/Libbluray.xcframework.zip",
            checksum: "bc037d34e2b0b5ab7f202fb371f5fb298136cc66fdf406c2172185d06f53f18d"
        ),

        .binaryTarget(
            name: "Libuavs3d",
            url: "https://github.com/mpvkit/libuavs3d-build/releases/download/1.2.1-xcode/Libuavs3d.xcframework.zip",
            checksum: "1e69250279be9334cd2f6849abdc884c8e4bb29212467b6f071fdc1ac2010b6b"
        ),

        .binaryTarget(
            name: "Libdovi",
            url: "https://github.com/mpvkit/libdovi-build/releases/download/3.3.2/Libdovi.xcframework.zip",
            checksum: "e693e239808350868e79c5448ef9f02e2716bc822dd8632a41a368a1eae5ca7d"
        ),

        .binaryTarget(
            name: "MoltenVK",
            url: "https://github.com/mpvkit/moltenvk-build/releases/download/1.4.1/MoltenVK.xcframework.zip",
            checksum: "9bd1ca1e4563bacd25d6e55d37b10341d50b2601bc2684bc332188e79daa2b79"
        ),

        .binaryTarget(
            name: "Libshaderc_combined",
            url: "https://github.com/mpvkit/libshaderc-build/releases/download/2025.5.0/Libshaderc_combined.xcframework.zip",
            checksum: "758047b615708575b580eb960a2d083f760a29dc462d6eaa360416c946ce433b"
        ),

        .binaryTarget(
            name: "lcms2",
            url: "https://github.com/mpvkit/lcms2-build/releases/download/2.17.0/lcms2.xcframework.zip",
            checksum: "dc0dce0606f6ab6841a8ec5a6bd4448e2f3ef00661a050460f806c9393dc6982"
        ),

        .binaryTarget(
            name: "Libplacebo",
            // Our Metal-backend fork build (kuckuck-prod-4: prod-3 rebased auf
            // videolan/master + MR !861 HDR-Peak-Fix + MR !859 film_grain; apiver
            // 365 unverändert). Ausgeliefert in renderpl.32 (mit dem libmpv-HDR-
            // Target-Fix). Siehe libplacebo-Fork Tag kuckuck-prod-4.
            // renderpl.40 = kuckuck-prod-6: prod-5 hatte die Picks VERLOREN
            // (von metal-pr-series getaggt) → !861 upstream-final (2d0979fb)
            // + !859-Re-Pick wiederhergestellt. apiver 365 unverändert.
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.40/Libplacebo.xcframework.zip",
            checksum: "91a5c712ffddedb124a63bc720dbabf3bcfd7907ea8bac4a54ebd76b38b9850f"
        ),

        .binaryTarget(
            name: "Libdav1d",
            url: "https://github.com/mpvkit/libdav1d-build/releases/download/1.5.2-xcode/Libdav1d.xcframework.zip",
            checksum: "8a8b78e23e28ecc213232805f3c1936141fc9befe113e87234f4f897f430a532"
        ),

        .binaryTarget(
            name: "Libavcodec",
            // renderpl.43: identisch .42 (FFmpeg 625ab01) MINUS dem FFmpeg-Patch
            // 0001-vt-inline-retry — die VT-Session-Recovery wandert von hier in
            // libmpv (vd_lavc-Patch 0017, s. Libmpv unten). Sonst kein Source-Δ.
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libavcodec.xcframework.zip",
            checksum: "da74d8697e6a30997b36636065985a3dd25f3ba776e7dd7120634a1121907cc8"
        ),
        .binaryTarget(
            name: "Libavdevice",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libavdevice.xcframework.zip",
            checksum: "55082c1b822557d163df80a05ed3edd5b78a4aca4f2e7297c27029c222fcfe68"
        ),
        .binaryTarget(
            name: "Libavformat",
            // renderpl.42 base + HTTP/3 (ff_http3_protocol). Other libav* stay on their pins (same 625ab01 ffmpeg).
            // h3.4: QUIC flow-control fix (4bb49ba) — recv_data extends max_stream/conn offset
            // so DATA-frame payload is credited; without it multi-segment HLS (recordings) stalled
            // after segment 1 (window never reopened past 1MB/8MB).
            // h3.5: ca_file via env KUCKUCK_H3_CA_FILE — lavf-o-Plumbing erreichte die private
            // AVOption nicht on-device → SecTrust fiel auf System-Store → Gateway-Playback
            // scheiterte auf Geräten ohne Caddy-Profil (alle iPads). Patch-Commit 157f738.
            // renderpl.54: libcurl-Pivot — hls.c open_url()-Gate lässt jetzt das `libcurl`-
            // Segment-Protokoll durch (sonst AVERROR_INVALIDDATA vor url_open → Scrub/Seek
            // kaputt, sequenziell lief nur über den Playlist-Connection-Reuse). pi-infra-Patch.
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.54/Libavformat.xcframework.zip",
            checksum: "3518440e896ea9d73243113d384246b66f394bbd6f547babd0ca17e545ed69a2"
        ),
        .binaryTarget(
            name: "Libngtcp2",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.42-h3/Libngtcp2.xcframework.zip",
            checksum: "d7fdd03870d05587a7eef225730a5b626654dc1bf85ae412b78e4a57b6d12baf"
        ),
        .binaryTarget(
            name: "Libnghttp3",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.42-h3/Libnghttp3.xcframework.zip",
            checksum: "f6ccabaf74d9b07dd3552786bfcc39123aaf242e379e171b821aa1e5731aebde"
        ),
        .binaryTarget(
            name: "Libngtcp2_crypto_gnutls",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.42-h3/Libngtcp2_crypto_gnutls.xcframework.zip",
            checksum: "01cedde8e4e50c718565cd751da1fd9dae4edd47c3038b8238a217024fbf27e9"
        ),
        .binaryTarget(
            name: "Libcurl",
            // curl-with-HTTP/3 (ngtcp2+nghttp3, GnuTLS), device-only, gebaut von
            // tv-h3-ios/build-curl-ios.sh. Trägt das FFmpeg `libcurl://`-Protokoll
            // (ff_libcurl_protocol, ersetzt das alte http3.c). curl_* sind in
            // Libavformat undefined → hier aufgelöst. CA via CURL_CA_BUNDLE-Env.
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libcurl.xcframework.zip",
            checksum: "eb3aade155b2ce8fde8b0770759dfbea834b23ec2084fae78391011339af6e3b"
        ),
        .binaryTarget(
            name: "Libavfilter",
            // prod-3-consistent rebuild (renderpl.17): vf_libplacebo against
            // libplacebo apiver 365, matching the Libplacebo target above.
            // renderpl.41: + dynaudnorm/speechnorm (Audio-Politur Hebel #4 —
            // Kuckuck.audioNormalize failte still mit -4, Filter fehlte in der
            // Whitelist).
            // renderpl.42: FFmpeg-Rebase d1faab7 → master 625ab01 (2026-06-11);
            // dynaudnorm/speechnorm/bwdif in der Whitelist bestätigt (Build-Log).
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libavfilter.xcframework.zip",
            checksum: "a45b7cc1cf1d4d6ca9451690ed208920e7dab31d297798f353edcdcb3aeb4202"
        ),
        .binaryTarget(
            name: "Libavutil",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libavutil.xcframework.zip",
            checksum: "241ee10f2f08b28ed6e0c7c7e3b0dd82794f524128e1d2552fdefaf7d5231150"
        ),
        .binaryTarget(
            name: "Libswresample",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libswresample.xcframework.zip",
            checksum: "1718ff7388298eb5d50aaf41d94dce0c9db463569d0816e2826bfd752ce2b0ae"
        ),
        .binaryTarget(
            name: "Libswscale",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.53/Libswscale.xcframework.zip",
            checksum: "f51f6201baf599b2068815fcaeb0511a09c006cbf124f64bf53003317a5d266f"
        ),

        .binaryTarget(
            name: "Libuchardet",
            url: "https://github.com/mpvkit/libuchardet-build/releases/download/0.0.8-xcode/Libuchardet.xcframework.zip",
            checksum: "503202caa0dafb6996b2443f53408a713b49f6c2d4a26d7856fd6143513a50d7"
        ),

        .binaryTarget(
            name: "Libluajit",
            url: "https://github.com/mpvkit/libluajit-build/releases/download/2.1.0-xcode/Libluajit.xcframework.zip",
            checksum: "8e76f267ee100ff5f3bbde7641b2240566df722241cdf8e135be7ef3d29e237a"
        ),

        .binaryTarget(
            name: "Libmpv",
            // render_pl libplacebo backend (patch 0016); ra_metal removed (renderpl.14).
            // renderpl.25: G Deband-pro-Content — env KUCKUCK_DEBAND (off|mild|strong;
            // unset=high_quality-Default), App-Gate pro Video. Grain 1.0 (IQ-Harness:
            // strukturelles Debanding folgt threshold, grain addierte nur Rauschen).
            // (renderpl.23: #3 Frame-Interpolation, vo_gpu_next-parity Multi-Frame-
            // pl_queue. .22: GPU-Deinterlace BWDIF + Error-Diffusion. .21: Shader-Swap.)
            // Pairs mit prod-3 Libplacebo/Libavfilter (renderpl.17).
            // renderpl.39: VT-SR auf Zero-Copy-hwdec — vt_sr_try_map nimmt
            // IMGFMT_VIDEOTOOLBOX (Decoder-CVPixelBuffer → kuckuck_vt_upscale_pixbuf,
            // Direct-Pass/Crop in der Bridge), map-Reorder vt_sr VOR vt_wrap. ABI:
            // neues extern kuckuck_vt_upscale_pixbuf — braucht App mit dem @_cdecl.
            // renderpl.42: mpv-Rebase 6444c05 → master 304426c (2026-06-09; u.a.
            // T.35-OOB-Read-Fix demux/packet) + FFmpeg-Rebase auf 625ab01. Patches
            // 0006-0016 unverändert re-appliet → ABI identisch .39 (kein App-Bump).
            // renderpl.38: Motion-Interp 50→60 (VT-Zwischenbilder, Default AN, enges
            // Phasenfenster) + vo_set_queue_params(3) (Multi-Frame-Mixe). ABI:
            // kuckuck_vt_upscale 13 Args — braucht App >=7714e4b.
            // renderpl.37: Zero-Copy-hwdec — VT-Device registriert (plain videotoolbox
            // funktioniert), Decoder-CVPixelBuffer via IOSurface direkt als pl_frame.
            // renderpl.36: No-finish-Modus — env KUCKUCK_NO_FINISH (flush statt finish;
            // Host presentet via CB auf geteilter Queue) + Host-Queue-Durchreichung.
            // renderpl.35: VT-Super-Resolution-Hook — env KUCKUCK_VT_SR, App-Bridge
            // kuckuck_vt_upscale (Apple-ML-SD-Upscaler, IOSurface zero-copy ins pl_frame).
            // renderpl.34: SDR-Helligkeits-Fix — Peak-Detection nur für HDR (CAS/ArtCNN-
            // User-Shader hoben lokale Peaks → prod-4/MR!861-Roll-off dimmte SDR „einen
            // Tick zu dunkel"). VT-Frame-Interp-Code wieder entfernt (verworfen).
            // renderpl.43: neuer Patch 0017 (vd_lavc) — VT-Background-Death-Recovery
            // an die richtige Stelle verlagert (war FFmpeg 0001-vt-inline-retry, jetzt
            // raus): bei hwdec-Decode-Fehler EINEN bounded reinit() des gleichen hwdec
            // vor dem SW-Abstieg (recover_count, Reset auf gutem Frame; wait_for_keyframe
            // schützt vor sofortigem Re-Fallback). Generisch (auch macOS-GPU-Reset).
            // ABI weiter identisch .39. ⚠️ Background-Recovery on-device noch zu re-verifizieren.
            // renderpl.44: BUGFIX Patch 0017 — hwdec_recover_count-Reset aus
            // uninit_avctx entfernt (reinit() ruft uninit → Reset → Endlos-reinit-
            // Loop wenn hwdec den Stream nie kann = interlaced raw-TS-Tuner: VOX/
            // RTL/… starteten nicht, nie SW-Fallback). Reset jetzt nur auf
            // erfolgreichem Decode. renderpl.43 ist tuner-kaputt → nicht nutzen.
            // renderpl.45: + env KUCKUCK_SDR_GAMMA (render_pl Patch 0016) —
            // libplacebo color_adjustment.gamma als Helligkeits-Nudge. gamma>1 hebt
            // die Mitteltöne (x^(1/gamma), kein Schwarz-Lift). NUR wenn das env gesetzt
            // ist + kein HDR-Target; sonst neutral. App setzt es ausschließlich für
            // SDR-Live-TV (Tuner + Mediathek-Live) — bt.1886-Broadcast wirkt auf dem
            // Handy (~sRGB-Erwartung) sonst einen Tick zu dunkel. Sonst identisch .44.
            // ABI unverändert (.39). Nur ios-arm64-Slice neu, Simulator unverändert.
            // renderpl.51: + kuckuck_hybrid_* (standalone libplacebo-Render-Entry für
            // den AVPlayer-Hybrid Mediathek-Live; render_pl.c + render_mtl.h). Device-
            // slice-swap auf .50; Libav* unverändert auf .50. App-Calls #if-guarded gegen
            // Simulator (Sim-Slice ohne die Symbole; App eh device-only via QUIC-Libs).
            // renderpl.52: kuckuck_hybrid_render erkennt HDR-Drawable (PL_FMT_FLOAT-Target)
            // → PQ/BT.2020-HDR10-Output mit HQ-Peak-Detect, gedeckelt via KUCKUCK_HDR_TARGET_NITS
            // (Default 1000). SDR-Pfad unverändert. Device-slice-swap auf .51; Sim unverändert.
            // ABI identisch (.39) — Signatur unverändert. Für den AV1-HDR-Hybrid-Pfad.
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.52/Libmpv.xcframework.zip",
            checksum: "3a79c8bbe1bf0a14c3f22f6a56e26cf0d54f364b832d41383554a4fbd4df649b"
        ),
        //AUTO_GENERATE_TARGETS_END//
    ]
)

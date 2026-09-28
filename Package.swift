// swift-tools-version:5.9

import PackageDescription

// libplacebo-Drop (2026-06-25): Der Fork vendet nur noch `Libkkrender` — den eigenen
// Metal-Renderer kk_gpu (SDR + HDR, self-contained, 0 libplacebo/spvc). Der alte
// mpv/FFmpeg/libplacebo/GPL-Stack (Decode lief eh über AVPlayer, Render jetzt kk_gpu)
// ist packaging-seitig komplett raus — ~29 binaryTargets + die GPL-/FFmpeg-Targets +
// das MPVKit-GPL-Produkt entfernt (waren alle ungenutzt). Build-Tooling zum Erzeugen
// alter Libs (BuildPlacebo/BuildFFMPEG/main.swift) ist davon unberührt.
let package = Package(
    name: "MPVKit",
    platforms: [.macOS(.v11), .iOS(.v14), .tvOS(.v14), .visionOS(.v1)],
    products: [
        .library(
            name: "MPVKit",
            targets: ["_MPVKit"]
        ),
    ],
    targets: [
        .target(
            name: "_MPVKit",
            dependencies: ["Libkkrender"],
            path: "Sources/_MPVKit",
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("Metal"),
                .linkedFramework("IOSurface"),
                .linkedFramework("QuartzCore"),
            ]
        ),
        .binaryTarget(
            name: "Libkkrender",
            // Standalone kk_gpu-Renderer (kuckuck_hybrid_* + kk_gpu_*). renderpl.60 =
            // libplacebo-frei, self-contained. Gebaut von kkrender/build-kkrender.sh.
            // renderpl.71 = echter SD-Deblock als MSL-Pass (~deblock-Gate) + die
            // Pass-Zeitmessung, die seit .70 im Branch lag, aber nie released war.
            // renderpl.73 = Bildqualität: Schwarzpunkt, Chroma-Ort (left), Dither.
            // renderpl.74 = Chroma per Lanczos3 (separabel, CHH-Vorpass; App gibt per Env vor).
            // renderpl.75 = Render-Sperre/Kontextzähler, H.273-Codepunkte, HD-Light 1:1 +
            // sRGB im letzten Pass, HDR-Peak aus MaxRGB (kk_gpu-Sweep 2026-09-25).
            // renderpl.76 = GPU-Fehler bis zur App (done(ud, ok) — ABI!), Caches pro Kontext,
            // setenv unter Sperre, Anti-Ringing, Dither-Ränder, HLG (kk_gpu-Sweep 2026-09-27).
            // renderpl.77 = Deband im HD-Sparpfad auf App-Vorgabe (KUCKUCK_DEBAND_HD, iPhone-Live).
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.77/Libkkrender.xcframework.zip",
            checksum: "5814e8b5cf8298e9cde490e271f1ee61f83d9b67e138060dab76d4d89ce32df1"
        ),
    ]
)

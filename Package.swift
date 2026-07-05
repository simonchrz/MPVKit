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
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-renderpl.67/Libkkrender.xcframework.zip",
            checksum: "d1065c0e545b5cfa332cf2c0aa030d3ed766c923cf0702a81ea1c03f577da6ff"
        ),
    ]
)

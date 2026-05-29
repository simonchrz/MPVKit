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
                "gmp", "nettle", "hogweed", "gnutls", "Libdav1d", "Libuavs3d"
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
                "Libsmbclient", "gmp", "nettle", "hogweed", "gnutls", "Libdav1d", "Libuavs3d"
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
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavcodec-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavcodec-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavdevice-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavdevice-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavformat-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavformat-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavfilter-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavfilter-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavutil-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libavutil-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libswresample-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libswresample-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libswscale-GPL",
            url: "https://github.com/mpvkit/MPVKit/releases/download/0.41.0-n8.1/Libswscale-GPL.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        //AUTO_GENERATE_TARGETS_BEGIN//

        .binaryTarget(
            name: "Libcrypto",
            url: "https://github.com/mpvkit/openssl-build/releases/download/3.3.5/Libcrypto.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libssl",
            url: "https://github.com/mpvkit/openssl-build/releases/download/3.3.5/Libssl.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "gmp",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/gmp.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "nettle",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/nettle.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "hogweed",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/hogweed.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "gnutls",
            url: "https://github.com/mpvkit/gnutls-build/releases/download/3.8.11/gnutls.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libunibreak",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libunibreak.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libfreetype",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libfreetype.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libfribidi",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libfribidi.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libharfbuzz",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libharfbuzz.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libass",
            url: "https://github.com/mpvkit/libass-build/releases/download/0.17.4/Libass.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libsmbclient",
            url: "https://github.com/mpvkit/libsmbclient-build/releases/download/4.15.13-2512/Libsmbclient.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libbluray",
            url: "https://github.com/mpvkit/libbluray-build/releases/download/1.4.0/Libbluray.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libuavs3d",
            url: "https://github.com/mpvkit/libuavs3d-build/releases/download/1.2.1-xcode/Libuavs3d.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libdovi",
            url: "https://github.com/mpvkit/libdovi-build/releases/download/3.3.2/Libdovi.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "MoltenVK",
            url: "https://github.com/mpvkit/moltenvk-build/releases/download/1.4.1/MoltenVK.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libshaderc_combined",
            url: "https://github.com/mpvkit/libshaderc-build/releases/download/2025.5.0/Libshaderc_combined.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "lcms2",
            url: "https://github.com/mpvkit/lcms2-build/releases/download/2.17.0/lcms2.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libplacebo",
            url: "https://github.com/mpvkit/libplacebo-build/releases/download/7.360.1/Libplacebo.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libdav1d",
            url: "https://github.com/mpvkit/libdav1d-build/releases/download/1.5.2-xcode/Libdav1d.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libavcodec",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libavcodec.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavdevice",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libavdevice.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavformat",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libavformat.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavfilter",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libavfilter.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libavutil",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libavutil.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libswresample",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libswresample.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        .binaryTarget(
            name: "Libswscale",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.41/Libswscale.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libuchardet",
            url: "https://github.com/mpvkit/libuchardet-build/releases/download/0.0.8-xcode/Libuchardet.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libluajit",
            url: "https://github.com/mpvkit/libluajit-build/releases/download/2.1.0-xcode/Libluajit.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),

        .binaryTarget(
            name: "Libmpv",
            url: "https://github.com/simonchrz/MPVKit/releases/download/0.41.0-master-n8.1-ios-metal.65-pinpoint/Libmpv.xcframework.zip",
            checksum: "60e40893df24f28f87db3a889ce45187af4a7e7aff83effddc95056205967b30"
        ),
        //AUTO_GENERATE_TARGETS_END//
    ]
)

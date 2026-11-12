// swift-tools-version: 6.2
// mlx-zonos3-tts-swift — Zyphra's ZONOS3-Mini (real-time multilingual TTS, 44.1 kHz) on Swift-MLX.
// Zonos3Core is the model (GDN / global-attention latent-MoE backbone, ZAC-1 iSTFT decoder, MECT-B2 speaker encoder);
// MLXZonos3TTS is the MLXEngine package on `tts`; zonos3-gates is the parity and validation CLI.
import PackageDescription

let package = Package(
    name: "mlx-zonos3-tts-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Zonos3Core", targets: ["Zonos3Core"]),
        .library(name: "MLXZonos3TTS", targets: ["MLXZonos3TTS"]),
        .executable(name: "zonos3-gates", targets: ["zonos3-gates"]),
    ],
    dependencies: [
        // The fleet's fork, as mlx-engine-swift ≥ 0.74.1 uses it (AB-D-0208): two URLs for one package identity is an error.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.32.4"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.77.0"),
    ],
    targets: [
        .target(
            name: "Zonos3Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
            ],
            path: "Sources/Zonos3Core",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXZonos3TTS",
            dependencies: [
                "Zonos3Core",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MLXZonos3TTS",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "zonos3-gates",
            dependencies: [
                "Zonos3Core", "MLXZonos3TTS",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/zonos3-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXZonos3TTSTests",
            dependencies: [
                "Zonos3Core", "MLXZonos3TTS",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXZonos3TTSTests"
        ),
    ]
)

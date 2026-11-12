// swift-tools-version: 6.2
// mlx-zonos3-tts-swift — Zyphra's ZONOS3-Mini (real-time multilingual TTS, 44.1 kHz) on Swift-MLX.
// Zonos3Core is the model (GDN / global-attention latent-MoE backbone, ZAC-1 iSTFT decoder, MECT-B2 speaker encoder);
// zonos3-gates is the parity and validation CLI. The MLXEngine package target is added in V3.
import PackageDescription

let package = Package(
    name: "mlx-zonos3-tts-swift",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "Zonos3Core", targets: ["Zonos3Core"]),
        .executable(name: "zonos3-gates", targets: ["zonos3-gates"]),
    ],
    dependencies: [
        // The fleet's fork, as mlx-engine-swift ≥ 0.74.1 uses it (AB-D-0208): two URLs for one package identity is an error.
        .package(url: "https://github.com/xocialize/mlx-swift", from: "0.32.4"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Zonos3Core",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "zonos3-gates",
            dependencies: [
                "Zonos3Core",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/zonos3-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)

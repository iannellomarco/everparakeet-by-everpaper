// swift-tools-version: 6.0
import PackageDescription

// EverParakeet native Swift runtime (Apple silicon: macOS 15+, iOS 18+).
//   EverParakeetKit      the library: Core ML encoder, CPU decoder, timed tokens, lanes, long-form. No Core AI.
//   EverParakeetCoreAI   optional Core AI encoder (macOS 27). A separate target so an app that only needs Core ML
//                        can leave it out; EverParakeetKit never links it.
//   everparakeet         command line (transcribe, smoke, batch).
let package = Package(
    name: "everparakeet",
    platforms: [.macOS("15.0"), .iOS("18.0")],
    products: [
        .library(name: "EverParakeetKit", targets: ["EverParakeetKit"]),
        .library(name: "EverParakeetCoreAI", targets: ["EverParakeetCoreAI"]),
        .executable(name: "everparakeet", targets: ["everparakeet"]),
    ],
    targets: [
        // Small NEON kernel (fp16 weights x fp32 activations) for the CPU decoder; pure C, no dependencies.
        .target(name: "CHalfKernels"),
        .target(
            name: "EverParakeetKit",
            dependencies: ["CHalfKernels"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "EverParakeetCoreAI",
            dependencies: ["EverParakeetKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "everparakeet",
            dependencies: ["EverParakeetKit", "EverParakeetCoreAI"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "EverParakeetKitTests",
            dependencies: ["EverParakeetKit"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)

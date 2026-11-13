// swift-tools-version: 5.9
import PackageDescription

// NOTE: this package is consumed only by the `swift build`/`swift test` CLI
// workflow. The real iOS app is built by ios/RoadSense.xcodeproj directly from
// the RoadSense/ source tree (plain Xcode file references, not an SPM
// package dependency), so nothing here affects that build. This package
// previously declared a "RoadSense" target meant to mirror the whole app, but
// it imports UIKit/AVFoundation multi-cam APIs and references generated CoreML
// model classes that only exist in an Xcode build -- it could never actually
// compile via `swift build` on macOS, which is why `swift test` had no working
// target to run until now. It has been removed rather than patched further:
// gutting it down to something that *does* compile on macOS would leave an
// empty, pointless duplicate of the real app target.
let package = Package(
    name: "RoadSense",
    defaultLocalization: "en",
    platforms: [
        // iOS 16 or later ensures full modern concurrency and background delegate support
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "RoadSenseAuth",
            targets: ["RoadSenseAuth"]
        )
    ],
    dependencies: [],
    targets: [
        // A subset of RoadSense's own source files that have no UIKit/
        // AVFoundation/CoreML-codegen dependency, so they can be unit tested
        // with plain `swift test` on macOS. These are the SAME files compiled
        // into the real app by the Xcode project above -- this target doesn't
        // copy or fork them.
        .target(
            name: "RoadSenseAuth",
            dependencies: [],
            path: "RoadSense",
            exclude: [
                "App",
                "Info.plist",
                "RoadSense.entitlements",
                "Models/FusionModel.mlpackage",
                "Models/VZCrashModel.mlpackage",
                "Models/SensorFrame.swift",
                "Services/BackgroundCaptureService.swift",
                // CollisionDetector.swift and CollisionIncidentStore.swift
                // were excluded here too, but both import only Foundation
                // (no UIKit/AVFoundation/CoreML) -- there was never a real
                // reason they couldn't be tested on macOS. Included so
                // StoredIncident's sync-rejection logic (this session's fix
                // for an infinite-retry bug) is actually unit tested.
                "Services/ContextManager.swift",
                "Services/AmazfitClient.swift",
                "Services/ObdBleClient.swift",
                "Services/RoadSenseOnDeviceRiskEngine.swift",
                "Services/Uploader.swift",
                "Services/WhoopClient.swift",
            ]
        ),
        .testTarget(
            name: "RoadSenseAuthTests",
            dependencies: ["RoadSenseAuth"],
            path: "Tests"
        ),
    ]
)

// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RoadSense",
    defaultLocalization: "en",
    platforms: [
        // iOS 16 or later ensures full modern concurrency and background delegate support
        .iOS(.v16)
    ],
    products: [
        // Exposes RoadSense as an executable app/library target
        .library(
            name: "RoadSense",
            targets: ["RoadSense"]
        )
    ],
    dependencies: [
        // All services (CoreLocation, CoreBluetooth, CoreMotion, Foundation) 
        // are native Apple system frameworks, so no third-party CocoaPods or SPM packages are needed.
    ],
    targets: [
        .target(
            name: "RoadSense",
            dependencies: [],
            path: "RoadSense",
            resources: [
                // Links Info.plist so background modes and permissions are bundled properly
                .process("Info.plist")
            ]
        )
    ]
)
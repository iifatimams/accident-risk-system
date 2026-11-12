#!/bin/bash
# Exercise the actual app networking/auth sources on macOS without changing its app package.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTH_TEST_DIR="$(mktemp -d /private/tmp/roadsense-auth-tests.XXXXXX)"
mkdir -p "$AUTH_TEST_DIR/Sources/RoadSenseAuth" "$AUTH_TEST_DIR/Tests/RoadSenseAuthTests"
cp "$REPO_ROOT/ios/RoadSense/Services/RoadSenseAPI.swift" "$REPO_ROOT/ios/RoadSense/Services/AuthManager.swift" "$AUTH_TEST_DIR/Sources/RoadSenseAuth/"
cp "$REPO_ROOT/ios/Tests/AuthTests.swift" "$AUTH_TEST_DIR/Tests/RoadSenseAuthTests/"
cat > "$AUTH_TEST_DIR/Package.swift" <<'PACKAGE'
// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "RoadSenseAuth", platforms: [.macOS(.v13)], targets: [
    .target(name: "RoadSenseAuth"),
    .testTarget(name: "RoadSenseAuthTests", dependencies: ["RoadSenseAuth"])
])
PACKAGE
printf 'Auth test artifacts: %s\n' "$AUTH_TEST_DIR"
swift test --package-path "$AUTH_TEST_DIR"

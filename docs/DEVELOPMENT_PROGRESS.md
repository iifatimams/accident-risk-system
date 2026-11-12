# RoadSense development progress

This tracks phase-by-phase status against the master development directive.
It is a resumption checkpoint, not a completion record: a phase is only
"done" when code is written AND integrated AND tested AND (where hardware is
required) verified on the physical device -- those are separate claims, and
this file keeps them separate. See `docs/DEVELOPMENT_CHECKPOINT_2026-11-12.md`
for the state immediately before this entry.

## Phase 1 -- Dual-camera capture: PARTIALLY COMPLETE

Scope actually completed this session (everything that does NOT require the
physical iPhone or a real driving session):

1. **Portrait/landscape rotation is no longer hardcoded.** Previously both the
   front and rear `AVCaptureConnection`s were hardcoded to
   `videoRotationAngle = 90`, which is only correct in portrait. Added
   `ios/RoadSense/Services/CameraOrientation.swift`: a pure, UIKit-free
   mapping from phone orientation + camera position to the correct rotation
   angle (front and back diverge in landscape because the two sensors are
   mounted rotated 180 degrees relative to each other).
   `BackgroundCaptureService.swift` now observes
   `UIDevice.orientationDidChangeNotification` and re-applies the correct
   angle to both connections live, instead of only setting it once at
   startup.
   - **UNVERIFIED on physical hardware.** The landscape front/back divergence
     follows Apple's documented AVCam sample convention from memory, not a
     measurement. Before trusting it: mount the phone in landscape (both
     orientations) and confirm the front-camera debug feed and rear bounding
     boxes are upright, not rotated or mirrored. If wrong, only the two
     `case` lines in `CameraOrientation.swift` need to change -- the dynamic
     update plumbing is already correct either way.
2. **Camera session lifecycle handling.** Added observers for
   `AVCaptureSession.wasInterruptedNotification`,
   `.interruptionEndedNotification`, and `.runtimeErrorNotification`. A
   runtime error now restarts the session instead of silently leaving stale
   `eyesOnRoad`/`trafficLevel` values on the dashboard forever. Previously
   there was no handling for phone calls, Siri, another app taking the
   camera, or a hardware fault at all.
3. **Real FPS/latency instrumentation**, published for the dashboard:
   `frontDeliveredFps`, `backDeliveredFps`, `frontProcessingLatencyMs`,
   `backProcessingLatencyMs`. These measure actual sample-buffer arrivals and
   actual Vision-request completion time, not the configured
   `activeVideoMinFrameDuration`. **No real numbers are reported here** --
   the point of this work was to make it *possible* to see them on a real
   device; nothing in this session ran on an iPhone.
4. **Fixed previously-broken Swift test infrastructure.** `ios/Package.swift`
   declared a `swift test`/`swift build`-able "RoadSense" target that could
   never actually compile (it depends on UIKit, which doesn't exist on
   macOS, plus a stray `Info.plist` resource declaration that failed
   immediately regardless). This meant `ios/Tests/AuthTests.swift` (12 tests,
   already written, already tracked in git) had never actually been run by
   anyone -- `swift test` failed before reaching it. Restructured
   `Package.swift` into a `RoadSenseAuth` library target containing only the
   UIKit/AVFoundation-free files (`AuthManager.swift`, `RoadSenseAPI.swift`,
   `CameraOrientation.swift`) plus a matching test target. This does **not**
   change what the real app builds -- `ios/RoadSense.xcodeproj` has no SPM
   package dependency at all and was already compiling every file directly;
   confirmed this by grepping the `.pbxproj` for any package reference
   (none) before making the change.
   - `swift test`: **17 passed** (12 pre-existing `AuthTests` that could not
     previously run, plus 5 new `CameraOrientationTests`).

What Phase 1 still needs, and why it couldn't happen this session:

- **Actual measured FPS/latency/thermal/battery numbers.** Requires a
  physical iPhone running a real or simulated drive. The directive is
  explicit that a build or simulator run does not establish this.
- **Confirming the landscape rotation mapping is actually correct**, not
  just internally consistent (see point 1 above).
- **Front-camera mirroring behavior** was not touched this session --
  `isVideoMirrored` was left at its existing (unset/default) value. This
  only matters once Phase 2 implements LEFT/RIGHT attention states, where
  mirroring determines which physical direction "left" corresponds to; it
  does not affect pitch, yaw, roll, or EAR math used today.
- **On-screen mounting/calibration guidance (Phase 1.3)** was not built.
  Recording *which* orientation a session used is now possible
  (`currentOrientation` is published) but nothing persists it into
  `SensorFrame`/the research export yet -- that's a small follow-up, not
  done here to keep this commit reviewable as one coherent change.
- Frame rate was **left at 3 fps** (the pre-existing
  `activeVideoMinFrameDuration`/`activeVideoMaxFrameDuration` of
  `CMTime(1, 3)`) for both cameras. The directive explicitly says not to
  raise capture rate without measurements, and no on-device measurement was
  possible this session. This is very likely part of why Phase 2's "looking
  sideways takes too long to detect" complaint exists, but changing it
  without evidence would violate the same rule it's trying to satisfy.

Verification performed this session:

- `xcodebuild -project ios/RoadSense.xcodeproj -scheme RoadSense -configuration Debug -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build`: **BUILD SUCCEEDED**. Compilation only, not a runtime or device test.
- `cd ios && swift test`: **17 passed, 0 failed.**
- `.venv/bin/python -m pytest -q`: **128 passed** (unchanged by this
  session's changes; run to confirm no regression).
- No Android files touched. No files under `data/raw` touched. No dataset
  downloads performed. `data/app.sqlite3` was already modified in the
  working tree at session start (unrelated to this work) and was
  deliberately left out of this commit.

## Phases 2-19

Not started this session. Phase 2 (front-camera attention states) is the
natural next step and does not require hardware to make progress on the
rule/state-machine logic, though validating it against the documented
nighttime failure modes does.

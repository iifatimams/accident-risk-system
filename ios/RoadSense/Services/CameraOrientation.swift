import Foundation

/// Physical orientation of the phone, independent of UIKit's `UIDeviceOrientation`
/// so this mapping can be unit tested on any platform without importing UIKit.
/// `faceUp`/`faceDown`/`unknown` map to `.portrait` by the caller (see
/// `BackgroundCaptureService`) since AVFoundation still needs a concrete rotation
/// angle even when the accelerometer can't tell us which way the phone is held.
public enum PhoneOrientation: Equatable {
  case portrait
  case portraitUpsideDown
  case landscapeLeft
  case landscapeRight
}

/// Which physical camera a connection belongs to. The front and rear sensors on
/// an iPhone are mounted rotated 180 degrees relative to each other, so the same
/// `PhoneOrientation` needs a different rotation angle for each.
public enum CameraPosition: Equatable {
  case front
  case back
}

/// Pure mapping from how the phone is physically held to the
/// `AVCaptureConnection.videoRotationAngle` needed so the delivered pixel buffer
/// is upright for Vision and for any on-screen preview. Values follow Apple's
/// AVCam sample pattern (rotation angle, in degrees, of the buffer's "up"
/// direction relative to gravity-up): 0/90/180/270 are the only angles
/// `AVCaptureConnection` accepts.
///
/// Previously this was hardcoded to 90 degrees for both cameras, which is only
/// correct when the phone is mounted in portrait. Mounting it in landscape
/// produced a sideways image and correspondingly wrong Vision pitch/yaw/roll
/// axes and bounding boxes.
///
/// Portrait/upside-down angles are the same for both cameras; landscape angles
/// diverge between front and back because the two sensors are mounted rotated
/// 180 degrees relative to each other. The divergence follows Apple's AVCam
/// sample convention (front=180/back=0 for landscapeLeft, front=0/back=180 for
/// landscapeRight). UNVERIFIED on physical hardware: confirm by mounting the
/// phone in each landscape orientation and checking that the front-camera
/// debug preview and rear-camera bounding boxes are upright, not rotated.
public func videoRotationAngle(
  for orientation: PhoneOrientation,
  camera: CameraPosition
) -> Double {
  switch (orientation, camera) {
  case (.portrait, .front), (.portrait, .back):
    return 90
  case (.portraitUpsideDown, .front), (.portraitUpsideDown, .back):
    return 270
  case (.landscapeLeft, .front):
    return 180
  case (.landscapeLeft, .back):
    return 0
  case (.landscapeRight, .front):
    return 0
  case (.landscapeRight, .back):
    return 180
  }
}

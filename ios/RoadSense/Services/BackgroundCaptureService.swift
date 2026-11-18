import AVFoundation
import CoreLocation
import CoreML
import CoreMotion
import Foundation
import UIKit
import Vision

// MARK: - Vision Manager (7-Factor Multimodal Tracking)
public class VisionManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate
{
  // DMD-trained gaze-ZONE model (scripts/train_dmd_gaze_v3.py): 3 classes
  // (road/brief_glance/off_road), trained on DMD's dedicated GAZE
  // category annotations -- a real improvement over the earlier binary
  // looking_road/not_looking_road model (which used the coarser
  // DISTRACTION category label). Only its "off_road" probability is
  // consumed, used ONLY to confirm/escalate Apple Vision's own (richer,
  // directional) gaze classification -- see combineDMDWithVision in
  // DriverAttentionState.swift for why it can never clear a Vision-
  // flagged distraction. nil (model unavailable or failed to load)
  // falls back to Vision alone, never silently to "attentive."
  //
  // Head-only trained (classifier layer only, backbone frozen) --
  // deliberately NOT fine-tuning deeper layers like the earlier binary
  // model did: that approach overfit badly on this dataset's small,
  // low-diversity-per-driver calibration clips (held-out accuracy
  // collapsed to 0.345, off_road recall 0.023) before this head-only
  // recipe was found to generalize far better (0.560 accuracy, 0.452
  // off_road recall) -- see docs/FINAL_MULTIMODAL_RESULTS.md for both
  // numbers, a genuine dataset-specific finding, not a general rule.
  private let dmdAttentionModel: DMDAttentionModel? = {
    let config = MLModelConfiguration()
    return try? DMDAttentionModel(configuration: config)
  }()
  // Real, trained, Core-ML-deployed road-object detector (YOLOv8n,
  // COCO-pretrained via the `ultralytics` package's own Core ML export
  // path -- the project's original BDD100K-fine-tune attempt regressed
  // vs. the pretrained baseline and was abandoned; the earlier pretrained
  // torchvision Faster R-CNN could not be exported to Core ML at all.
  // This one verified correctly on a real test image before being
  // bundled here: a bus + 4 people photo returned class 5/"bus" at 0.92
  // confidence and 4 class-0/"person" detections at 0.37-0.85 -- see
  // docs/FINAL_MULTIMODAL_RESULTS.md for the exact command). Wrapped as
  // VNCoreMLModel (not the generated Swift class) because Vision handles
  // this model's image resize/NMS-threshold inputs automatically.
  private let roadObjectDetector: VNCoreMLModel? = {
    let config = MLModelConfiguration()
    guard let model = try? RoadObjectDetector(configuration: config) else {
      return nil
    }
    return try? VNCoreMLModel(for: model.model)
  }()
  // COCO class names this project treats as "a vehicle ahead" evidence
  // for lead-object tracking -- deliberately excludes "person" (the
  // existing, separately-validated VNDetectHumanRectanglesRequest
  // pipeline already owns pedestrian detection) and non-road COCO
  // classes the model can also emit.
  private static let vehicleClassLabels: Set<String> = ["car", "truck", "bus", "motorcycle", "bicycle", "train"]
  // MARK: Per-Driver Calibration (DriverCalibration.swift)
  @Published public private(set) var isCalibrating: Bool = false
  @Published public private(set) var calibrationProgress: Double = 0.0
  private var calibrationSession: CalibrationSession?
  private var calibratingDriverId: Int?
  private var currentCalibrationBaseline: CalibrationBaseline?

  /// Starts a short "just look at the road normally" calibration
  /// session for the given signed-in driver. Safe to call mid-trip;
  /// every subsequent `processDriverFace` frame records a sample until
  /// `minimumSamples` is reached, at which point the resulting baseline
  /// is saved (DriverCalibrationStore) and activated immediately.
  public func startCalibration(driverId: Int, minimumSamples: Int = 45) {
    calibratingDriverId = driverId
    calibrationSession = CalibrationSession(minimumSamples: minimumSamples)
    Task { @MainActor in
      self.isCalibrating = true
      self.calibrationProgress = 0.0
    }
  }

  public func cancelCalibration() {
    calibrationSession = nil
    calibratingDriverId = nil
    Task { @MainActor in
      self.isCalibrating = false
      self.calibrationProgress = 0.0
    }
  }

  /// Loads a previously-saved baseline for this driver (e.g. right after
  /// sign-in, so a returning driver benefits from calibration done on an
  /// earlier trip without re-calibrating). Clears to nil (uncalibrated,
  /// not a fabricated "neutral") if this driver has never calibrated.
  public func loadCalibration(forDriverId driverId: Int) {
    currentCalibrationBaseline = DriverCalibrationStore.shared.baseline(forDriverId: driverId)
  }

  @Published public var eyesClosedSec: Double = 0.0
  /// How long the COMMITTED (post-hysteresis) attention state has been
  /// continuously off-road -- i.e. this only starts counting once a
  /// brief glance has already been filtered out by AttentionHysteresis's
  /// own dwell time, so a mirror check never shows up here at all. Feeds
  /// MultimodalRiskFusion's attention-dimension severity/confidence
  /// (duration of sustained off-road gaze, not just its current
  /// instantaneous state).
  @Published public var offRoadGazeSec: Double = 0.0
  @Published public var eyesOnRoad: Bool = true
  @Published public var postureCollapse: Bool = false
  @Published public var leadClosingFast: Bool = false
  @Published public var trafficLevel: String = "empty"
  @Published public var pedestriansDetected: Bool = false
  @Published public var pedestrianCount: Int = 0
  @Published public var cameraStatus: String = "idle"
  // MARK: Forward Collision Estimation (Phase 5/6)
  // The current best "lead vehicle" candidate, if any: the largest
  // car-class track from vehicleTracker below. Published as the raw track
  // (not yet classified into a ForwardHazardState) because classification
  // needs ego speed, which lives on BackgroundCaptureService, not here --
  // see sendCurrentSnapshot. Deliberately NOT claiming a hazard verdict at
  // this layer.
  @Published public var leadTrack: TrackedObject?
  private var vehicleTracker = ObjectTracker()
  // MARK: Pedestrian Hazard (Phase 7)
  @Published public var pedestrianHazardState: PedestrianHazardState = .insufficientEvidence
  private var pedestrianTracker = ObjectTracker()
  // Raw diagnostic values. These exist so the thresholds below can be
  // calibrated against real readings (phone mount angle, lighting) instead
  // of guessed blind. Watch these on the dashboard while driving/parked.
  @Published public var debugFaceDetected: Bool = false
  @Published public var debugPitch: Double = 0.0
  @Published public var debugYaw: Double = 0.0
  @Published public var debugRoll: Double = 0.0
  @Published public var debugEAR: Double = 0.0
  // MARK: Driver Attention
  // Discrete state (see DriverAttentionState.swift) debounced through
  // AttentionHysteresis. `eyesOnRoad` above is kept for backend/dashboard
  // compatibility and is now always derived from this, not tracked
  // separately -- see processDriverFace.
  @Published public var attentionState: DriverAttentionState = .attentive
  private var attentionHysteresis = AttentionHysteresis(
    enterDwell: 1.5,
    // Not empirically tuned -- there is no labeled real-driving dataset
    // available to calibrate against. This fixes a real bug (see
    // DriverAttentionState.swift) but the exact value is a starting point
    // for on-device tuning, not a validated number.
    exitDwell: 0.5
  )
  // MARK: Mounting + Performance Diagnostics
  // Added so capture FPS, processing latency and mounting orientation can
  // actually be measured on a real device instead of assumed. These are
  // deliberately exposed on the dashboard, not just logged, because the
  // only way to validate camera performance claims is to watch real numbers
  // while driving/parked -- see docs/DEVELOPMENT_PROGRESS.md Phase 1.
  @Published public var currentOrientation: PhoneOrientation = .portrait
  @Published public var frontDeliveredFps: Double = 0.0
  @Published public var backDeliveredFps: Double = 0.0
  @Published public var frontProcessingLatencyMs: Double = 0.0
  @Published public var backProcessingLatencyMs: Double = 0.0
  private var frontFrameTimestamps: [CFAbsoluteTime] = []
  private var backFrameTimestamps: [CFAbsoluteTime] = []
  private let fpsWindowSize = 10
  private var multiCamSession: AVCaptureMultiCamSession?
  private let frontQueue = DispatchQueue(
    label: "front_camera_queue",
    qos: .userInteractive
  )
  private let backQueue = DispatchQueue(
    label: "back_camera_queue",
    qos: .userInteractive
  )
  private var frontConnection: AVCaptureConnection?
  private var backConnection: AVCaptureConnection?
  private var eyesClosedStart: Date?
  private var offRoadGazeStart: Date?
  private var lastPitch: Double = 0.0
  private var lastYaw: Double = 0.0
  public override init() {
    super.init()
  }
  public func startCamera() {
    UIDevice.current.beginGeneratingDeviceOrientationNotifications()
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(deviceOrientationDidChange),
      name: UIDevice.orientationDidChangeNotification,
      object: nil
    )
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(sessionWasInterrupted),
      name: AVCaptureSession.wasInterruptedNotification,
      object: nil
    )
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(sessionInterruptionEnded),
      name: AVCaptureSession.interruptionEndedNotification,
      object: nil
    )
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(sessionRuntimeError),
      name: AVCaptureSession.runtimeErrorNotification,
      object: nil
    )
    Task.detached {
      switch AVCaptureDevice.authorizationStatus(for: .video) {
      case .authorized:
        self.setupMultiCam()
      case .notDetermined:
        AVCaptureDevice.requestAccess(for: .video) { granted in
          if granted {
            self.setupMultiCam()
          } else {
            self.reportCameraFailure("Camera access denied by user")
          }
        }
      default:
        self.reportCameraFailure(
          "Camera access not authorized (denied or restricted in Settings)"
        )
      }
    }
  }
  private func reportCameraFailure(_ reason: String) {
    print("[RoadSense] Camera setup failed: \(reason)")
    Task { @MainActor in
      self.cameraStatus = "failed: \(reason)"
    }
  }
  // MARK: Orientation
  // UIDeviceOrientation is translated to the UIKit-free `PhoneOrientation`
  // from CameraOrientation.swift immediately, so the actual rotation-angle
  // decision stays unit-testable. `.faceUp`/`.faceDown`/`.unknown` have no
  // defined rotation and keep whatever the last known orientation was,
  // rather than guessing -- those cases mean the phone is lying flat
  // (unmounted) or the accelerometer can't tell, not that it changed.
  private func mappedDeviceOrientation(_ device: UIDeviceOrientation) -> PhoneOrientation? {
    switch device {
    case .portrait: return .portrait
    case .portraitUpsideDown: return .portraitUpsideDown
    case .landscapeLeft: return .landscapeLeft
    case .landscapeRight: return .landscapeRight
    default: return nil
    }
  }
  // UIDevice posts this notification on the main thread, so this can touch
  // the @Published currentOrientation directly without a MainActor hop.
  @objc private func deviceOrientationDidChange() {
    guard
      let orientation = mappedDeviceOrientation(UIDevice.current.orientation),
      orientation != currentOrientation
    else {
      return
    }
    currentOrientation = orientation
    applyRotation(for: orientation)
  }
  private func applyRotation(for orientation: PhoneOrientation) {
    let frontAngle = videoRotationAngle(for: orientation, camera: .front)
    let backAngle = videoRotationAngle(for: orientation, camera: .back)
    if let connection = frontConnection,
      connection.isVideoRotationAngleSupported(frontAngle)
    {
      connection.videoRotationAngle = frontAngle
    }
    if let connection = backConnection,
      connection.isVideoRotationAngleSupported(backAngle)
    {
      connection.videoRotationAngle = backAngle
    }
  }
  // MARK: Session Lifecycle
  // Without these, a phone call, Siri, another app grabbing the camera, or a
  // transient hardware fault silently stops frame delivery -- the dashboard
  // would keep showing the last cached state (e.g. "eyesOnRoad: true")
  // forever instead of surfacing that the camera actually stopped.
  @objc private func sessionWasInterrupted(_ note: Notification) {
    Task { @MainActor in
      self.cameraStatus = "interrupted"
    }
  }
  @objc private func sessionInterruptionEnded(_ note: Notification) {
    Task { @MainActor in
      self.cameraStatus = "running"
    }
  }
  @objc private func sessionRuntimeError(_ note: Notification) {
    reportCameraFailure("Session runtime error; restarting")
    // AVCaptureMultiCamSession does not reliably recover on its own from a
    // runtime error (e.g. the system reclaiming camera hardware under
    // thermal pressure). Restarting from scratch is simpler and safer than
    // trying to patch the existing session back to a running state.
    stopCamera()
    startCamera()
  }
  // Called the moment driving is confirmed (speed sustained above the
  // threshold). Eye-closed/distraction timers run continuously even while
  // parked, so without this reset, time spent looking down or with eyes
  // closed before you start moving carries straight into the trip and can
  // immediately trip the hard "prolonged eye closure" override.
  public func resetDrivingState() {
    attentionHysteresis = AttentionHysteresis(enterDwell: attentionHysteresis.enterDwell, exitDwell: attentionHysteresis.exitDwell)
    Task { @MainActor in
      self.eyesClosedStart = nil
      self.offRoadGazeStart = nil
      self.eyesClosedSec = 0.0
      self.offRoadGazeSec = 0.0
      self.eyesOnRoad = true
      self.attentionState = .attentive
    }
  }
  public func stopCamera() {
    NotificationCenter.default.removeObserver(self)
    UIDevice.current.endGeneratingDeviceOrientationNotifications()
    multiCamSession?.stopRunning()
    multiCamSession = nil
    attentionHysteresis = AttentionHysteresis(enterDwell: attentionHysteresis.enterDwell, exitDwell: attentionHysteresis.exitDwell)
    eyesClosedStart = nil
    offRoadGazeStart = nil
    frontFrameTimestamps.removeAll()
    backFrameTimestamps.removeAll()
    vehicleTracker = ObjectTracker()
    pedestrianTracker = ObjectTracker()
    Task { @MainActor in
      self.eyesOnRoad = true
      self.attentionState = .attentive
      self.eyesClosedSec = 0.0
      self.offRoadGazeSec = 0.0
      self.postureCollapse = false
      self.trafficLevel = "empty"
      self.leadClosingFast = false
      self.leadTrack = nil
      self.pedestriansDetected = false
      self.pedestrianCount = 0
      self.pedestrianHazardState = .insufficientEvidence
      self.cameraStatus = "idle"
      self.frontDeliveredFps = 0.0
      self.backDeliveredFps = 0.0
      self.frontProcessingLatencyMs = 0.0
      self.backProcessingLatencyMs = 0.0
    }
  }
  private func setupMultiCam() {
    guard AVCaptureMultiCamSession.isMultiCamSupported else {
      reportCameraFailure("Multi-camera capture is not supported on this device")
      return
    }
    // Seed from whatever orientation the phone is actually in when capture
    // starts, rather than assuming portrait. deviceOrientationDidChange keeps
    // this correct afterward if the mount changes mid-session.
    let startupOrientation = mappedDeviceOrientation(UIDevice.current.orientation) ?? .portrait
    Task { @MainActor in
      self.currentOrientation = startupOrientation
    }
    let session = AVCaptureMultiCamSession()
    session.beginConfiguration()
    // MARK: Front Camera
    guard
      let frontCam = AVCaptureDevice.default(
        .builtInWideAngleCamera,
        for: .video,
        position: .front
      ),
      let frontInput = try? AVCaptureDeviceInput(
        device: frontCam
      )
    else {
      reportCameraFailure("Front camera unavailable")
      return
    }
    if (try? frontCam.lockForConfiguration()) != nil {
      frontCam.activeVideoMinFrameDuration = CMTime(
        value: 1,
        timescale: 3
      )
      frontCam.activeVideoMaxFrameDuration = CMTime(
        value: 1,
        timescale: 3
      )
      frontCam.unlockForConfiguration()
    }
    if session.canAddInput(frontInput) {
      session.addInputWithNoConnections(frontInput)
    }
    let frontOutput = AVCaptureVideoDataOutput()
    frontOutput.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA)
    ]
    frontOutput.setSampleBufferDelegate(
      self,
      queue: frontQueue
    )
    if session.canAddOutput(frontOutput) {
      session.addOutputWithNoConnections(frontOutput)
    }
    if let frontPort = frontInput.ports.first(
      where: { $0.mediaType == .video }
    ) {
      let connection = AVCaptureConnection(
        inputPorts: [frontPort],
        output: frontOutput
      )
      let angle = videoRotationAngle(for: startupOrientation, camera: .front)
      if connection.isVideoRotationAngleSupported(angle) {
        connection.videoRotationAngle = angle
      }
      if session.canAddConnection(connection) {
        session.addConnection(connection)
        self.frontConnection = connection
      }
    }
    // MARK: Back Camera
    guard
      let backCam = AVCaptureDevice.default(
        .builtInWideAngleCamera,
        for: .video,
        position: .back
      ),
      let backInput = try? AVCaptureDeviceInput(
        device: backCam
      )
    else {
      reportCameraFailure("Back camera unavailable")
      return
    }
    if (try? backCam.lockForConfiguration()) != nil {
      backCam.activeVideoMinFrameDuration = CMTime(
        value: 1,
        timescale: 3
      )
      backCam.activeVideoMaxFrameDuration = CMTime(
        value: 1,
        timescale: 3
      )
      backCam.unlockForConfiguration()
    }
    if session.canAddInput(backInput) {
      session.addInputWithNoConnections(backInput)
    }
    let backOutput = AVCaptureVideoDataOutput()
    backOutput.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String:
        Int(kCVPixelFormatType_32BGRA)
    ]
    backOutput.setSampleBufferDelegate(
      self,
      queue: backQueue
    )
    if session.canAddOutput(backOutput) {
      session.addOutputWithNoConnections(backOutput)
    }
    if let backPort = backInput.ports.first(
      where: { $0.mediaType == .video }
    ) {
      let connection = AVCaptureConnection(
        inputPorts: [backPort],
        output: backOutput
      )
      let angle = videoRotationAngle(for: startupOrientation, camera: .back)
      if connection.isVideoRotationAngleSupported(angle) {
        connection.videoRotationAngle = angle
      }
      if session.canAddConnection(connection) {
        session.addConnection(connection)
        self.backConnection = connection
      }
    }
    session.commitConfiguration()
    session.startRunning()
    self.multiCamSession = session
    print("[RoadSense] Multi-cam session started")
    Task { @MainActor in
      self.cameraStatus = "running"
    }
  }
  public func captureOutput(
    _ output: AVCaptureOutput,
    didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    guard
      let pixelBuffer = CMSampleBufferGetImageBuffer(
        sampleBuffer
      )
    else {
      return
    }
    let frameArrival = CFAbsoluteTimeGetCurrent()
    if connection == frontConnection {
      recordDeliveredFrame(timestamps: &frontFrameTimestamps, at: frameArrival, isFront: true)
      processDriverFace(pixelBuffer)
      publishProcessingLatency(since: frameArrival, isFront: true)
    } else if connection == backConnection {
      recordDeliveredFrame(timestamps: &backFrameTimestamps, at: frameArrival, isFront: false)
      processRoadTraffic(pixelBuffer)
      publishProcessingLatency(since: frameArrival, isFront: false)
    }
  }
  // MARK: Capture Diagnostics
  // Delivered FPS is measured from actual sample-buffer arrivals, not the
  // configured activeVideoMinFrameDuration -- the two can diverge under
  // thermal throttling, low light (longer exposure), or if the risk engine
  // or network uploader ever blocks this queue. Published so real numbers
  // are visible on the dashboard instead of assumed from the config value.
  private func recordDeliveredFrame(
    timestamps: inout [CFAbsoluteTime], at now: CFAbsoluteTime, isFront: Bool
  ) {
    timestamps.append(now)
    if timestamps.count > fpsWindowSize {
      timestamps.removeFirst(timestamps.count - fpsWindowSize)
    }
    guard timestamps.count >= 2,
      let first = timestamps.first
    else {
      return
    }
    let elapsed = now - first
    guard elapsed > 0 else {
      return
    }
    let fps = Double(timestamps.count - 1) / elapsed
    Task { @MainActor in
      if isFront {
        self.frontDeliveredFps = fps
      } else {
        self.backDeliveredFps = fps
      }
    }
  }
  // Processing latency: time from frame arrival to Vision (and, for the rear
  // camera, the rectangle/pedestrian requests) completing on this same
  // serial queue. This is what determines whether a frame can be fully
  // processed before the next one arrives at the configured capture rate.
  private func publishProcessingLatency(since start: CFAbsoluteTime, isFront: Bool) {
    let latencyMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
    Task { @MainActor in
      if isFront {
        self.frontProcessingLatencyMs = latencyMs
      } else {
        self.backProcessingLatencyMs = latencyMs
      }
    }
  }
  private func getEyeOpenness(
    eye: VNFaceLandmarkRegion2D?
  ) -> CGFloat {
    guard let pts = eye?.normalizedPoints,
      !pts.isEmpty
    else {
      return 1.0
    }
    let minX = pts.map { $0.x }.min() ?? 0
    let maxX = pts.map { $0.x }.max() ?? 0
    let minY = pts.map { $0.y }.min() ?? 0
    let maxY = pts.map { $0.y }.max() ?? 0
    let width = maxX - minX
    let height = maxY - minY
    if width == 0 {
      return 1.0
    }
    return height / width
  }
  /// Bridges Vision's `VNFaceLandmarkRegion2D` (CGPoint-based) to
  /// DriverAttentionState.swift's pure, Foundation-only `gazeOffset(...)`
  /// (which can't import Vision -- see that file's own testability
  /// rationale). nil when either the pupil or the eye-contour region is
  /// missing this frame; Vision's pupil region, when present, is a
  /// single point, so `.first` is the pupil location, not an arbitrary
  /// choice among several.
  private static func gazeOffsetFromLandmarks(
    pupil: VNFaceLandmarkRegion2D?, eye: VNFaceLandmarkRegion2D?
  ) -> GazeOffset? {
    guard let pupilPoint = pupil?.normalizedPoints.first,
      let eyePoints = eye?.normalizedPoints, !eyePoints.isEmpty
    else {
      return nil
    }
    return gazeOffset(
      pupil: (x: Double(pupilPoint.x), y: Double(pupilPoint.y)),
      eyeContour: eyePoints.map { (x: Double($0.x), y: Double($0.y)) }
    )
  }
  // MARK: Driver Face Tracking
  private func processDriverFace(
    _ pixelBuffer: CVPixelBuffer
  ) {
    // Two separate requests against the same frame:
    //  - landmarksRequest: eye region points, for the EAR (eyes-closed) calc.
    //  - poseRequest: head rotation, for distraction (pitch/yaw/roll).
    // These used to come from one VNDetectFaceLandmarksRequest. Setting
    // its revision to 3 made no difference -- on-device testing showed
    // pitch/yaw/roll reading exactly 0.0 on every single frame regardless
    // of head position, even while visibly looking down or turned away,
    // which means isDistracted could never fire. VNDetectFaceRectanglesRequest
    // is documented to populate pitch/yaw/roll independently of landmarks,
    // so pose now comes from that dedicated request instead; landmarks
    // stays only for the eye points EAR already relies on (which do work,
    // confirmed by the debug EAR readout tracking real values).
    let landmarksRequest = VNDetectFaceLandmarksRequest()
    let poseRequest = VNDetectFaceRectanglesRequest()
    if #available(iOS 15.0, *) {
      poseRequest.revision = VNDetectFaceRectanglesRequestRevision3
    }
    do {
      try VNSequenceRequestHandler().perform(
        [landmarksRequest, poseRequest],
        on: pixelBuffer,
        orientation: .up
      )
    } catch {
      return
    }
    let now = Date().timeIntervalSince1970
    guard
      let face = (landmarksRequest.results as? [VNFaceObservation])?.first
    else {
      // A missing face is ambiguous on its own -- a blink, a bump, or a
      // momentary tracking miss can drop a single frame even while the
      // driver is looking straight ahead. AttentionHysteresis (not this
      // call site) is what decides whether a miss is sustained enough to
      // actually report faceNotDetected/not-eyes-on-road; it requires the
      // SAME sustained evidence to clear back to attentive, so one stray
      // re-detection mid-turn can no longer instantly cancel a real
      // sustained loss (see DriverAttentionState.swift for why that
      // mattered).
      let committed = attentionHysteresis.update(instantaneous: .faceNotDetected, now: now)
      Task { @MainActor in
        self.debugFaceDetected = false
        self.eyesClosedStart = nil
        self.eyesClosedSec = 0.0
        // A genuinely missing face is a CONFIDENCE problem, not a
        // confirmed-distraction one -- never let it grow the off-road-
        // gaze severity duration (see the matching comment in the
        // face-detected path below).
        self.offRoadGazeStart = nil
        self.offRoadGazeSec = 0.0
        self.postureCollapse = false
        self.attentionState = committed
        self.eyesOnRoad = committed.isEyesOnRoad
      }
      return
    }
    // Prefer the dedicated pose request's face; fall back to the
    // landmarks face's own (likely-nil) pose fields only if the pose
    // pass somehow found nothing this frame.
    let poseFace =
      (poseRequest.results as? [VNFaceObservation])?.first
    let pitch =
      poseFace?.pitch?.doubleValue
      ?? face.pitch?.doubleValue
      ?? 0.0
    let yaw =
      poseFace?.yaw?.doubleValue
      ?? face.yaw?.doubleValue
      ?? 0.0
    let roll =
      poseFace?.roll?.doubleValue
      ?? face.roll?.doubleValue
      ?? 0.0
    let isCollapsed =
      (pitch < -0.8) || (abs(roll) > 0.7)
    let leftEAR = self.getEyeOpenness(
      eye: face.landmarks?.leftEye
    )
    let rightEAR = self.getEyeOpenness(
      eye: face.landmarks?.rightEye
    )
    let avgEAR = (leftEAR + rightEAR) / 2.0
    let isEyesClosed = avgEAR < GazeThresholds.default.earClosed
    // Eye-gaze-by-pupil-position (DriverAttentionState.swift's GazeOffset)
    // -- genuinely independent of head pose, so it catches "phone held
    // below the wheel, head level" the way pitch alone cannot. nil when
    // either eye's pupil/contour landmarks weren't reliably found this
    // frame (e.g. a blink, poor lighting), not fabricated.
    let leftGaze = Self.gazeOffsetFromLandmarks(pupil: face.landmarks?.leftPupil, eye: face.landmarks?.leftEye)
    let rightGaze = Self.gazeOffsetFromLandmarks(pupil: face.landmarks?.rightPupil, eye: face.landmarks?.rightEye)
    let combinedGaze = averageGazeOffset(left: leftGaze, right: rightGaze)
    // If a calibration session is active, this frame's RAW (uncalibrated)
    // reading is recorded toward the driver's personal baseline --
    // calibration must be measured from the same raw signal it will
    // later be subtracted from, not from an already-adjusted one.
    if var session = calibrationSession {
      session.record(CalibrationSample(pitch: pitch, yaw: yaw, roll: roll, gaze: combinedGaze))
      calibrationSession = session
      let progress = session.progress
      if session.isComplete, let driverId = calibratingDriverId,
        let baseline = session.finish(now: Date().timeIntervalSince1970)
      {
        DriverCalibrationStore.shared.save(baseline, forDriverId: driverId)
        currentCalibrationBaseline = baseline
        calibrationSession = nil
        calibratingDriverId = nil
        Task { @MainActor in
          self.calibrationProgress = 1.0
          self.isCalibrating = false
        }
      } else {
        Task { @MainActor in
          self.calibrationProgress = progress
        }
      }
    }
    // Re-centers this frame's reading around the driver's own personal
    // baseline (if calibrated) BEFORE classification, so the shared,
    // still-unvalidated-against-real-footage thresholds in
    // GazeThresholds apply more fairly to a driver whose own "neutral"
    // isn't exactly 0 -- see DriverCalibration.swift.
    let calibrated = applyCalibration(pitch: pitch, yaw: yaw, roll: roll, gaze: combinedGaze, baseline: currentCalibrationBaseline)
    let visionInstantaneous = classifyGaze(
      pitch: calibrated.pitch,
      yaw: calibrated.yaw,
      roll: calibrated.roll,
      eyeAspectRatio: avgEAR,
      faceDetected: true,
      confidence: face.confidence,
      gaze: calibrated.gaze
    )
    // DMD's own training frames come from DMD's dedicated face-facing
    // "rgb_face" camera stream, not a software crop of a wider scene --
    // the iPhone's front (selfie) camera is the same kind of driver-
    // facing view, so the FULL pixelBuffer is passed here, matching
    // training distribution, rather than cropping to Vision's face
    // bounding box.
    //
    // DMDAttentionModel is now the 3-class gaze-ZONE model
    // (scripts/train_dmd_gaze_v3.py, trained on DMD's dedicated GAZE
    // category annotations: road/brief_glance/off_road), replacing the
    // earlier binary not_looking_road model. Only the "off_road"
    // probability feeds `combineDMDWithVision` below, deliberately NOT
    // "brief_glance" -- a mirror/instrument-cluster glance is legitimate
    // driving behavior and must not push toward a distraction escalation
    // the way a genuine off-road reading should. "brief_glance"'s own
    // probability is real model output but not yet consumed anywhere;
    // see docs/FINAL_MULTIMODAL_RESULTS.md for why the richer 3-class
    // signal is only partially used so far.
    var dmdOffRoadProb: Double?
    if let model = dmdAttentionModel,
      let prediction = try? model.prediction(image: pixelBuffer)
    {
      dmdOffRoadProb = prediction.classLabel_probs["off_road"]
    }
    let instantaneous = combineDMDWithVision(
      visionState: visionInstantaneous, dmdNotLookingRoadProb: dmdOffRoadProb
    )
    let committed = attentionHysteresis.update(instantaneous: instantaneous, now: now)
    Task { @MainActor in
      self.debugFaceDetected = true
      self.debugPitch = pitch
      self.debugYaw = yaw
      self.debugRoll = roll
      self.debugEAR = avgEAR
      self.postureCollapse = isCollapsed
      self.attentionState = committed
      self.eyesOnRoad = committed.isEyesOnRoad
      if isEyesClosed {
        if self.eyesClosedStart == nil {
          self.eyesClosedStart = Date()
        }
        if let start = self.eyesClosedStart {
          self.eyesClosedSec =
            Date().timeIntervalSince(start)
        }
      } else {
        self.eyesClosedStart = nil
        self.eyesClosedSec = 0.0
      }
      // Deliberately excludes .faceNotDetected/.uncertain: those mean "we
      // don't know where the driver is looking," which must stay a
      // CONFIDENCE problem (handled separately, via roadEvidenceFresh-
      // style confidence weighting in sendCurrentSnapshot/MultimodalRiskFusion),
      // never escalated into a growing distraction-SEVERITY duration the
      // way a confirmed lookingDown/lookingLeft/etc. state is.
      let isConfirmedOffRoad = !committed.isEyesOnRoad && committed != .faceNotDetected && committed != .uncertain
      if isConfirmedOffRoad {
        if self.offRoadGazeStart == nil {
          self.offRoadGazeStart = Date()
        }
        if let start = self.offRoadGazeStart {
          self.offRoadGazeSec = Date().timeIntervalSince(start)
        }
      } else {
        self.offRoadGazeStart = nil
        self.offRoadGazeSec = 0.0
      }
    }
  }
  // MARK: Road Traffic Detection
  private func processRoadTraffic(
    _ pixelBuffer: CVPixelBuffer
  ) {
    // Real, trained object detector (roadObjectDetector, a Core-ML-
    // exported YOLOv8n -- see its own doc comment above for how it was
    // verified). Replaces the previous generic-rectangle "is this
    // specifically a car" guess, which had no actual evidence a box was
    // a vehicle rather than a sign, a building, or a shadow -- that
    // fabricated the traffic-density signal removed earlier (see
    // docs/FINAL_MULTIMODAL_RESULTS.md). Only vehicle-class detections
    // feed the lead-object tracker; pedestrians stay on the existing,
    // separately-validated VNDetectHumanRectanglesRequest below.
    var objectRequest: VNCoreMLRequest?
    if let detectorModel = roadObjectDetector {
      let request = VNCoreMLRequest(model: detectorModel) {
        [weak self] req, _ in
        guard let self = self else {
          return
        }
        let results =
          req.results as? [VNRecognizedObjectObservation] ?? []
        Task { @MainActor in
          let vehicleDetections = results.compactMap { observation -> Detection? in
            guard let topLabel = observation.labels.first,
              Self.vehicleClassLabels.contains(topLabel.identifier)
            else {
              return nil
            }
            return Detection(
              classLabel: topLabel.identifier,
              box: NormalizedRect(
                x: observation.boundingBox.origin.x, y: observation.boundingBox.origin.y,
                width: observation.boundingBox.width, height: observation.boundingBox.height
              ),
              confidence: topLabel.confidence
            )
          }
          let tracks = self.vehicleTracker.update(
            detections: vehicleDetections, at: Date().timeIntervalSince1970
          )
          self.leadTrack = tracks.filter { Self.vehicleClassLabels.contains($0.classLabel) }
            .max { $0.box.area < $1.box.area }
        }
      }
      // .scaleFit letterboxes rather than center-cropping -- a dashcam-
      // style wide frame has relevant vehicles near the edges that a
      // center crop could cut off entirely.
      request.imageCropAndScaleOption = .scaleFit
      objectRequest = request
    }
    // Pedestrians: unlike the car heuristic above, VNDetectHumanRectanglesRequest
    // is a real Apple-trained on-device detector, not a shape guess, so it's
    // trustworthy enough to actually factor into risk. This previously didn't
    // exist at all -- a pedestrian in or near the road had no effect on the
    // score no matter how obviously they were in frame.
    let pedestrianRequest = VNDetectHumanRectanglesRequest {
      [weak self] req, _ in
      guard let self = self else {
        return
      }
      let people =
        req.results as? [VNHumanObservation] ?? []
      Task { @MainActor in
        self.pedestrianCount = people.count
        self.pedestriansDetected = !people.isEmpty
        // Phase 7: track real pedestrian detections (unlike the vehicle
        // rectangle heuristic above, VNDetectHumanRectanglesRequest IS a
        // validated Apple-trained detector) and classify the single most
        // severe tracked pedestrian's hazard state.
        let detections = people.map {
          Detection(
            classLabel: "person",
            box: NormalizedRect(
              x: $0.boundingBox.origin.x, y: $0.boundingBox.origin.y,
              width: $0.boundingBox.width, height: $0.boundingBox.height
            ),
            confidence: $0.confidence
          )
        }
        let tracks = self.pedestrianTracker.update(
          detections: detections, at: Date().timeIntervalSince1970
        )
        self.pedestrianHazardState = tracks
          .map { classifyPedestrianHazard(track: $0) }
          .max { $0.severityRank < $1.severityRank } ?? .insufficientEvidence
      }
    }
    // upperBodyOnly defaults to true (torso-only boxes); pedestrians seen
    // from a dashboard camera are usually visible full-length, and
    // full-body detection needs revision 2 explicitly set to take effect.
    pedestrianRequest.upperBodyOnly = false
    if #available(iOS 15.0, *) {
      pedestrianRequest.revision = VNDetectHumanRectanglesRequestRevision2
    }
    var requests: [VNRequest] = [pedestrianRequest]
    if let objectRequest = objectRequest {
      requests.append(objectRequest)
    }
    try? VNSequenceRequestHandler().perform(
      requests,
      on: pixelBuffer,
      orientation: .up
    )
  }
}
// MARK: - Background Capture Service
public class BackgroundCaptureService:
  NSObject,
  ObservableObject,
  CLLocationManagerDelegate,
  ObdDelegate,
  WhoopDelegate,
  AmazfitDelegate
{
  private lazy var locationManager = CLLocationManager()
  private lazy var motionManager = CMMotionManager()
  public let obdClient = ObdBleClient()
  public let whoopClient = WhoopClient()
  // Phase 15: UNVERIFIED on physical hardware -- see AmazfitClient.swift's
  // own doc comment for exactly what is and isn't confirmed.
  public let amazfitClient = AmazfitClient()
  public let visionManager = VisionManager()
  // MARK: On-Device Core ML
  private let edgeML = RoadSenseOnDeviceRiskEngine()
  // MARK: Independent Collision Detection
  // Deliberately separate from edgeML above -- this must keep running and
  // reacting even if the risk engine, camera pipeline, or network is
  // unavailable. See CollisionDetector.swift.
  private let collisionDetector = CollisionDetector()
  private let incidentStore = CollisionIncidentStore.shared
  private var incidentRetryTimer: Timer?
  private var lastSpeedMps: Double?
  private var lastSpeedSource: String?
  private var lastTime: Double = 0.0
  private var lastGPSUpdateAt: TimeInterval?
  private var lastOBDSpeedAt: TimeInterval?
  // MARK: Driving Detection
  private var movementBeganAt: TimeInterval?
  private var lastMovementAt: TimeInterval?
  private var drivingConfirmed = false
  private let motionQueue = OperationQueue()
  private let syncQueue = DispatchQueue(
    label: "ae.aus.roadsense.syncQueue"
  )
  @Published public private(set) var streamId: String = ""
  @Published public private(set) var isRunning: Bool = false
  @Published public private(set) var isObdConnected: Bool = false
  @Published public private(set) var isWhoopConnected: Bool = false
  @Published public private(set) var isAmazfitConnected: Bool = false
  // All four required sources (OBD-II, a wearable, the front camera, and
  // the rear camera) must be connected for risk estimation to run, for
  // the whole session, not only at the start. Empty once every source
  // is present; otherwise lists exactly which ones are still missing, so
  // the UI can show "Waiting for: X" instead of a risk score computed
  // from partial evidence.
  @Published public private(set) var waitingForSources: [String] = []
  // A UI can observe this to present an immediate "did you crash?"
  // confirm/dismiss prompt, independent of polling the server's
  // pending_collision status.
  @Published public private(set) var suspectedCollision: CollisionSuspicion?
  // Phase 6: classified here, not in VisionManager, because classification
  // needs ego speed (vehicle/GPS), which is only known at this layer.
  @Published public private(set) var forwardHazardState: ForwardHazardState = .noRelevantLeadObject
  public var serverUrl: String = ServerConfig.activeBaseUrl
  private var currentFrame = SensorFrame(streamId: "")
  private var streamTimer: Timer?
  public override init() {
    super.init()
    motionQueue.name = "ae.aus.roadsense.motionQueue"
    motionQueue.qualityOfService = .utility
    obdClient.delegate = self
    whoopClient.delegate = self
    amazfitClient.delegate = self
  }
  // MARK: Start
  public func start(
    withStreamId streamId: String? = nil,
    userId: Int? = nil
  ) {
    guard !isRunning else {
      return
    }
    self.streamId = streamId ?? UUID().uuidString
    self.currentFrame = SensorFrame(
      streamId: self.streamId
    )
    self.currentFrame.userId = userId
    self.lastSpeedMps = nil
    self.lastSpeedSource = nil
    self.lastTime = 0.0
    self.lastGPSUpdateAt = nil
    self.lastOBDSpeedAt = nil
    self.movementBeganAt = nil
    self.lastMovementAt = nil
    self.drivingConfirmed = false
    DispatchQueue.main.async {
      self.isRunning = true
      self.isObdConnected = false
      self.isWhoopConnected = false
      self.isAmazfitConnected = false
      // Re-present any crash the app never finished showing (e.g. it was
      // killed right after detection) -- without creating a new detection,
      // and never for one already confirmed/dismissed.
      if self.suspectedCollision == nil, let unresolved = self.incidentStore.unresolvedSuspected().last {
        self.suspectedCollision = CollisionSuspicion(
          id: unresolved.id, accelMagnitudeMs2: unresolved.accelMagnitudeMs2,
          score: unresolved.accelMagnitudeMs2 / CollisionDetector.thresholdMs2,
          onsetEstimate: unresolved.onsetEstimate, modelVersion: unresolved.modelVersion
        )
      }
    }
    incidentRetryTimer?.invalidate()
    DispatchQueue.main.async { [weak self] in
      guard let self = self else {
        return
      }
      self.incidentRetryTimer = Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { [weak self] _ in
        self?.syncQueue.async {
          self?.retryPendingIncidents()
        }
      }
    }
    // MARK: GPS
    locationManager.delegate = self
    locationManager.desiredAccuracy =
      kCLLocationAccuracyBestForNavigation
    locationManager.requestWhenInUseAuthorization()
    locationManager.startUpdatingLocation()
    // MARK: Motion Sensors
    if motionManager.isDeviceMotionAvailable {
      motionManager.deviceMotionUpdateInterval = 0.1
      motionManager.startDeviceMotionUpdates(
        to: motionQueue
      ) { [weak self] motion, _ in
        guard let self = self,
          let m = motion
        else {
          return
        }
        self.syncQueue.async {
          let now = Date().timeIntervalSince1970
          // User acceleration excludes gravity.
          let ax = m.userAcceleration.x * 9.80665
          let ay = m.userAcceleration.y * 9.80665
          let az = m.userAcceleration.z * 9.80665
          let gx = m.rotationRate.x
          let gy = m.rotationRate.y
          let gz = m.rotationRate.z
          self.currentFrame.accelX = ax
          self.currentFrame.accelY = ay
          self.currentFrame.accelZ = az
          self.currentFrame.gyroX = gx
          self.currentFrame.gyroY = gy
          self.currentFrame.gyroZ = gz
          // Recombine gravity for the total
          // acceleration magnitude expected by
          // the existing VZCrash feature pipeline.
          let totalX =
            (m.userAcceleration.x + m.gravity.x) * 9.80665
          let totalY =
            (m.userAcceleration.y + m.gravity.y) * 9.80665
          let totalZ =
            (m.userAcceleration.z + m.gravity.z) * 9.80665
          self.currentFrame.accelMag = sqrt(
            totalX * totalX + totalY * totalY + totalZ * totalZ
          )
          self.currentFrame.gyroMag = sqrt(
            gx * gx + gy * gy + gz * gz
          )
          // MARK: Independent Collision Detection
          // Evaluated on every raw motion sample (10Hz), immediately, in
          // this same callback -- NOT in sendCurrentSnapshot below, which
          // only runs every 0.3s and is entangled with edgeML.predictRisk.
          // A real impact must be caught and reacted to even if the risk
          // engine or the next upload tick never happens.
          if let suspicion = self.collisionDetector.evaluate(
            accelMagnitudeMs2: self.currentFrame.accelMag ?? 0.0,
            at: now
          ) {
            // Saved to local storage FIRST, synchronously, before any
            // network attempt -- the phone must never depend on the server
            // to preserve this event. Everything downstream (the UI prompt,
            // the upload, retries) reads from this same stored record.
            self.incidentStore.record(
              suspicion, streamId: self.streamId, accelX: totalX, accelY: totalY, accelZ: totalZ
            )
            // VZCrash-model discrimination evidence (master directive
            // section 7: "improve discrimination between suspected
            // collisions and ordinary events"), attached AFTER the
            // threshold-trigger incident already exists -- this never
            // gates whether the incident is recorded, only adds
            // supporting evidence to it. accelObd is passed as 0.0 here
            // (not available at this 10Hz callback without duplicating
            // sendCurrentSnapshot's own speed-delta state-tracking);
            // effectiveAccel/gyroMag/speedKmh are the three features
            // that actually matter for this classification, since a
            // true collision always shows an extreme accelMag/gyroMag
            // spike regardless of accelObd.
            if let evidence = self.edgeML.classifyCollisionEvidence(
              effectiveAccel: self.currentFrame.accelMag ?? 0.0,
              gyroMag: self.currentFrame.gyroMag ?? 0.0,
              speedKmh: self.currentFrame.vehicleSpeed ?? 0.0,
              accelObd: 0.0
            ) {
              self.incidentStore.recordCollisionEvidence(
                id: suspicion.id, predictedClass: evidence.predictedClass.rawValue,
                confidence: evidence.confidence
              )
            }
            DispatchQueue.main.async {
              self.suspectedCollision = suspicion
            }
            self.reportCollisionImmediately(
              suspicion, accelX: totalX, accelY: totalY, accelZ: totalZ
            )
          }
        }
      }
    }
    // MARK: External Sensors
    obdClient.start()
    whoopClient.startPolling()
    amazfitClient.startPolling()
    visionManager.startCamera()
    // MARK: Streaming
    DispatchQueue.main.async { [weak self] in
      guard let self = self else {
        return
      }
      self.streamTimer = Timer.scheduledTimer(
        withTimeInterval: 0.3,
        repeats: true
      ) { [weak self] _ in
        self?.sendCurrentSnapshot()
      }
    }
  }
  // MARK: Sensor Snapshot and Risk Inference
  private func sendCurrentSnapshot() {
    syncQueue.async { [weak self] in
      guard let self = self else {
        return
      }
      let now = Date().timeIntervalSince1970
      var snapshot = self.currentFrame
      snapshot.timestamp = now
      // MARK: Environmental Context
      if let lat = snapshot.gpsLatitude,
        let lon = snapshot.gpsLongitude
      {
        ContextManager.shared.updateLocation(
          lat: lat,
          lon: lon
        )
      }
      // MARK: Vision Snapshot
      var leadTrackSnapshot: TrackedObject?
      DispatchQueue.main.sync {
        leadTrackSnapshot = self.visionManager.leadTrack
        snapshot.eyesOnRoad =
          self.visionManager.eyesOnRoad
        snapshot.attentionState =
          self.visionManager.attentionState.payloadValue
        snapshot.eyesClosedSec =
          self.visionManager.eyesClosedSec
        snapshot.offRoadGazeSec =
          self.visionManager.offRoadGazeSec
        snapshot.postureCollapse =
          self.visionManager.postureCollapse
        snapshot.leadClosingFast =
          self.visionManager.leadClosingFast
        snapshot.trafficLevel =
          self.visionManager.trafficLevel
        snapshot.pedestriansDetected =
          self.visionManager.pedestriansDetected
        snapshot.pedestrianCount =
          self.visionManager.pedestrianCount
        snapshot.pedestrianHazardState =
          self.visionManager.pedestrianHazardState.payloadValue
        snapshot.cameraStatus =
          self.visionManager.cameraStatus
        snapshot.debugFaceDetected =
          self.visionManager.debugFaceDetected
        snapshot.debugPitch =
          self.visionManager.debugPitch
        snapshot.debugYaw =
          self.visionManager.debugYaw
        snapshot.debugRoll =
          self.visionManager.debugRoll
        snapshot.debugEAR =
          self.visionManager.debugEAR
      }
      snapshot.weatherMultiplier = ContextManager.shared.weatherMultiplier
      snapshot.weatherConditionCode = ContextManager.shared.weatherConditionCode
      snapshot.weatherIsLive = ContextManager.shared.isWeatherLiveData
      // MARK: Speed Normalization
      // OBD-II speed is stored in km/h.
      // GPS speed is stored in m/s.
      // Convert both to m/s before calculations.
      let obdFresh =
        snapshot.isObdConnected
        && (self.lastOBDSpeedAt.map {
          now - $0 <= 4.0
        } ?? false)
      let gpsFresh =
        self.lastGPSUpdateAt.map {
          now - $0 <= 5.0
        } ?? false
      let obdSpeedMps =
        obdFresh
        ? snapshot.vehicleSpeed.map { $0 / 3.6 }
        : nil
      let gpsSpeedMps =
        gpsFresh
        ? snapshot.gpsSpeed
        : nil
      let measuredSpeedMps =
        obdSpeedMps ?? gpsSpeedMps
      let speedSource: String? =
        obdSpeedMps != nil
        ? "obd"
        : (gpsSpeedMps != nil ? "gps" : nil)
      let speedMps = measuredSpeedMps ?? 0.0
      let speedKmh = speedMps * 3.6
      // MARK: Forward Collision Estimation (Phase 6)
      // nil (not 0.0) when speed is genuinely unknown, so
      // classifyForwardHazard's "insufficient context" path is distinct
      // from "measured, actually near zero."
      let hazard = classifyForwardHazard(
        leadTrack: leadTrackSnapshot,
        egoSpeedKmh: measuredSpeedMps != nil ? speedKmh : nil
      )
      snapshot.forwardHazardState = hazard.payloadValue
      // Backward-compatible boolean, now actually derived instead of the
      // hardcoded `false` VisionManager still publishes on its own (it
      // lacks speed context to know this itself) -- only the strongest
      // hazard state counts as "closing fast."
      snapshot.leadClosingFast = (hazard == .increasingClosingHazard)
      DispatchQueue.main.async { [weak self] in
        self?.forwardHazardState = hazard
      }
      // MARK: Driving State Detection
      // Require at least 8 km/h for 3 seconds
      // before activating on-device predictions.
      if measuredSpeedMps != nil
        && speedKmh >= 8.0
      {
        if self.movementBeganAt == nil {
          self.movementBeganAt = now
        }
        self.lastMovementAt = now
        if let began = self.movementBeganAt,
          now - began >= 3.0,
          !self.drivingConfirmed
        {
          self.drivingConfirmed = true
          // Discard any eye-closed/distraction timer that accumulated
          // while parked so it can't immediately trip a hard override
          // the moment the trip is confirmed.
          self.visionManager.resetDrivingState()
        }
      } else {
        self.movementBeganAt = nil
        // Keep the trip confirmed for a brief
        // stop, such as a traffic light.
        if self.lastMovementAt.map({
          now - $0 > 45.0
        }) ?? true {
          self.drivingConfirmed = false
        }
      }
      // NOTE: intentionally does NOT also require
      // measuredSpeedMps != nil here. A single frame where GPS/OBD
      // speed is momentarily stale (common right as speed drops near
      // a stop) used to flip this false for just that one tick, which
      // made RoadSenseOnDeviceRiskEngine wipe its rolling accel/HR
      // history (see its `isDriving` guard). The history buffer would
      // then rebuild from only 1-2 samples spanning a few hundred ms,
      // and computing "jerk" as (last-first)/(last.time-first.time)
      // over that tiny window amplified ordinary braking into a huge
      // spurious jerk value -- the real cause of a ~40% risk score
      // during a perfectly gentle stop. drivingConfirmed already has
      // its own 45s grace period for brief stops, so that alone is the
      // right signal for "is this trip still active".
      // MARK: Required-Source Gate
      // All four sources, OBD-II, wearable, front camera, and rear camera,
      // must be connected for risk estimation to run, for the whole
      // session and not only at its start. Losing any one, at any point,
      // pauses risk output until it reconnects, rather than falling back
      // to a score fused from whatever evidence remains.
      var waiting: [String] = []
      if !obdFresh { waiting.append("OBD-II adapter") }
      if !(self.isWhoopConnected || self.isAmazfitConnected) { waiting.append("wearable") }
      if snapshot.cameraStatus != "running" { waiting.append("camera") }
      DispatchQueue.main.async { [weak self] in
        self?.waitingForSources = waiting
      }

      let riskActive = self.drivingConfirmed && waiting.isEmpty
      // MARK: Speed-Change Acceleration
      var obdAccel = 0.0
      if let previous = self.lastSpeedMps,
        speedSource != nil,
        speedSource == self.lastSpeedSource,
        self.lastTime > 0,
        now > self.lastTime
      {
        obdAccel =
          (speedMps - previous)
          / (now - self.lastTime)
      }
      self.lastSpeedMps = measuredSpeedMps
      self.lastSpeedSource = speedSource
      self.lastTime = now
      // MARK: On-Device Core ML Prediction
      // Phase 9.1: predictRiskBreakdown (not predictRisk) so the
      // interpretable components -- raw ML fusion vs. each hand-authored
      // heuristic adjustment -- are available to send to the dashboard,
      // not just the final opaque number. See RiskFusionHeuristics.swift.
      let breakdown = self.edgeML.predictRiskBreakdown(
        effectiveAccel: snapshot.accelMag ?? 9.80665,
        gyroMag: snapshot.gyroMag ?? 0.0,
        speedKmh: speedKmh,
        accelObd: obdAccel,
        // Passed through as true optionals, NOT collapsed to 0.0 here --
        // an unconnected OBD adapter or wearable must read as genuinely
        // missing to RoadSenseOnDeviceRiskEngine, not as "reading zero",
        // which for RPM/throttle/engine load would otherwise look like a
        // stopped engine while the car is actually moving.
        rpm: snapshot.rpm,
        throttle: snapshot.throttlePosition,
        load: snapshot.engineLoad,
        hr: snapshot.heartRate,
        eyesOnRoad: snapshot.eyesOnRoad,
        eyesClosedSec: snapshot.eyesClosedSec,
        trafficLevel: snapshot.trafficLevel,
        pedestriansDetected: snapshot.pedestriansDetected,
        leadClosingFast: snapshot.leadClosingFast,
        inRoundabout:
          ContextManager.shared.isRoundabout,
        weatherMult:
          ContextManager.shared.weatherMultiplier,
        weatherCode:
          ContextManager.shared.weatherConditionCode,
        designSpeed:
          ContextManager.shared.designSpeed,
        isDriving: riskActive,
        postureCollapse: snapshot.postureCollapse
      )
      // MARK: Multimodal Risk Fusion (master directive section 6)
      // Groups the same breakdown components already computed above
      // into the three principal evidence dimensions, with
      // confidence/evidence-quality derived from whether each
      // modality's underlying data is actually fresh right now --
      // missing OBD/GPS, an undetected face, or no rear-camera hazard
      // evidence must lower confidence, NOT silently read as "safe."
      // See MultimodalRiskFusion.swift.
      let vehicleEvidence = ModalityEvidence(
        value: breakdown.mlFusionBase + breakdown.jerkBoost + breakdown.hrVelocityBoost,
        confidence: measuredSpeedMps != nil ? 1.0 : 0.4
      )
      let lowConfidenceAttentionStates: Set<String> = ["face_not_detected", "uncertain"]
      // Severity ramps with how long the gaze has been SUSTAINED off-road
      // (offRoadGazeSec, which already excludes brief hysteresis-filtered
      // glances and the face_not_detected/uncertain confidence case) --
      // a distraction that has lasted 4 seconds is more severe than one
      // that just started, which the previous instantaneous-only penalty
      // could not express. Capped at 2x so persistence alone can't run
      // away unboundedly.
      let offRoadSeverityMultiplier = 1.0 + min(1.0, snapshot.offRoadGazeSec / 5.0)
      let attentionEvidence = ModalityEvidence(
        value: (breakdown.eyesClosedPenalty + breakdown.attentionPenalty) * offRoadSeverityMultiplier,
        confidence: lowConfidenceAttentionStates.contains(snapshot.attentionState) ? 0.2 : 1.0
      )
      let noRoadEvidenceStates: Set<String> = ["no_relevant_lead_object", "insufficient_evidence"]
      let roadEvidenceFresh = !(noRoadEvidenceStates.contains(snapshot.forwardHazardState)
        && noRoadEvidenceStates.contains(snapshot.pedestrianHazardState))
      let roadEvidence = ModalityEvidence(
        value: breakdown.trafficContext + breakdown.pedestrianContext + breakdown.leadClosingHardBoost,
        confidence: roadEvidenceFresh ? 1.0 : 0.3
      )
      let multimodalEstimate = fuseMultimodalRisk(
        vehicle: vehicleEvidence, attention: attentionEvidence, road: roadEvidence,
        fiveSecondVehicleHazardProb: breakdown.fiveSecondAheadVehicleHazardProb,
        contextualAdjustment: 0.0  // weather is already folded into mlFusionBase via weatherMult
      )
      // The 5-second multimodal estimate (vehicle forecast + live
      // attention/road evidence) is now the one authoritative risk score --
      // breakdown.total is no longer reported as a second, competing
      // number. The one exception: breakdown.hardOverrideApplied means
      // eyes have been closed for 3+ seconds, an absolute safety tripwire
      // (RiskFusionHeuristics.swift), not a competing prediction -- that
      // floor must never be diluted by the multimodal blend, so it wins
      // outright rather than being averaged in.
      let risk = breakdown.hardOverrideApplied
        ? breakdown.total
        : (multimodalEstimate.fiveSecondAheadRisk ?? breakdown.total)
      // MARK: Server Payload
      var payload = snapshot.toDictionary()
      payload["risk_active"] = riskActive
      // Dashboard-only explainability, same status as camera_status/debug_*
      // elsewhere in this payload -- NOT part of the validated SensorFrame
      // schema, and the index itself remains experimental/uncalibrated
      // regardless of how it's broken down (see RiskFusionHeuristics.swift).
      if riskActive {
        payload["risk_breakdown"] = [
          "ml_fusion_base": breakdown.mlFusionBase,
          "jerk_boost": breakdown.jerkBoost,
          "hr_velocity_boost": breakdown.hrVelocityBoost,
          "traffic_context": breakdown.trafficContext,
          "pedestrian_context": breakdown.pedestrianContext,
          "eyes_closed_penalty": breakdown.eyesClosedPenalty,
          "attention_penalty": breakdown.attentionPenalty,
          "lead_closing_hard_boost": breakdown.leadClosingHardBoost,
          "false_positive_suppression_applied": breakdown.falsePositiveSuppressionApplied,
          "hard_override_applied": breakdown.hardOverrideApplied,
          "collision_support_evidence": breakdown.collisionSupportEvidence,
          "off_road_gaze_sec": snapshot.offRoadGazeSec,
        ]
        payload["multimodal_risk"] = [
          "current_risk": multimodalEstimate.currentRisk,
          "five_second_ahead_risk": multimodalEstimate.fiveSecondAheadRisk as Any,
          "risk_level": multimodalEstimate.riskLevel.rawValue,
          "vehicle_contribution": multimodalEstimate.vehicleContribution,
          "attention_contribution": multimodalEstimate.attentionContribution,
          "road_hazard_contribution": multimodalEstimate.roadHazardContribution,
          "contextual_adjustment": multimodalEstimate.contextualAdjustment,
          "evidence_quality": multimodalEstimate.evidenceQuality,
          "dominant_factor": multimodalEstimate.dominantFactor,
        ]
      }
      // Inactive driving is not a low risk score.
      // Use JSON null to represent no prediction.
      if riskActive {
        payload["risk_score"] = risk
      } else {
        payload["risk_score"] = NSNull()
      }
      // MARK: HTTP Upload
      guard
        let url = URL(
          string: "\(self.serverUrl)/api/v1/ingest"
        )
      else {
        return
      }
      var request = URLRequest(url: url)
      request.httpMethod = "POST"
      request.setValue(
        "application/json",
        forHTTPHeaderField: "Content-Type"
      )
      request.httpBody = try? JSONSerialization.data(
        withJSONObject: payload,
        options: []
      )
      URLSession.shared.dataTask(
        with: request
      ).resume()
    }
  }
  // MARK: Independent Collision Reporting
  // Fires immediately on detection, out-of-band from the 0.3s
  // sendCurrentSnapshot cycle, carrying the EXACT triggering sample -- the
  // regular upload cadence might decimate past it. Best-effort and never
  // blocking: the on-device reaction (storing locally, updating
  // suspectedCollision) already happened synchronously before this is
  // called. If it fails here (offline, server down), incidentRetryTimer
  // below will retry it from the local store -- client_incident_id makes a
  // retry, however late, resolve to the SAME server-side incident rather
  // than creating a duplicate.
  private func reportCollisionImmediately(
    _ suspicion: CollisionSuspicion, accelX: Double, accelY: Double, accelZ: Double
  ) {
    uploadIncident(id: suspicion.id, streamId: self.streamId, timestamp: suspicion.onsetEstimate, accelX: accelX, accelY: accelY, accelZ: accelZ)
  }

  private func uploadIncident(
    id: String, streamId: String, timestamp: Double, accelX: Double, accelY: Double, accelZ: Double
  ) {
    guard let url = URL(string: "\(self.serverUrl)/api/v1/ingest") else {
      return
    }
    let payload: [String: Any] = [
      "timestamp": timestamp,
      "stream_id": streamId,
      "source": "ios",
      "accel_x": accelX,
      "accel_y": accelY,
      "accel_z": accelZ,
      "client_incident_id": id,
    ]
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try? JSONSerialization.data(withJSONObject: payload, options: [])
    URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
      guard let self = self, error == nil,
        let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode),
        let data = data,
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let alertId = json["collision_alert_id"] as? Int
      else {
        return  // left un-uploaded; the retry timer will try again
      }
      self.incidentStore.markUploaded(id: id, serverAlertId: alertId)
    }.resume()
  }

  /// Retries anything the local store still has pending: an incident whose
  /// initial upload failed (offline, server down), and any confirm/dismiss
  /// decision made locally that the server doesn't know about yet. Runs on
  /// the SAME 0.3s streamTimer-adjacent cadence used elsewhere in this
  /// file, not a new high-frequency loop.
  private func retryPendingIncidents() {
    for incident in incidentStore.pendingUpload() {
      uploadIncident(
        id: incident.id, streamId: incident.streamId, timestamp: incident.onsetEstimate,
        accelX: incident.accelX, accelY: incident.accelY, accelZ: incident.accelZ
      )
    }
    for incident in incidentStore.pendingDecisionSync() {
      guard let alertId = incident.serverAlertId else {
        continue
      }
      let action = incident.status == .confirmed ? "confirm" : "dismiss"
      syncDecision(alertId: alertId, action: action, incidentId: incident.id)
    }
  }

  // AuthManager.token is @MainActor-isolated. This used to be read with
  // `DispatchQueue.main.sync { AuthManager.shared.token }`, which deadlocks
  // the instant the CALLER is already on the main thread -- exactly what
  // confirmSuspectedCollision/dismissSuspectedCollision below do, since
  // they're invoked directly from SwiftUI button actions (see
  // ContentView.swift's onConfirm/onDismiss closures). Tapping Confirm or
  // Dismiss on an already-uploaded incident would hang the entire UI
  // forever. `Task { await ... }` never blocks the calling thread, however
  // it's called from, so this is safe regardless of caller.
  private func syncDecision(alertId: Int, action: String, incidentId: String) {
    guard let url = URL(string: "\(self.serverUrl)/app/api/alerts/\(alertId)/\(action)") else {
      return
    }
    Task {
      guard let token = await AuthManager.shared.token else {
        return
      }
      var request = URLRequest(url: url)
      request.httpMethod = "POST"
      request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
      URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
        guard let self = self else {
          return
        }
        guard error == nil, let http = response as? HTTPURLResponse else {
          return  // network/transport failure -- transient, retried next cycle
        }
        if (200...299).contains(http.statusCode) {
          self.incidentStore.markDecisionSynced(id: incidentId)
        } else if http.statusCode == 409 {
          // Invalid lifecycle transition -- the server has already settled
          // this incident to some other final state (e.g. dismissed via
          // the dashboard). Permanent, not transient: retrying this exact
          // request can never succeed, so stop rather than retry forever.
          self.incidentStore.markDecisionRejected(id: incidentId)
        }
        // Any other status (401/403/5xx): left un-synced, retried next
        // cycle -- those CAN plausibly resolve (re-auth, server recovers).
      }.resume()
    }
  }

  /// Driver confirms a suspected collision. Updates local storage
  /// IMMEDIATELY -- this works with zero network connectivity -- and then
  /// attempts to tell the server; a failure there is retried later, never
  /// blocking the local confirmation.
  public func confirmSuspectedCollision(id: String) {
    incidentStore.confirm(id: id)
    if suspectedCollision?.id == id {
      suspectedCollision = nil
    }
    if let incident = incidentStore.incident(id: id), let alertId = incident.serverAlertId {
      syncDecision(alertId: alertId, action: "confirm", incidentId: id)
    }
  }

  /// Driver dismisses a false alarm. Same offline-first contract as confirm.
  public func dismissSuspectedCollision(id: String) {
    incidentStore.dismiss(id: id)
    if suspectedCollision?.id == id {
      suspectedCollision = nil
    }
    if let incident = incidentStore.incident(id: id), let alertId = incident.serverAlertId {
      syncDecision(alertId: alertId, action: "dismiss", incidentId: id)
    }
  }

  // MARK: Location Updates
  public func locationManager(
    _ manager: CLLocationManager,
    didUpdateLocations locations: [CLLocation]
  ) {
    guard let loc = locations.last else {
      return
    }
    syncQueue.async { [weak self] in
      guard let self = self else {
        return
      }
      if CLLocationCoordinate2DIsValid(loc.coordinate),
        loc.horizontalAccuracy >= 0,
        loc.horizontalAccuracy <= 100
      {
        self.currentFrame.gpsLatitude =
          loc.coordinate.latitude
        self.currentFrame.gpsLongitude =
          loc.coordinate.longitude
      }
      // Reject inaccurate or outdated GPS speed.
      let age = abs(
        Date().timeIntervalSince(loc.timestamp)
      )
      if loc.speed.isFinite,
        loc.speed >= 0,
        loc.speedAccuracy >= 0,
        loc.speedAccuracy <= 1.5,
        loc.horizontalAccuracy >= 0,
        loc.horizontalAccuracy <= 100,
        age <= 5.0
      {
        self.currentFrame.gpsSpeed = loc.speed
        self.lastGPSUpdateAt =
          Date().timeIntervalSince1970
      } else {
        self.currentFrame.gpsSpeed = nil
        self.lastGPSUpdateAt = nil
      }
      if loc.course >= 0 {
        self.currentFrame.gpsHeading =
          loc.course * (.pi / 180.0)
      }
    }
  }
  public func locationManagerDidChangeAuthorization(
    _ manager: CLLocationManager
  ) {
    let status = manager.authorizationStatus
    if status == .authorizedWhenInUse
      || status == .authorizedAlways
    {
      if status == .authorizedWhenInUse {
        manager.requestAlwaysAuthorization()
      }
      locationManager.startUpdatingLocation()
    }
  }
  // MARK: OBD Connection
  public func didChangeConnectionStatus(
    isConnected: Bool
  ) {
    syncQueue.async { [weak self] in
      guard let self = self else {
        return
      }
      self.currentFrame.isObdConnected = isConnected
      if !isConnected {
        self.currentFrame.vehicleSpeed = nil
        self.lastOBDSpeedAt = nil
      }
      DispatchQueue.main.async {
        self.isObdConnected = isConnected
      }
    }
  }
  // MARK: OBD Telemetry
  public func didUpdateObdData(
    speed: Double?,
    rpm: Double?,
    throttle: Double?,
    load: Double?
  ) {
    syncQueue.async { [weak self] in
      guard let self = self else {
        return
      }
      if let r = rpm, r > 0 {
        self.currentFrame.isObdConnected = true
        self.currentFrame.rpm = r
        if let s = speed,
          s.isFinite,
          s >= 0
        {
          // OBD delegate provides m/s.
          // SensorFrame stores km/h.
          self.currentFrame.vehicleSpeed = s * 3.6
          self.lastOBDSpeedAt =
            Date().timeIntervalSince1970
        }
        if let t = throttle {
          self.currentFrame.throttlePosition = t
        }
        if let l = load {
          self.currentFrame.engineLoad = l
        }
        DispatchQueue.main.async {
          self.isObdConnected = true
        }
      } else if let s = speed,
        s.isFinite,
        s >= 0
      {
        self.currentFrame.vehicleSpeed = s * 3.6
        self.lastOBDSpeedAt =
          Date().timeIntervalSince1970
        if let r = rpm {
          self.currentFrame.rpm = r
        }
        if let t = throttle {
          self.currentFrame.throttlePosition = t
        }
        if let l = load {
          self.currentFrame.engineLoad = l
        }
      }
    }
  }
  // MARK: WHOOP Telemetry
  public func didUpdateWhoop(
    heartRate: Double?,
    rrInterval: Double?
  ) {
    syncQueue.async { [weak self] in
      guard let self = self else {
        return
      }
      if let hr = heartRate,
        hr > 0
      {
        self.currentFrame.heartRate = hr
        self.currentFrame.isWhoopConnected = true
        DispatchQueue.main.async {
          self.isWhoopConnected = true
        }
      }
    }
  }
  // MARK: Amazfit Telemetry
  // Merges into the SAME currentFrame.heartRate field WHOOP uses -- this
  // app has no per-sensor source attribution for physiology readings
  // today (a pre-existing limitation, not introduced here; see
  // Phase 15's progress notes). Do not assume WHOOP and Amazfit running
  // simultaneously produces a sensible merged reading -- untested.
  public func didUpdateAmazfit(
    heartRate: Double?,
    rrInterval: Double?
  ) {
    syncQueue.async { [weak self] in
      guard let self = self else {
        return
      }
      if let hr = heartRate,
        hr > 0
      {
        self.currentFrame.heartRate = hr
        self.currentFrame.isAmazfitConnected = true
        DispatchQueue.main.async {
          self.isAmazfitConnected = true
        }
      }
    }
  }
  // MARK: Stop
  public func stop() {
    guard isRunning else {
      return
    }
    streamTimer?.invalidate()
    streamTimer = nil
    incidentRetryTimer?.invalidate()
    incidentRetryTimer = nil
    locationManager.stopUpdatingLocation()
    motionManager.stopDeviceMotionUpdates()
    obdClient.stop()
    whoopClient.stop()
    amazfitClient.stop()
    visionManager.stopCamera()
    syncQueue.async {
      self.movementBeganAt = nil
      self.lastMovementAt = nil
      self.drivingConfirmed = false
      self.lastSpeedMps = nil
      self.lastSpeedSource = nil
      self.lastGPSUpdateAt = nil
      self.lastOBDSpeedAt = nil
    }
    DispatchQueue.main.async {
      self.isRunning = false
      self.isObdConnected = false
      self.isWhoopConnected = false
      self.isAmazfitConnected = false
      self.forwardHazardState = .noRelevantLeadObject
    }
  }
}

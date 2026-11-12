import AVFoundation
import CoreLocation
import CoreMotion
import Foundation
import UIKit
import Vision

// MARK: - Vision Manager (7-Factor Multimodal Tracking)
public class VisionManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate
{
  @Published public var eyesClosedSec: Double = 0.0
  @Published public var eyesOnRoad: Bool = true
  @Published public var postureCollapse: Bool = false
  @Published public var fixedGaze: Bool = false
  @Published public var sluggishGaze: Bool = false
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
  private var fixedGazeStart: Date?
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
      self.fixedGazeStart = nil
      self.eyesClosedSec = 0.0
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
    fixedGazeStart = nil
    frontFrameTimestamps.removeAll()
    backFrameTimestamps.removeAll()
    vehicleTracker = ObjectTracker()
    Task { @MainActor in
      self.eyesOnRoad = true
      self.attentionState = .attentive
      self.eyesClosedSec = 0.0
      self.postureCollapse = false
      self.fixedGaze = false
      self.trafficLevel = "empty"
      self.leadClosingFast = false
      self.leadTrack = nil
      self.sluggishGaze = false
      self.pedestriansDetected = false
      self.pedestrianCount = 0
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
    let instantaneous = classifyGaze(
      pitch: pitch,
      yaw: yaw,
      roll: roll,
      eyeAspectRatio: avgEAR,
      faceDetected: true,
      confidence: face.confidence
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
    }
  }
  // MARK: Road Traffic Detection
  private func processRoadTraffic(
    _ pixelBuffer: CVPixelBuffer
  ) {
    // Vehicle density: still a crude aspect-ratio heuristic over generic
    // rectangles (there's no validated car detector bundled), but now
    // banded into 4 levels instead of 3 so "a couple of cars" and
    // "genuinely crowded" don't collapse into the same bucket.
    let vehicleRequest = VNDetectRectanglesRequest {
      [weak self] req, _ in
      guard let self = self else {
        return
      }
      let results =
        req.results as? [VNRectangleObservation] ?? []
      Task { @MainActor in
        let carRects = results.filter {
          $0.boundingBox.height > 0
            && ($0.boundingBox.width
              / $0.boundingBox.height) > 1.2
        }
        switch carRects.count {
        case 0:
          self.trafficLevel = "empty"
        case 1...2:
          self.trafficLevel = "light"
        case 3...5:
          self.trafficLevel = "moderate"
        default:
          self.trafficLevel = "crowded"
        }
        // Rectangle detection cannot reliably
        // identify cars or estimate closing speed.
        // Disable this collision cue until a
        // validated object detector is available.
        self.leadClosingFast = false
        // Phase 5/6: track car-like rectangles across frames and surface
        // the largest one as the lead-vehicle candidate. This inherits the
        // SAME limitation as leadClosingFast above -- these are still
        // crude aspect-ratio rectangles, not a validated detector -- which
        // is exactly why classification into an actual hazard verdict
        // (ForwardCollisionHazard.swift, applied in sendCurrentSnapshot)
        // stays conservative rather than trusting this track's identity.
        let detections = carRects.map {
          Detection(
            classLabel: "car",
            box: NormalizedRect(
              x: $0.boundingBox.origin.x, y: $0.boundingBox.origin.y,
              width: $0.boundingBox.width, height: $0.boundingBox.height
            ),
            confidence: $0.confidence
          )
        }
        let tracks = self.vehicleTracker.update(
          detections: detections, at: Date().timeIntervalSince1970
        )
        self.leadTrack = tracks.filter { $0.classLabel == "car" }
          .max { $0.box.area < $1.box.area }
      }
    }
    vehicleRequest.minimumSize = 0.1
    vehicleRequest.maximumObservations = 10
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
      }
    }
    // upperBodyOnly defaults to true (torso-only boxes); pedestrians seen
    // from a dashboard camera are usually visible full-length, and
    // full-body detection needs revision 2 explicitly set to take effect.
    pedestrianRequest.upperBodyOnly = false
    if #available(iOS 15.0, *) {
      pedestrianRequest.revision = VNDetectHumanRectanglesRequestRevision2
    }
    try? VNSequenceRequestHandler().perform(
      [vehicleRequest, pedestrianRequest],
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
  WhoopDelegate
{
  private lazy var locationManager = CLLocationManager()
  private lazy var motionManager = CMMotionManager()
  public let obdClient = ObdBleClient()
  public let whoopClient = WhoopClient()
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
        snapshot.postureCollapse =
          self.visionManager.postureCollapse
        snapshot.fixedGaze =
          self.visionManager.fixedGaze
        snapshot.leadClosingFast =
          self.visionManager.leadClosingFast
        snapshot.trafficLevel =
          self.visionManager.trafficLevel
        snapshot.pedestriansDetected =
          self.visionManager.pedestriansDetected
        snapshot.pedestrianCount =
          self.visionManager.pedestrianCount
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
      let riskActive = self.drivingConfirmed
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
      let risk = self.edgeML.predictRisk(
        effectiveAccel: snapshot.accelMag ?? 9.80665,
        gyroMag: snapshot.gyroMag ?? 0.0,
        speedKmh: speedKmh,
        accelObd: obdAccel,
        rpm: snapshot.rpm ?? 0.0,
        throttle: snapshot.throttlePosition ?? 0.0,
        load: snapshot.engineLoad ?? 0.0,
        hr: snapshot.heartRate ?? 0.0,
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
        isDriving: riskActive
      )
      // MARK: Server Payload
      var payload = snapshot.toDictionary()
      payload["risk_active"] = riskActive
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
        guard let self = self, error == nil,
          let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode)
        else {
          return  // left un-synced; retried next cycle
        }
        self.incidentStore.markDecisionSynced(id: incidentId)
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
      if let rr = rrInterval {
        self.currentFrame.rrIntervalMs = rr
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
      self.forwardHazardState = .noRelevantLeadObject
    }
  }
}

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
  // Raw diagnostic values. These exist so the thresholds below can be
  // calibrated against real readings (phone mount angle, lighting) instead
  // of guessed blind. Watch these on the dashboard while driving/parked.
  @Published public var debugFaceDetected: Bool = false
  @Published public var debugPitch: Double = 0.0
  @Published public var debugYaw: Double = 0.0
  @Published public var debugRoll: Double = 0.0
  @Published public var debugEAR: Double = 0.0
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
  private var distractedStartTime: Date?
  private var fixedGazeStart: Date?
  private var noFaceStartTime: Date?
  private let distractionThresholdSeconds: TimeInterval = 1.5
  private var lastPitch: Double = 0.0
  private var lastYaw: Double = 0.0
  public override init() {
    super.init()
  }
  public func startCamera() {
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
  // Called the moment driving is confirmed (speed sustained above the
  // threshold). Eye-closed/distraction timers run continuously even while
  // parked, so without this reset, time spent looking down or with eyes
  // closed before you start moving carries straight into the trip and can
  // immediately trip the hard "prolonged eye closure" override.
  public func resetDrivingState() {
    noFaceStartTime = nil
    Task { @MainActor in
      self.eyesClosedStart = nil
      self.distractedStartTime = nil
      self.fixedGazeStart = nil
      self.eyesClosedSec = 0.0
      self.eyesOnRoad = true
    }
  }
  public func stopCamera() {
    multiCamSession?.stopRunning()
    multiCamSession = nil
    distractedStartTime = nil
    eyesClosedStart = nil
    fixedGazeStart = nil
    noFaceStartTime = nil
    Task { @MainActor in
      self.eyesOnRoad = true
      self.eyesClosedSec = 0.0
      self.postureCollapse = false
      self.fixedGaze = false
      self.trafficLevel = "empty"
      self.leadClosingFast = false
      self.sluggishGaze = false
      self.pedestriansDetected = false
      self.pedestrianCount = 0
      self.cameraStatus = "idle"
    }
  }
  private func setupMultiCam() {
    guard AVCaptureMultiCamSession.isMultiCamSupported else {
      reportCameraFailure("Multi-camera capture is not supported on this device")
      return
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
      if connection.isVideoRotationAngleSupported(90.0) {
        connection.videoRotationAngle = 90.0
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
      if connection.isVideoRotationAngleSupported(90.0) {
        connection.videoRotationAngle = 90.0
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
    if connection == frontConnection {
      processDriverFace(pixelBuffer)
    } else if connection == backConnection {
      processRoadTraffic(pixelBuffer)
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
    guard
      let face = (landmarksRequest.results as? [VNFaceObservation])?.first
    else {
      // A missing face is ambiguous on its own -- a blink, a bump,
      // or a momentary tracking miss can drop a single frame even
      // while the driver is looking straight ahead. But a face that
      // stays missing is usually the driver turned far enough away
      // (or looking down at a phone) that the front camera can no
      // longer frame it at all -- the exact case we need to flag as
      // NOT on the road. Previously this branch forced
      // eyesOnRoad = true immediately, which meant looking away hard
      // enough to defeat face detection was reported as the safest
      // possible state. Treat a sustained miss the same way sustained
      // distraction is treated below.
      Task { @MainActor in
        self.debugFaceDetected = false
        self.eyesClosedStart = nil
        self.eyesClosedSec = 0.0
        self.postureCollapse = false
        if self.noFaceStartTime == nil {
          self.noFaceStartTime = Date()
        }
        if let start = self.noFaceStartTime,
          Date().timeIntervalSince(start)
            > self.distractionThresholdSeconds
        {
          self.eyesOnRoad = false
        }
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
    let isDistracted =
      (pitch < -0.4) || (abs(yaw) > 0.45)
    let isCollapsed =
      (pitch < -0.8) || (abs(roll) > 0.7)
    let leftEAR = self.getEyeOpenness(
      eye: face.landmarks?.leftEye
    )
    let rightEAR = self.getEyeOpenness(
      eye: face.landmarks?.rightEye
    )
    let avgEAR = (leftEAR + rightEAR) / 2.0
    let isEyesClosed = avgEAR < 0.18
    Task { @MainActor in
      self.noFaceStartTime = nil
      self.debugFaceDetected = true
      self.debugPitch = pitch
      self.debugYaw = yaw
      self.debugRoll = roll
      self.debugEAR = avgEAR
      self.postureCollapse = isCollapsed
      if isDistracted {
        if self.distractedStartTime == nil {
          self.distractedStartTime = Date()
        }
        if let start = self.distractedStartTime,
          Date().timeIntervalSince(start)
            > self.distractionThresholdSeconds
        {
          self.eyesOnRoad = false
        }
      } else {
        self.distractedStartTime = nil
        self.eyesOnRoad = true
      }
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
      DispatchQueue.main.sync {
        snapshot.eyesOnRoad =
          self.visionManager.eyesOnRoad
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
    }
  }
}

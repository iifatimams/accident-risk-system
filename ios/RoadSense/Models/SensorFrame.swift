import Foundation

public struct SensorFrame: Codable {
    public var timestamp: Double
    public var streamId: String
    public var userId: Int?
    public var vehicleId: String?
    public var source: String = "ios"
    
    // Inertial (Phone)
    public var accelX: Double?
    public var accelY: Double?
    public var accelZ: Double?
    public var gyroX: Double?
    public var gyroY: Double?
    public var gyroZ: Double?
    
    // Calculated Magnitudes
    public var accelMag: Double?
    public var gyroMag: Double?
    
    // GPS
    public var gpsLatitude: Double?
    public var gpsLongitude: Double?
    public var gpsSpeed: Double?      // m/s
    public var gpsHeading: Double?    // radians
    
    // OBD-II
    public var vehicleSpeed: Double?  // km/h
    public var rpm: Double?
    public var throttlePosition: Double?
    public var engineLoad: Double?
    public var isObdConnected: Bool = false
    
    // Physiology (WHOOP)
    public var heartRate: Double?
    // The actual RR/IBI interval read from the standard BLE Heart Rate
    // Measurement characteristic (0x2A37) -- real per-beat hardware data,
    // never derived from heartRate/bpm. This is what the backend's
    // physiology_features.py RMSSD computation (rmssd_ms) is built to
    // consume; hrvRmssd/SDNN are deliberately not computed on-device (see
    // AmazfitClient.swift's own comment: no HRV is fabricated here).
    public var rrIntervalMs: Double?
    public var isWhoopConnected: Bool = false
    // Phase 15: UNVERIFIED on physical hardware -- see AmazfitClient.swift.
    public var isAmazfitConnected: Bool = false
    
    // 7-Factor Multimodal Variables
    public var eyesOnRoad: Bool = true
    // Discrete reading behind eyesOnRoad above, e.g. "lookingDown",
    // "faceNotDetected", "uncertain" -- see DriverAttentionState.swift.
    // Kept as a plain String here (not the enum) since SensorFrame is a
    // simple Codable/dictionary payload shape, matching every other field.
    public var attentionState: String = "attentive"
    public var leadClosingFast: Bool = false
    // Discrete reading behind leadClosingFast, e.g. "possible_closing_motion",
    // "insufficient_evidence" -- see ForwardCollisionHazard.swift (Phase 6).
    // Experimental: derived from a monocular optical time-to-contact
    // estimate over a still-crude rectangle-heuristic car detection, not a
    // validated detector -- never a calibrated collision probability.
    public var forwardHazardState: String = "no_relevant_lead_object"
    public var trafficLevel: String = "empty"
    public var pedestriansDetected: Bool = false
    public var pedestrianCount: Int = 0
    // Discrete reading combining pedestrian position/motion, e.g.
    // "entering_conflict_zone", "roadside_visible" -- see
    // PedestrianHazard.swift (Phase 7). The most severe of any currently
    // tracked pedestrians, never a claim about a specific future path.
    public var pedestrianHazardState: String = "insufficient_evidence"
    public var eyesClosedSec: Double = 0.0
    /// How long the COMMITTED (post-hysteresis) attention state has been
    /// continuously off-road -- excludes faceNotDetected/uncertain, which
    /// are a confidence problem, not a confirmed-distraction severity one.
    /// Feeds MultimodalRiskFusion's attention-dimension severity.
    public var offRoadGazeSec: Double = 0.0
    public var cameraStatus: String = "idle"
    public var debugFaceDetected: Bool = false
    public var debugPitch: Double = 0.0
    public var debugYaw: Double = 0.0
    public var debugRoll: Double = 0.0
    public var debugEAR: Double = 0.0
    /// Pupil-within-eye-socket vertical offset (DriverAttentionState.swift's
    /// GazeOffset) -- the eye-gaze-alone signal that can flag a downward
    /// look even when head pitch stays level (e.g. a phone held low).
    /// `debugGazeAvailable` is false when pupil landmarks weren't reliably
    /// found this frame, distinguishing "no reading" from a genuine 0.0.
    public var debugGazeAvailable: Bool = false
    public var debugGazeVertical: Double = 0.0
    public var debugGazeHorizontal: Double = 0.0
    public var postureCollapse: Bool = false
    
    // Web UI Camera Feed (Disabled in production to prevent overheating)
    public var cameraFrame: String? = nil
    
    // Map Context
    public var isRoundabout: Bool? = false
    // Weather Context -- see ContextManager.swift. weatherIsLive distinguishes
    // a real fetched reading from the training-aligned "unknown" fallback
    // (always true right now: no real OpenWeather API key is configured on
    // either the iOS or backend side, so this is always the CLEAR/1.0
    // default until one is added).
    public var weatherMultiplier: Double = 1.0
    public var weatherConditionCode: Double = 6.0
    public var weatherIsLive: Bool = false
    
    public init(streamId: String, timestamp: Double = Date().timeIntervalSince1970) {
        self.streamId = streamId
        self.timestamp = timestamp
    }
    
    public func toDictionary() -> [String: Any] {
        var dict: [String: Any] = [
            "timestamp": timestamp,
            "stream_id": streamId,
            "source": source
        ]
        
        if let uid = userId { dict["user_id"] = uid }
        if let vid = vehicleId { dict["vehicle_id"] = vid }
        
        if let lat = gpsLatitude {
            dict["lat"] = lat
            dict["gps_latitude"] = lat
        }
        if let lon = gpsLongitude {
            dict["lon"] = lon
            dict["gps_longitude"] = lon
        }
        if let s = gpsSpeed { dict["gps_speed"] = s }
        if let h = gpsHeading { dict["gps_heading"] = h }
        
        if let ax = accelX { dict["accel_x"] = ax }
        if let ay = accelY { dict["accel_y"] = ay }
        if let az = accelZ { dict["accel_z"] = az }
        if let gx = gyroX { dict["gyro_x"] = gx }
        if let gy = gyroY { dict["gyro_y"] = gy }
        if let gz = gyroZ { dict["gyro_z"] = gz }

        if let am = accelMag {
            dict["accel_mag"] = am
        } else if let ax = accelX, let ay = accelY, let az = accelZ {
            dict["accel_mag"] = sqrt(ax * ax + ay * ay + az * az)
        }

        if let gm = gyroMag {
            dict["gyro_mag"] = gm
        } else if let gx = gyroX, let gy = gyroY, let gz = gyroZ {
            dict["gyro_mag"] = sqrt(gx * gx + gy * gy + gz * gz)
        }
        
        dict["obd_connected"] = isObdConnected
        dict["is_obd_connected"] = isObdConnected
        
        let reportedRpm = rpm ?? 0.0
        let reportedSpeed = vehicleSpeed ?? 0.0
        let reportedThrottle = throttlePosition ?? 0.0
        let reportedLoad = engineLoad ?? 0.0
        
        dict["rpm"] = reportedRpm
        dict["engine_speed"] = reportedRpm
        dict["speed"] = reportedSpeed
        dict["vehicle_speed"] = reportedSpeed
        dict["throttle"] = reportedThrottle
        dict["throttle_position"] = reportedThrottle
        dict["load"] = reportedLoad
        dict["engine_load"] = reportedLoad
        
        let hr = heartRate ?? 0.0
        dict["heart_rate"] = hr
        dict["bpm"] = hr
        dict["hr"] = hr
        if let rr = rrIntervalMs { dict["rr_interval_ms"] = rr }
        dict["whoop_connected"] = isWhoopConnected
        dict["amazfit_connected"] = isAmazfitConnected
        
        // Output Vision Factors
        dict["eyes_on_road"] = eyesOnRoad
        dict["attention_state"] = attentionState
        dict["lead_closing_fast"] = leadClosingFast
        dict["forward_hazard_state"] = forwardHazardState
        dict["traffic_level"] = trafficLevel
        dict["pedestrians_detected"] = pedestriansDetected
        dict["pedestrian_count"] = pedestrianCount
        dict["pedestrian_hazard_state"] = pedestrianHazardState
        dict["eyes_closed_sec"] = eyesClosedSec
        dict["off_road_gaze_sec"] = offRoadGazeSec
        dict["camera_status"] = cameraStatus
        dict["debug_face_detected"] = debugFaceDetected
        dict["debug_pitch"] = debugPitch
        dict["debug_yaw"] = debugYaw
        dict["debug_roll"] = debugRoll
        dict["debug_ear"] = debugEAR
        dict["debug_gaze_available"] = debugGazeAvailable
        dict["debug_gaze_vertical"] = debugGazeVertical
        dict["debug_gaze_horizontal"] = debugGazeHorizontal
        dict["posture_collapse"] = postureCollapse
        dict["is_roundabout"] = isRoundabout
        dict["weather_multiplier"] = weatherMultiplier
        dict["weather_condition_code"] = weatherConditionCode
        dict["weather_is_live"] = weatherIsLive
        
        if let cf = cameraFrame { dict["camera_frame"] = cf }
        
        return dict
    }
}
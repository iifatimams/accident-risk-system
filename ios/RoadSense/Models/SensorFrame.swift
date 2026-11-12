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
    public var rrIntervalMs: Double?
    public var isWhoopConnected: Bool = false
    
    // 7-Factor Multimodal Variables
    public var eyesOnRoad: Bool = true
    // Discrete reading behind eyesOnRoad above, e.g. "lookingDown",
    // "faceNotDetected", "uncertain" -- see DriverAttentionState.swift.
    // Kept as a plain String here (not the enum) since SensorFrame is a
    // simple Codable/dictionary payload shape, matching every other field.
    public var attentionState: String = "attentive"
    public var leadClosingFast: Bool = false
    public var trafficLevel: String = "empty"
    public var pedestriansDetected: Bool = false
    public var pedestrianCount: Int = 0
    public var eyesClosedSec: Double = 0.0
    public var cameraStatus: String = "idle"
    public var debugFaceDetected: Bool = false
    public var debugPitch: Double = 0.0
    public var debugYaw: Double = 0.0
    public var debugRoll: Double = 0.0
    public var debugEAR: Double = 0.0
    public var phoneScreenOn: Bool = false
    public var facialTension: Bool = false
    public var postureCollapse: Bool = false
    public var activeCall: Bool = false
    public var fixedGaze: Bool = false
    public var handsOccupied: Bool = false
    public var weavingPattern: Bool = false
    public var sluggishGaze: Bool = false
    
    // Web UI Camera Feed (Disabled in production to prevent overheating)
    public var cameraFrame: String? = nil
    
    // Map Context
    public var isRoundabout: Bool? = false
    
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
        dict["whoop_connected"] = isWhoopConnected
        
        // Output Vision Factors
        dict["eyes_on_road"] = eyesOnRoad
        dict["attention_state"] = attentionState
        dict["lead_closing_fast"] = leadClosingFast
        dict["traffic_level"] = trafficLevel
        dict["pedestrians_detected"] = pedestriansDetected
        dict["pedestrian_count"] = pedestrianCount
        dict["eyes_closed_sec"] = eyesClosedSec
        dict["camera_status"] = cameraStatus
        dict["debug_face_detected"] = debugFaceDetected
        dict["debug_pitch"] = debugPitch
        dict["debug_yaw"] = debugYaw
        dict["debug_roll"] = debugRoll
        dict["debug_ear"] = debugEAR
        dict["phone_screen_on"] = phoneScreenOn
        dict["facial_tension"] = facialTension
        dict["posture_collapse"] = postureCollapse
        dict["active_call"] = activeCall
        dict["fixed_gaze"] = fixedGaze
        dict["hands_occupied"] = handsOccupied
        dict["weaving_pattern"] = weavingPattern
        dict["sluggish_gaze"] = sluggishGaze
        dict["is_roundabout"] = isRoundabout
        
        if let cf = cameraFrame { dict["camera_frame"] = cf }
        
        return dict
    }
}
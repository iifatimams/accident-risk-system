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
        
        return dict
    }
}
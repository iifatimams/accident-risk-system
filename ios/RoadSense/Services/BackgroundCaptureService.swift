import Foundation
import CoreLocation
import CoreMotion

public class BackgroundCaptureService: NSObject, ObservableObject, CLLocationManagerDelegate, ObdDelegate, WhoopDelegate {
    private let locationManager = CLLocationManager()
    private let motionManager = CMMotionManager()
    public let obdClient = ObdBleClient()
    public let whoopClient = WhoopClient()
    private let uploader = Uploader()
    
    private let motionQueue = OperationQueue()
    private let syncQueue = DispatchQueue(label: "ae.aus.roadsense.syncQueue")
    
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
    
    public func start(withStreamId streamId: String? = nil, userId: Int? = nil) {
        guard !isRunning else { return }
        
        self.streamId = streamId ?? UUID().uuidString
        self.currentFrame = SensorFrame(streamId: self.streamId)
        self.currentFrame.userId = userId
        
        DispatchQueue.main.async {
            self.isRunning = true
            self.isObdConnected = false
            self.isWhoopConnected = false
        }
        
        // 1. Setup GPS Location
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        locationManager.distanceFilter = kCLDistanceFilterNone
        locationManager.requestWhenInUseAuthorization()
        
        if let lastLocation = locationManager.location {
            self.currentFrame.gpsLatitude = lastLocation.coordinate.latitude
            self.currentFrame.gpsLongitude = lastLocation.coordinate.longitude
            if lastLocation.speed >= 0 {
                self.currentFrame.gpsSpeed = lastLocation.speed
            }
        }
        
        if let backgroundModes = Bundle.main.object(forInfoDictionaryKey: "UIBackgroundModes") as? [String],
           backgroundModes.contains("location") {
            locationManager.allowsBackgroundLocationUpdates = true
            locationManager.showsBackgroundLocationIndicator = true
            locationManager.pausesLocationUpdatesAutomatically = false
        }
        locationManager.startUpdatingLocation()
        
        // 2. Setup Inertial Sensors
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.1
            motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, _ in
                guard let self = self, let m = motion else { return }
                self.syncQueue.async {
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
                    
                    self.currentFrame.accelMag = sqrt(ax * ax + ay * ay + az * az)
                    self.currentFrame.gyroMag = sqrt(gx * gx + gy * gy + gz * gz)
                }
            }
        }
        
        // 3. Connect Peripherals
        obdClient.start()
        whoopClient.startPolling()
        
        // 4. Stream Loop: 300 ms cadence
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.streamTimer = Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { [weak self] _ in
                self?.sendCurrentSnapshot()
            }
        }
    }
    
    private func sendCurrentSnapshot() {
        syncQueue.async { [weak self] in
            guard let self = self else { return }
            var snapshot = self.currentFrame
            snapshot.timestamp = Date().timeIntervalSince1970
            self.uploader.postSingle(serverUrl: self.serverUrl, frame: snapshot) { _ in }
        }
    }
    
    public func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let loc = locations.last else { return }
        syncQueue.async { [weak self] in
            guard let self = self else { return }
            self.currentFrame.gpsLatitude = loc.coordinate.latitude
            self.currentFrame.gpsLongitude = loc.coordinate.longitude
            if loc.speed >= 0 {
                self.currentFrame.gpsSpeed = loc.speed
            }
            if loc.course >= 0 {
                self.currentFrame.gpsHeading = loc.course * (.pi / 180.0)
            }
        }
    }
    
    public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        let status = manager.authorizationStatus
        if status == .authorizedWhenInUse || status == .authorizedAlways {
            if status == .authorizedWhenInUse {
                manager.requestAlwaysAuthorization()
            }
            if let backgroundModes = Bundle.main.object(forInfoDictionaryKey: "UIBackgroundModes") as? [String],
               backgroundModes.contains("location") {
                locationManager.allowsBackgroundLocationUpdates = true
            }
            locationManager.startUpdatingLocation()
        }
    }
    
    public func didChangeConnectionStatus(isConnected: Bool) {
        syncQueue.async { [weak self] in
            guard let self = self else { return }
            self.currentFrame.isObdConnected = isConnected
            if !isConnected {
                self.currentFrame.rpm = 0.0
                self.currentFrame.vehicleSpeed = 0.0
                self.currentFrame.throttlePosition = 0.0
                self.currentFrame.engineLoad = 0.0
            }
            DispatchQueue.main.async {
                self.isObdConnected = isConnected
            }
        }
    }
    
    public func didUpdateObdData(speed: Double?, rpm: Double?, throttle: Double?, load: Double?) {
        syncQueue.async { [weak self] in
            guard let self = self else { return }
            if let r = rpm, r > 0 {
                self.currentFrame.isObdConnected = true
                self.currentFrame.rpm = r
                if let s = speed { self.currentFrame.vehicleSpeed = s * 3.6 }
                if let t = throttle { self.currentFrame.throttlePosition = t }
                if let l = load { self.currentFrame.engineLoad = l }
                DispatchQueue.main.async {
                    self.isObdConnected = true
                }
            } else if let s = speed {
                self.currentFrame.vehicleSpeed = s * 3.6
                if let r = rpm { self.currentFrame.rpm = r }
                if let t = throttle { self.currentFrame.throttlePosition = t }
                if let l = load { self.currentFrame.engineLoad = l }
            }
        }
    }
    
    public func didUpdateWhoop(heartRate: Double?, rrInterval: Double?) {
        syncQueue.async { [weak self] in
            guard let self = self else { return }
            if let hr = heartRate, hr > 0 {
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
    
    public func stop() {
        guard isRunning else { return }
        streamTimer?.invalidate()
        streamTimer = nil
        locationManager.stopUpdatingLocation()
        motionManager.stopDeviceMotionUpdates()
        obdClient.stop()
        whoopClient.stop()
        
        DispatchQueue.main.async {
            self.isRunning = false
            self.isObdConnected = false
            self.isWhoopConnected = false
        }
    }
}
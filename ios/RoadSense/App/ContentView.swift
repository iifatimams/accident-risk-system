import SwiftUI
import CoreLocation
import CoreBluetooth

// MARK: - Server Configuration
public enum ServerConfig {
    // PASTE YOUR ACTIVE NGROK HTTPS URL HERE (WITHOUT A TRAILING SLASH)
    public static let activeBaseUrl = "https://evasion-stung-makeover.ngrok-free.dev"
}

// MARK: - Models
public struct UserProfile: Codable {
    public let id: Int
    public let name: String
    public let phone: String
    public let role: String
    public let kit: [KitDevice]?
}

public struct KitDevice: Codable {
    public let kind: String
    public let identifier: String
    public let status: String
}

// MARK: - Auth Manager
@MainActor
public class AuthManager: ObservableObject {
    public static let shared = AuthManager()
    
    @Published public var token: String? = UserDefaults.standard.string(forKey: "roadsense_auth_token")
    @Published public var currentUser: UserProfile?
    @Published public var assignedObdId: String?
    @Published public var assignedWhoopId: String?
    @Published public var isCodeSent = false
    @Published public var errorMessage: String?
    @Published public var isLoading = false
    @Published public var devCodeHint: String?
    
    public var baseUrl: String {
        return ServerConfig.activeBaseUrl.trimmingCharacters(in: CharacterSet(charactersIn: " /"))
    }
    
    public var isAuthenticated: Bool {
        return token != nil
    }
    
    public init() {
        if token != nil {
            Task {
                await fetchProfile()
            }
        }
    }
    
    public func requestCode(phone: String) async {
        isLoading = true
        errorMessage = nil
        devCodeHint = nil
        
        let cleanedPhone = phone.replacingOccurrences(of: " ", with: "")
                                .replacingOccurrences(of: "-", with: "")
                                .trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard let url = URL(string: "\(baseUrl)/app/api/auth/request-code") else {
            errorMessage = "Invalid server URL: \(baseUrl)"
            isLoading = false
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        let body = ["phone": cleanedPhone]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                let detail = json["detail"] as? String ?? "Request failed with HTTP \((response as? HTTPURLResponse)?.statusCode ?? 502)"
                errorMessage = detail
                isLoading = false
                return
            }
            
            if let code = json["code"] as? String {
                devCodeHint = "Code: \(code)"
            }
            isCodeSent = true
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
    
    public func verifyCode(phone: String, code: String) async -> Bool {
        isLoading = true
        errorMessage = nil
        
        let cleanedPhone = phone.replacingOccurrences(of: " ", with: "")
                                .replacingOccurrences(of: "-", with: "")
                                .trimmingCharacters(in: .whitespacesAndNewlines)
        
        guard let url = URL(string: "\(baseUrl)/app/api/auth/verify") else {
            errorMessage = "Invalid server URL: \(baseUrl)"
            isLoading = false
            return false
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        let body = [
            "phone": cleanedPhone,
            "code": code.trimmingCharacters(in: .whitespacesAndNewlines)
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let json = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            
            guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                let detail = json["detail"] as? String ?? "Verification failed"
                errorMessage = detail
                isLoading = false
                return false
            }
            
            if let receivedToken = json["token"] as? String {
                self.token = receivedToken
                UserDefaults.standard.set(receivedToken, forKey: "roadsense_auth_token")
                await fetchProfile()
                isLoading = false
                return true
            }
        } catch {
            errorMessage = error.localizedDescription
        }
        
        isLoading = false
        return false
    }
    
    public func fetchProfile() async {
        guard let token = token,
              let url = URL(string: "\(baseUrl)/app/api/me") else { return }
        
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let profile = try JSONDecoder().decode(UserProfile.self, from: data)
            self.currentUser = profile
            
            if let kit = profile.kit {
                self.assignedObdId = kit.first(where: { $0.kind == "adapter" })?.identifier
                self.assignedWhoopId = kit.first(where: { $0.kind == "wearable" })?.identifier
            }
        } catch {
            print("Failed to fetch driver profile: \(error)")
        }
    }
    
    public func signOut() {
        token = nil
        currentUser = nil
        assignedObdId = nil
        assignedWhoopId = nil
        isCodeSent = false
        UserDefaults.standard.removeObject(forKey: "roadsense_auth_token")
    }
}

// MARK: - Permission Trigger Manager
public class PermissionTriggerManager: NSObject, ObservableObject, CLLocationManagerDelegate, CBCentralManagerDelegate {
    private var locationManager: CLLocationManager?
    private var centralManager: CBCentralManager?
    
    @Published public var locationStatus: String = "Unknown"
    @Published public var bluetoothStatus: String = "Unknown"
    
    public override init() {
        super.init()
    }
    
    public func requestLocationPrompt() {
        if locationManager == nil {
            locationManager = CLLocationManager()
            locationManager?.delegate = self
        }
        locationManager?.requestAlwaysAuthorization()
    }
    
    public func requestBluetoothPrompt() {
        if centralManager == nil {
            centralManager = CBCentralManager(delegate: self, queue: nil)
        } else if centralManager?.state == .poweredOn {
            centralManager?.scanForPeripherals(withServices: nil, options: nil)
        }
    }
    
    public func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        DispatchQueue.main.async {
            switch manager.authorizationStatus {
            case .authorizedAlways: self.locationStatus = "Always"
            case .authorizedWhenInUse: self.locationStatus = "When In Use"
            case .denied: self.locationStatus = "Denied"
            case .restricted: self.locationStatus = "Restricted"
            case .notDetermined: self.locationStatus = "Not Determined"
            @unknown default: break
            }
        }
    }
    
    public func centralManagerDidUpdateState(_ central: CBCentralManager) {
        DispatchQueue.main.async {
            switch central.state {
            case .poweredOn: self.bluetoothStatus = "Powered On"
            case .unauthorized: self.bluetoothStatus = "Unauthorized"
            case .poweredOff: self.bluetoothStatus = "Powered Off"
            default: self.bluetoothStatus = "Other"
            }
        }
    }
}

// MARK: - Login View
public struct LoginView: View {
    @ObservedObject var auth = AuthManager.shared
    @AppStorage("saved_driver_phone") private var phoneNumber = ""
    @State private var verificationCode = ""
    @FocusState private var isFieldFocused: Bool
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer(minLength: 40)
                
                VStack(spacing: 8) {
                    Image(systemName: "car.side.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 56, height: 56)
                        .foregroundColor(.teal)
                    
                    Text("RoadSense Driver")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Sign in with your enrolled mobile number")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.bottom, 10)
                
                if !auth.isCodeSent {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Mobile Number")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.secondary)
                        
                        TextField("+971508151511", text: $phoneNumber)
                            .keyboardType(.phonePad)
                            .focused($isFieldFocused)
                            .padding()
                            .background(Color(.secondarySystemBackground))
                            .cornerRadius(12)
                        
                        if let error = auth.errorMessage {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                        
                        Button(action: {
                            isFieldFocused = false
                            Task {
                                await auth.requestCode(phone: phoneNumber)
                            }
                        }) {
                            HStack {
                                if auth.isLoading {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Text("Send Verification Code")
                                        .fontWeight(.bold)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.teal)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(phoneNumber.count < 7 || auth.isLoading)
                    }
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Enter 6-Digit Code")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.secondary)
                        
                        TextField("123456", text: $verificationCode)
                            .keyboardType(.numberPad)
                            .focused($isFieldFocused)
                            .multilineTextAlignment(.center)
                            .font(.title)
                            .padding()
                            .background(Color(.secondarySystemBackground))
                            .cornerRadius(12)
                        
                        if let hint = auth.devCodeHint {
                            Text(hint)
                                .font(.caption)
                                .foregroundColor(.blue)
                        }
                        
                        if let error = auth.errorMessage {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                        
                        Button(action: {
                            isFieldFocused = false
                            Task {
                                _ = await auth.verifyCode(phone: phoneNumber, code: verificationCode)
                            }
                        }) {
                            HStack {
                                if auth.isLoading {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Text("Confirm & Start")
                                        .fontWeight(.bold)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.teal)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        .disabled(verificationCode.count != 6 || auth.isLoading)
                        
                        Button(action: {
                            isFieldFocused = false
                            auth.isCodeSent = false
                            auth.errorMessage = nil
                        }) {
                            Text("Use different number")
                                .font(.footnote)
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity)
                                .padding(.top, 4)
                        }
                    }
                }
                
                Spacer()
            }
            .padding(28)
        }
        .onTapGesture {
            isFieldFocused = false
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") {
                    isFieldFocused = false
                }
                .fontWeight(.semibold)
                .foregroundColor(.teal)
            }
        }
    }
}

// MARK: - Main Content View
public struct ContentView: View {
    @EnvironmentObject var captureService: BackgroundCaptureService
    @ObservedObject var auth = AuthManager.shared
    @StateObject private var permissions = PermissionTriggerManager()
    
    public var body: some View {
        if !auth.isAuthenticated {
            LoginView()
        } else {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(auth.currentUser?.name ?? "Driver")
                                .font(.title2)
                                .fontWeight(.bold)
                            Text(auth.currentUser?.phone ?? "")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Button("Sign Out") {
                            if captureService.isRunning {
                                captureService.stop()
                            }
                            auth.signOut()
                        }
                        .font(.footnote)
                        .foregroundColor(.red)
                    }
                    .padding(.horizontal)
                    .padding(.top, 10)
                    
                    // Permission Buttons
                    VStack(spacing: 10) {
                        Button(action: {
                            permissions.requestLocationPrompt()
                        }) {
                            HStack {
                                Image(systemName: "location.circle.fill")
                                Text("1. Location Permission")
                                Spacer()
                                Text(permissions.locationStatus)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .background(Color(.secondarySystemBackground))
                            .cornerRadius(10)
                        }
                        
                        Button(action: {
                            permissions.requestBluetoothPrompt()
                        }) {
                            HStack {
                                Image(systemName: "antenna.radiowaves.left.and.right")
                                Text("2. Bluetooth Permission")
                                Spacer()
                                Text(permissions.bluetoothStatus)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .background(Color(.secondarySystemBackground))
                            .cornerRadius(10)
                        }
                    }
                    .padding(.horizontal)
                    
                    // Hardware Kit
                    VStack(alignment: .leading, spacing: 6) {
                        Text("ASSIGNED HARDWARE KIT")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(.secondary)
                        HStack {
                            Image(systemName: "car.fill")
                            Text("OBD: \(auth.assignedObdId ?? "Default (OBDBLE)")")
                                .font(.subheadline)
                            Spacer()
                        }
                        HStack {
                            Image(systemName: "heart.fill")
                                .foregroundColor(.red)
                            Text("Wearable: \(auth.assignedWhoopId ?? "Any WHOOP")")
                                .font(.subheadline)
                            Spacer()
                        }
                    }
                    .padding(12)
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(10)
                    .padding(.horizontal)
                    
                    // Status Card
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Session:")
                            Spacer()
                            Text(captureService.isRunning ? "RECORDING" : "IDLE")
                                .fontWeight(.bold)
                                .foregroundColor(captureService.isRunning ? .green : .orange)
                        }
                        
                        HStack {
                            Text("Stream ID:")
                            Spacer()
                            Text(captureService.streamId.isEmpty ? "Inactive" : String(captureService.streamId.prefix(12)))
                                .font(.system(.body, design: .monospaced))
                        }
                        
                        HStack {
                            Text("OBD Adapter:")
                            Spacer()
                            Text(captureService.isObdConnected ? "CONNECTED" : "DISCONNECTED")
                                .fontWeight(.semibold)
                                .foregroundColor(captureService.isObdConnected ? .green : .red)
                        }
                        
                        HStack {
                            Text("WHOOP Wearable:")
                            Spacer()
                            Text(captureService.isWhoopConnected ? "CONNECTED" : "DISCONNECTED")
                                .fontWeight(.semibold)
                                .foregroundColor(captureService.isWhoopConnected ? .green : .red)
                        }
                    }
                    .padding()
                    .background(Color(.secondarySystemBackground))
                    .cornerRadius(12)
                    .padding(.horizontal)
                    
                    // Action Button
                    Button(action: {
                        if captureService.isRunning {
                            stopDriverSession()
                        } else {
                            startDriverSession()
                        }
                    }) {
                        Text(captureService.isRunning ? "Stop Drive Session" : "Start Drive Session")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(captureService.isRunning ? Color.red : Color.teal)
                            .cornerRadius(12)
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 30)
                }
            }
            // Offline-capable: Confirm/Dismiss write to CollisionIncidentStore
            // directly and only best-effort attempt the server afterward --
            // this overlay works with zero connectivity. Non-dismissable by
            // swipe (unlike a sheet) since this is a safety prompt, not a
            // casual screen. suspectedCollision is cleared by the confirm/
            // dismiss call itself, so acting on it removes this overlay and
            // it will not reappear for the same incident id.
            .overlay {
                if let suspicion = captureService.suspectedCollision {
                    CollisionAlertOverlay(
                        suspicion: suspicion,
                        onConfirm: { captureService.confirmSuspectedCollision(id: suspicion.id) },
                        onDismiss: { captureService.dismissSuspectedCollision(id: suspicion.id) }
                    )
                }
            }
        }
    }

    private func startDriverSession() {
        guard let token = auth.token,
              let url = URL(string: "\(auth.baseUrl)/app/api/driver/session/start") else { return }
        
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        req.httpBody = try? JSONSerialization.data(withJSONObject: [:])
        
        URLSession.shared.dataTask(with: req) { data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let streamId = json["stream_id"] as? String else {
                print("Failed to start session on backend")
                return
            }
            
            DispatchQueue.main.async {
                captureService.obdClient.targetDeviceIdentifier = auth.assignedObdId
                captureService.whoopClient.targetDeviceIdentifier = auth.assignedWhoopId
                // Point capture service directly to active base URL
                captureService.serverUrl = auth.baseUrl
                captureService.start(withStreamId: streamId, userId: auth.currentUser?.id)
            }
        }.resume()
    }
    
    private func stopDriverSession() {
        captureService.stop()
        
        guard let token = auth.token,
              let url = URL(string: "\(auth.baseUrl)/app/api/driver/session/stop") else { return }
        
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        URLSession.shared.dataTask(with: req).resume()
    }
}

// MARK: - Collision Confirmation
/// Shown the instant CollisionDetector triggers. Confirm/Dismiss call back
/// into BackgroundCaptureService, which writes to CollisionIncidentStore
/// synchronously -- this view needs no network to be fully usable.
public struct CollisionAlertOverlay: View {
    public let suspicion: CollisionSuspicion
    public let onConfirm: () -> Void
    public let onDismiss: () -> Void

    public var body: some View {
        ZStack {
            Color.black.opacity(0.6).ignoresSafeArea()
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundColor(.red)
                Text("Possible Collision Detected")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("A sudden impact was detected. Were you in an accident?")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.secondary)
                Text("This works without an internet connection; it will sync automatically.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 4)
                Button(action: onConfirm) {
                    Text("Yes, I Was in an Accident")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red)
                        .cornerRadius(12)
                }
                Button(action: onDismiss) {
                    Text("No, False Alarm")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.secondarySystemBackground))
                        .cornerRadius(12)
                }
            }
            .padding(24)
            .background(Color(.systemBackground))
            .cornerRadius(20)
            .padding(.horizontal, 30)
            .shadow(radius: 20)
        }
    }
}
import SwiftUI
import CoreLocation
import CoreBluetooth

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
    @State private var showingSettings = false
    @FocusState private var isFieldFocused: Bool
    
    public var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Button("Server Settings") { showingSettings = true }
                    .disabled(auth.isLoading)
                Text(auth.baseUrl.isEmpty ? "Choose a backend in Settings" : auth.baseUrl)
                    .font(.caption).foregroundColor(.secondary)
                if auth.token != nil {
                    Button("Retry Sign-In Connection") { Task { await auth.fetchProfile() } }
                        .disabled(auth.isLoading)
                }
                Spacer(minLength: 16)
                
                VStack(spacing: 8) {
                    Image(systemName: "car.side.fill")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 56, height: 56)
                        .foregroundColor(.teal)
                    
                    Text("RoadSense Driver")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text(auth.demoMode ? "Restricted university demo account" : "Sign in using an SMS verification code")
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
                        
                        TextField("Mobile number with country code", text: $phoneNumber)
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
                                    Text("Sign In")
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
        .sheet(isPresented: $showingSettings) { ServerSettingsView() }
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
    @State private var showingSettings = false
    
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
                        Button("Settings") { showingSettings = true }
                            .disabled(captureService.isRunning || auth.isLoading)
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
                    
                    if let error = auth.errorMessage {
                        Text(error).font(.callout).foregroundColor(.red).padding(.horizontal)
                    }
                    if auth.isLoading { ProgressView("Contacting RoadSense…") }
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
                    .disabled(auth.isLoading)
                }
            }
            .sheet(isPresented: $showingSettings) { ServerSettingsView() }
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
        Task {
            guard let streamId = await auth.startSession() else { return }
            captureService.obdClient.targetDeviceIdentifier = auth.assignedObdId
            captureService.whoopClient.targetDeviceIdentifier = auth.assignedWhoopId
            captureService.serverUrl = auth.baseUrl
            captureService.start(withStreamId: streamId, userId: auth.currentUser?.id)
        }
    }

    private func stopDriverSession() {
        captureService.stop()
        Task { await auth.stopSession() }
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

/// Available before sign-in so an expired tunnel can always be replaced.
public struct ServerSettingsView: View {
    @ObservedObject private var auth = AuthManager.shared
    @Environment(\.dismiss) private var dismiss
    @State private var endpoint = ""
    @State private var demoMode = false
    @State private var pairingKey = ""

    public var body: some View {
        NavigationStack {
            Form {
                Section("Backend address") {
                    TextField("https://your-tunnel.ngrok-free.app", text: $endpoint)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    Text("For different Wi-Fi or mobile data, use your current HTTPS tunnel. localhost refers to the iPhone, not your Mac. Changing servers signs you out.")
                        .font(.caption)
                }
                Section("University demo") {
                    Toggle("Restricted demo login", isOn: $demoMode)
                    if demoMode {
                        SecureField("Private pairing key from backend operator", text: $pairingKey)
                            .textInputAutocapitalization(.never).autocorrectionDisabled()
                        Text("Use test accounts on the separate demo server. The pairing key authorizes code display and is kept only until this app closes. Never share it publicly.")
                            .font(.caption)
                    }
                }
                Section {
                    Button("Save & Test Connection") {
                        if auth.saveEndpoint(endpoint) {
                            auth.demoMode = demoMode
                            auth.demoKey = demoMode ? pairingKey.trimmingCharacters(in: .whitespacesAndNewlines) : ""
                            Task { await auth.testConnection() }
                        }
                    }
                    .disabled(auth.isLoading)
                    if auth.isLoading { ProgressView() }
                    if let message = auth.connectionMessage { Text(message).foregroundColor(.green) }
                    if let error = auth.errorMessage { Text(error).foregroundColor(.red) }
                }
            }
            .navigationTitle("Server Settings")
            .toolbar { Button("Done") { dismiss() }.disabled(auth.isLoading) }
            .onAppear {
                endpoint = auth.baseUrl
                demoMode = auth.demoMode
                pairingKey = auth.demoKey
            }
        }
    }
}

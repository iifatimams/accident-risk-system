import Foundation
import Combine

/// Main-actor login state. Tokens are bound to one endpoint and never imply a valid profile.
@MainActor
public class AuthManager: ObservableObject {
    public static let shared = AuthManager()
    @Published public private(set) var token: String?
    @Published public private(set) var currentUser: UserProfile?
    @Published public private(set) var assignedObdId: String?
    @Published public private(set) var assignedWhoopId: String?
    @Published public var isCodeSent = false
    @Published public var errorMessage: String?
    @Published public private(set) var isLoading = false
    @Published public private(set) var devCodeHint: String?
    @Published public private(set) var connectionMessage: String?
    @Published public private(set) var baseUrl: String
    // Pairing credential is intentionally memory-only and never logged or saved in preferences.
    @Published public var demoKey = ""
    @Published public var demoMode = false
    private let defaults: UserDefaults
    private let api: RoadSenseAPI
    private var generation = 0
    private var requestedPhone: String?

    public var isAuthenticated: Bool { token != nil && currentUser != nil }

    public init(defaults: UserDefaults = .standard, api: RoadSenseAPI = RoadSenseAPI(), restore: Bool = true) {
        self.defaults = defaults
        self.api = api
        baseUrl = defaults.string(forKey: "roadsense_server_url") ?? ""
        // Discard old unscoped tokens left by the hardcoded tunnel implementation.
        if !baseUrl.isEmpty && defaults.string(forKey: "roadsense_token_endpoint") == baseUrl {
            token = defaults.string(forKey: "roadsense_auth_token")
        } else {
            defaults.removeObject(forKey: "roadsense_auth_token")
            defaults.removeObject(forKey: "roadsense_token_endpoint")
        }
        if restore && token != nil { Task { await fetchProfile() } }
    }

    /// Saving a new endpoint invalidates both in-flight responses and the previous login.
    @discardableResult
    public func saveEndpoint(_ value: String) -> Bool {
        do {
            let normalized = try ServerConfig.normalize(value)
            if normalized != baseUrl {
                signOut()
                demoKey = ""
                demoMode = false
                baseUrl = normalized
                defaults.set(normalized, forKey: "roadsense_server_url")
            }
            connectionMessage = nil
            errorMessage = nil
            return true
        } catch { errorMessage = error.localizedDescription; return false }
    }

    public func testConnection() async {
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil
        connectionMessage = nil
        let attempt = generation
        defer { isLoading = false }
        do {
            let result: ConnectionResponse = try await api.request(base: baseUrl, path: "/app/api/connection")
            guard attempt == generation else { return }
            guard result.service == "roadsense", result.api_version == 1 else { throw APIError.invalidResponse }
            connectionMessage = "Connected to RoadSense at \(baseUrl). \(result.demo_mode ? "Restricted demo server." : "SMS login server.")"
        } catch { if attempt == generation { errorMessage = error.localizedDescription } }
    }

    public func requestCode(phone: String) async {
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil
        devCodeHint = nil
        isCodeSent = false
        let attempt = generation
        defer { isLoading = false }
        do {
            if demoMode && demoKey.count < 32 {
                throw APIError.message("Enter the private demo pairing key in Settings. Ask the backend operator for it.")
            }
            let result: CodeResponse = try await api.request(
                base: baseUrl, path: "/app/api/auth/request-code", method: "POST", body: ["phone": phone],
                demoKey: demoMode ? demoKey : nil)
            guard attempt == generation else { return }
            guard result.sent, result.expires_in > 0 else { throw APIError.invalidResponse }
            if result.delivery == "demo" {
                guard demoMode, let code = result.code, code.count == 6,
                      code.allSatisfy({ "0123456789".contains($0) }) else { throw APIError.invalidResponse }
                devCodeHint = "DEMO ONLY — Code: \(code)"
            } else {
                guard result.delivery == "sms", result.code == nil else { throw APIError.invalidResponse }
            }
            requestedPhone = phone
            isCodeSent = true
        } catch { if attempt == generation { errorMessage = error.localizedDescription } }
    }

    @discardableResult
    public func verifyCode(phone: String, code: String) async -> Bool {
        guard !isLoading else { return false }
        isLoading = true
        errorMessage = nil
        let attempt = generation
        defer { isLoading = false }
        do {
            guard isCodeSent, phone == requestedPhone else { throw APIError.message("Request a code for this number first.") }
            let result: VerifyResponse = try await api.request(base: baseUrl, path: "/app/api/auth/verify", method: "POST",
                body: ["phone": phone, "code": code.trimmingCharacters(in: .whitespacesAndNewlines)])
            guard attempt == generation else { return false }
            guard !result.token.isEmpty, result.expires_in > 0 else { throw APIError.invalidResponse }
            token = result.token
            defaults.set(result.token, forKey: "roadsense_auth_token")
            defaults.set(baseUrl, forKey: "roadsense_token_endpoint")
            devCodeHint = nil
            // The code has been consumed. A failed profile load is retried with the token.
            isCodeSent = false
            return await loadProfile(attempt: attempt)
        } catch {
            if attempt == generation { errorMessage = error.localizedDescription }
            return false
        }
    }

    public func fetchProfile() async {
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        _ = await loadProfile(attempt: generation)
    }

    private func loadProfile(attempt: Int) async -> Bool {
        guard let token else { return false }
        do {
            let profile: UserProfile = try await api.request(base: baseUrl, path: "/app/api/me", token: token)
            guard attempt == generation else { return false }
            guard profile.id > 0, !profile.phone.isEmpty,
                  ["driver", "account_manager", "administrator"].contains(profile.role) else { throw APIError.invalidResponse }
            currentUser = profile
            assignedObdId = profile.kit?.first { $0.kind == "adapter" }?.identifier
            assignedWhoopId = profile.kit?.first { $0.kind == "wearable" }?.identifier
            return true
        } catch {
            if attempt == generation { handleAuthenticatedError(error) }
            return false
        }
    }

    /// Capture starts only after a successful, typed, owned backend session response.
    public func startSession() async -> String? {
        guard !isLoading, isAuthenticated, let token else { return nil }
        isLoading = true
        errorMessage = nil
        let attempt = generation
        defer { isLoading = false }
        do {
            let result: DriveResponse = try await api.request(base: baseUrl, path: "/app/api/driver/session/start",
                                                             method: "POST", body: [:], token: token)
            guard attempt == generation else { return nil }
            guard result.id > 0, !result.stream_id.isEmpty, result.user_id == currentUser?.id else { throw APIError.invalidResponse }
            return result.stream_id
        } catch {
            if attempt == generation { handleAuthenticatedError(error) }
            return nil
        }
    }

    public func stopSession() async {
        guard !isLoading, let token else { return }
        isLoading = true
        errorMessage = nil
        let attempt = generation
        defer { isLoading = false }
        do {
            let result: StopResponse = try await api.request(base: baseUrl, path: "/app/api/driver/session/stop", method: "POST", token: token)
            guard attempt == generation else { return }
            guard result.status == "no_active_session" || result.session?.ended_at != nil else { throw APIError.invalidResponse }
        } catch {
            if attempt == generation {
                handleAuthenticatedError(error)
                errorMessage = "Capture stopped on this phone. Backend stop was not confirmed: \(error.localizedDescription)"
            }
        }
    }

    private func handleAuthenticatedError(_ error: Error) {
        if let apiError = error as? APIError, apiError.invalidatesToken { signOut() }
        errorMessage = error.localizedDescription
    }

    public func signOut() {
        generation += 1
        token = nil
        currentUser = nil
        assignedObdId = nil
        assignedWhoopId = nil
        isCodeSent = false
        requestedPhone = nil
        devCodeHint = nil
        errorMessage = nil
        defaults.removeObject(forKey: "roadsense_auth_token")
        defaults.removeObject(forKey: "roadsense_token_endpoint")
    }
}

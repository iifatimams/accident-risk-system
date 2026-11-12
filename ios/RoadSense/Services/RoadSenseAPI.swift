import Foundation

/// Validated endpoint shared by authentication and driving capture. No baked-in tunnel.
public enum ServerConfig {
    public static var activeBaseUrl: String {
        UserDefaults.standard.string(forKey: "roadsense_server_url") ?? ""
    }

    public static func normalize(_ value: String) throws -> String {
        let text = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var url = URLComponents(string: text),
              let host = url.host?.lowercased(), !host.isEmpty,
              ["https", "http"].contains(url.scheme?.lowercased() ?? ""),
              url.user == nil, url.password == nil, url.query == nil, url.fragment == nil,
              url.path.isEmpty || url.path == "/",
              url.port == nil || (1...65535).contains(url.port!) else {
            throw APIError.message("Enter the backend address, such as https://your-tunnel.ngrok-free.app, without a page path.")
        }
        let octets = host.split(separator: ".").compactMap { Int($0) }
        let privateIPv4 = octets.count == 4 && octets.allSatisfy { (0...255).contains($0) } &&
            (octets[0] == 10 || octets[0] == 127 ||
             (octets[0] == 192 && octets[1] == 168) ||
             (octets[0] == 172 && (16...31).contains(octets[1])))
        if url.scheme?.lowercased() == "http" &&
            !(privateIPv4 || host == "localhost" || host == "[::1]" || host.hasSuffix(".local")) {
            throw APIError.message("Use HTTPS for a public server or tunnel. HTTP is allowed only for a trusted local network.")
        }
        url.scheme = url.scheme?.lowercased()
        url.host = host
        url.path = ""
        guard let normalized = url.url?.absoluteString else { throw APIError.invalidResponse }
        return normalized
    }
}

public enum APIError: LocalizedError {
    case message(String)
    case http(Int, String)
    case invalidResponse

    public var errorDescription: String? {
        switch self {
        case .message(let text), .http(_, let text): return text
        case .invalidResponse: return "The server returned invalid JSON or missing fields. Check the backend address and version in Settings."
        }
    }
    public var invalidatesToken: Bool {
        if case .http(let status, _) = self { return status == 401 || status == 403 }
        return false
    }
}

/// Refuse redirects so neither bearer tokens nor demo keys follow a stale tunnel elsewhere.
private final class NoRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

public final class RoadSenseAPI {
    private let session: URLSession

    public init(session: URLSession? = nil) {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 12
        config.timeoutIntervalForResource = 20
        self.session = session ?? URLSession(configuration: config, delegate: NoRedirects(), delegateQueue: nil)
    }

    /// Validate transport, status, MIME type, JSON and typed fields before returning success.
    public func request<T: Decodable>(base: String, path: String, method: String = "GET",
                                      body: [String: String]? = nil, token: String? = nil,
                                      demoKey: String? = nil) async throws -> T {
        let endpoint = try ServerConfig.normalize(base)
        guard let url = URL(string: endpoint + path) else { throw APIError.invalidResponse }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let demoKey { request.setValue(demoKey, forHTTPHeaderField: "X-RoadSense-Demo-Key") }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        do {
            let (data, response) = try await session.data(for: request)
            return try Self.decode(data, response: response)
        } catch let error as APIError {
            throw error
        } catch let error as URLError {
            if error.code == .cancelled { throw error }
            throw APIError.message("Cannot reach \(url.host ?? "the backend"). Check that the server and tunnel are running, the URL is current, and the iPhone has internet or local-network access. (\(error.code.rawValue))")
        }
    }

    static func decode<T: Decodable>(_ data: Data, response: URLResponse) throws -> T {
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        if (300...399).contains(http.statusCode) {
            throw APIError.message("The server redirected this request. Enter the final HTTPS backend URL in Settings.")
        }
        let mime = http.mimeType?.lowercased() ?? ""
        let prefix = String(decoding: data.prefix(256), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        if mime.contains("html") || prefix.hasPrefix("<") {
            throw APIError.message("The server returned an HTML page (HTTP \(http.statusCode)), not RoadSense JSON. The tunnel may have expired or the address may be wrong. Update it in Settings.")
        }
        guard mime == "application/json" || (mime.hasPrefix("application/") && mime.hasSuffix("+json")) else {
            throw APIError.message("The server returned \(mime.isEmpty ? "no content type" : mime) (HTTP \(http.statusCode)), not JSON. Check the backend URL in Settings.")
        }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError.invalidResponse
        }
        guard (200...299).contains(http.statusCode) else {
            let detail = object["detail"] as? String
            throw APIError.http(http.statusCode, detail ?? "RoadSense rejected the request (HTTP \(http.statusCode)). Check your input or sign in again.")
        }
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw APIError.invalidResponse }
    }
}

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
struct CodeResponse: Decodable { let sent: Bool; let delivery: String; let expires_in: Int; let code: String? }
struct VerifyResponse: Decodable { let token: String; let expires_in: Int }
struct ConnectionResponse: Decodable { let service: String; let api_version: Int; let demo_mode: Bool }
struct DriveResponse: Decodable { let id: Int; let stream_id: String; let user_id: Int }
struct StopResponse: Decodable {
    let status: String?
    let session: StoppedSession?
    struct StoppedSession: Decodable { let ended_at: Double? }
}

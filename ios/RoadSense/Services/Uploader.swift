import Foundation

public class Uploader {
    private let session: URLSession
    private var isUploading = false
    
    public init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 4.0
        config.timeoutIntervalForResource = 4.0
        config.httpMaximumConnectionsPerHost = 2
        config.shouldUseExtendedBackgroundIdleMode = true
        self.session = URLSession(configuration: config)
    }
    
    public func postSingle(serverUrl: String, frame: SensorFrame, completion: @escaping (Bool) -> Void) {
        guard !isUploading else {
            completion(false)
            return
        }
        
        var trimmedUrl = serverUrl.trimmingCharacters(in: CharacterSet(charactersIn: " /"))
        // If no API path is present, append /api/v1/ingest automatically
        if !trimmedUrl.contains("/ingest") {
            trimmedUrl += "/api/v1/ingest"
        }
        
        guard let url = URL(string: trimmedUrl) else {
            completion(false)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        request.setValue("RoadSense-iOS", forHTTPHeaderField: "User-Agent")
        request.setValue("keep-alive", forHTTPHeaderField: "Connection")
        
        let payload = frame.toDictionary()
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: payload, options: [])
        } catch {
            completion(false)
            return
        }
        
        isUploading = true
        session.dataTask(with: request) { [weak self] _, response, error in
            self?.isUploading = false
            if error == nil, let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) {
                completion(true)
            } else {
                completion(false)
            }
        }.resume()
    }
}
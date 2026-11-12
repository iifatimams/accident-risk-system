import XCTest
@testable import RoadSenseAuth

private final class StubProtocol: URLProtocol {
    static var handler: ((URLRequest) throws -> (Int, String, String))!
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, mime, body) = try Self.handler(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil,
                                           headerFields: ["Content-Type": mime])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@MainActor
final class AuthTests: XCTestCase {
    let base = "https://test.example"
    let phone = "+12025550101"
    let profile = #"{"id":1,"name":"Test","phone":"+12025550101","role":"driver","kit":[]}"#

    func makeAuth() -> (AuthManager, UserDefaults) {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        let defaults = UserDefaults(suiteName: "roadsense-test-\(UUID().uuidString)")!
        let auth = AuthManager(defaults: defaults, api: RoadSenseAPI(session: URLSession(configuration: config)), restore: false)
        XCTAssertTrue(auth.saveEndpoint(base))
        return (auth, defaults)
    }

    func signIn(_ auth: AuthManager) async {
        StubProtocol.handler = { request in
            switch request.url!.path {
            case "/app/api/auth/request-code": return (200, "application/json", #"{"sent":true,"delivery":"sms","expires_in":300}"#)
            case "/app/api/auth/verify": return (200, "application/json", #"{"token":"synthetic-token","expires_in":3600}"#)
            default: return (200, "application/json", self.profile)
            }
        }
        await auth.requestCode(phone: phone)
        let success = await auth.verifyCode(phone: phone, code: "123456")
        XCTAssertTrue(success)
        XCTAssertTrue(auth.isAuthenticated)
    }

    func testValidJSONLoginAndSession() async {
        let (auth, _) = makeAuth()
        await signIn(auth)
        StubProtocol.handler = { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer synthetic-token")
            XCTAssertEqual(request.value(forHTTPHeaderField: "ngrok-skip-browser-warning"), "true")
            return (200, "application/json", #"{"id":2,"stream_id":"test-trip","user_id":1}"#)
        }
        let stream = await auth.startSession()
        XCTAssertEqual(stream, "test-trip")
    }

    func testHTMLExpiredTunnelWrongMIMEAndMalformedJSONAreNotSuccess() async {
        for (status, mime, body) in [(200,"text/html","<html>Warning</html>"),
                                     (404,"text/html","<html>ERR_NGROK_3200</html>"),
                                     (502,"application/json","<html>Offline</html>"),
                                     (200,"text/plain","{}"), (200,"application/json","{"),
                                     (200,"application/json","{}"), (200,"application/json","[]"),
                                     (200,"application/json",#"{"sent":false,"delivery":"sms","expires_in":300}"#),
                                     (302,"text/html","")] {
            let (auth, _) = makeAuth()
            StubProtocol.handler = { _ in (status, mime, body) }
            await auth.requestCode(phone: phone)
            XCTAssertFalse(auth.isCodeSent)
            XCTAssertNotNil(auth.errorMessage)
            XCTAssertFalse(auth.isLoading)
        }
    }

    func testUnreachableServerAndConnectionProbe() async {
        let (auth, _) = makeAuth()
        StubProtocol.handler = { _ in throw URLError(.cannotConnectToHost) }
        await auth.testConnection()
        XCTAssertTrue(auth.errorMessage!.contains("Cannot reach"))
        XCTAssertNil(auth.connectionMessage)
        StubProtocol.handler = { _ in (200, "application/json", #"{"service":"roadsense","api_version":1,"demo_mode":true}"#) }
        await auth.testConnection()
        XCTAssertNotNil(auth.connectionMessage)
        XCTAssertNil(auth.errorMessage)
        StubProtocol.handler = { _ in (200, "application/json", #"{"service":"other","api_version":1,"demo_mode":false}"#) }
        await auth.testConnection()
        XCTAssertNil(auth.connectionMessage)
        XCTAssertNotNil(auth.errorMessage)
    }

    func testDemoCodeRequiresOptInAndPairingHeader() async {
        let (auth, _) = makeAuth()
        auth.demoMode = true
        await auth.requestCode(phone: phone)
        XCTAssertFalse(auth.isCodeSent)
        auth.demoKey = String(repeating: "x", count: 32)
        StubProtocol.handler = { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-RoadSense-Demo-Key"), String(repeating: "x", count: 32))
            return (200, "application/json", #"{"sent":true,"delivery":"demo","expires_in":300,"code":"123456"}"#)
        }
        await auth.requestCode(phone: phone)
        XCTAssertEqual(auth.devCodeHint, "DEMO ONLY — Code: 123456")
        XCTAssertTrue(auth.isCodeSent)
    }

    func testInvalidTokenClearedButNetworkFailureKeepsRetryableToken() async {
        let (auth, defaults) = makeAuth()
        await signIn(auth)
        StubProtocol.handler = { _ in throw URLError(.timedOut) }
        await auth.fetchProfile()
        XCTAssertNotNil(auth.token)
        StubProtocol.handler = { _ in (401, "application/json", #"{"detail":"Your session has expired; sign in again"}"#) }
        let stream = await auth.startSession()
        XCTAssertNil(stream)
        XCTAssertFalse(auth.isAuthenticated)
        XCTAssertNil(defaults.string(forKey: "roadsense_auth_token"))
        XCTAssertTrue(auth.errorMessage!.contains("expired"))
    }

    func testChangingEndpointClearsTokensAndDemoCredentials() async {
        let (auth, defaults) = makeAuth()
        await signIn(auth)
        auth.demoKey = "secret"
        XCTAssertTrue(auth.saveEndpoint("https://new.example/"))
        XCTAssertFalse(auth.isAuthenticated)
        XCTAssertNil(defaults.string(forKey: "roadsense_auth_token"))
        XCTAssertTrue(auth.demoKey.isEmpty)
        XCTAssertEqual(auth.baseUrl, "https://new.example")
    }

    func testLegacyUnscopedTokenIsDiscarded() {
        let defaults = UserDefaults(suiteName: "roadsense-legacy-\(UUID().uuidString)")!
        defaults.set("old-token", forKey: "roadsense_auth_token")
        let auth = AuthManager(defaults: defaults, restore: false)
        XCTAssertNil(auth.token)
        XCTAssertFalse(auth.isAuthenticated)
    }

    func testProfileFailureDoesNotFinishSignIn() async {
        let (auth, _) = makeAuth()
        StubProtocol.handler = { request in
            if request.url!.path.hasSuffix("request-code") { return (200,"application/json",#"{"sent":true,"delivery":"sms","expires_in":300}"#) }
            if request.url!.path.hasSuffix("verify") { return (200,"application/json",#"{"token":"test-token","expires_in":3600}"#) }
            return (200,"text/html","<html>Wrong server</html>")
        }
        await auth.requestCode(phone: phone)
        let success = await auth.verifyCode(phone: phone, code: "123456")
        XCTAssertFalse(success)
        XCTAssertFalse(auth.isAuthenticated)
        XCTAssertNotNil(auth.token) // The consumed code need not be reused; profile load can be retried.
        XCTAssertTrue(auth.errorMessage!.contains("HTML"))
    }

    func testFailedOrWrongOwnerSessionNeverStartsCapture() async {
        let (auth, _) = makeAuth()
        await signIn(auth)
        for (status, mime, body) in [(502,"text/html","<html>Tunnel expired</html>"),
            (200,"application/json",#"{"id":2,"stream_id":"","user_id":1}"#),
            (200,"application/json",#"{"id":2,"stream_id":"other","user_id":9}"#)] {
            StubProtocol.handler = { _ in (status,mime,body) }
            let stream = await auth.startSession()
            XCTAssertNil(stream)
            XCTAssertNotNil(auth.errorMessage)
        }
    }

    func testRestoredTokenMustLoadProfileAndIsClearedOnRejection() async {
        let (first, defaults) = makeAuth()
        await signIn(first)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        let restored = AuthManager(defaults: defaults,
            api: RoadSenseAPI(session: URLSession(configuration: config)), restore: false)
        XCTAssertNotNil(restored.token)
        XCTAssertFalse(restored.isAuthenticated)
        StubProtocol.handler = { _ in (403,"application/json",#"{"detail":"This account is not active"}"#) }
        await restored.fetchProfile()
        XCTAssertNil(restored.token)
        XCTAssertFalse(restored.isAuthenticated)
        XCTAssertNil(defaults.string(forKey: "roadsense_auth_token"))
    }

    func testSessionStopValidatesResponseAndReportsOfflineFailure() async {
        let (auth, _) = makeAuth()
        await signIn(auth)
        StubProtocol.handler = { _ in throw URLError(.notConnectedToInternet) }
        await auth.stopSession()
        XCTAssertTrue(auth.errorMessage!.contains("Backend stop was not confirmed"))
        StubProtocol.handler = { _ in (200,"application/json",#"{"session":{"ended_at":123.0},"summary":{}}"#) }
        await auth.stopSession()
        XCTAssertNil(auth.errorMessage)
    }

    func testEndpointValidation() throws {
        XCTAssertEqual(try ServerConfig.normalize(" https://test.example/ "), base)
        XCTAssertEqual(try ServerConfig.normalize("http://192.168.1.2:8000"), "http://192.168.1.2:8000")
        for value in ["", "http://public.example", "https://user:pass@test.example", "https://test.example/app", "https://test.example?key=secret", "file:///tmp/file"] {
            XCTAssertThrowsError(try ServerConfig.normalize(value))
        }
    }
}

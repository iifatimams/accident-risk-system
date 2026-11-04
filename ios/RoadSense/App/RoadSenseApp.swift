import SwiftUI

@main
struct RoadSenseApp: App {
    @StateObject private var captureService = BackgroundCaptureService()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(captureService)
        }
    }
}
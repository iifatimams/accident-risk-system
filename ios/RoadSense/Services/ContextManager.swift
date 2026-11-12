import Foundation
import CoreLocation

public class ContextManager {
    public static let shared = ContextManager()

    // Cached State
    public var weatherMultiplier: Double = WeatherResult.unknown.multiplier
    public var weatherConditionCode: Double = WeatherResult.unknown.conditionCode
    public var isRoundabout: Bool = false
    public var designSpeed: Double = RoadSpeedResult.unknown.designSpeedKmh
    // Phase 10: whether the values above are real, live observations or
    // still the training-aligned "unknown" fallback -- see RoadContext.swift
    // for why this distinction matters (never silently label a guess as an
    // official limit or a real weather observation).
    public var isDesignSpeedFromMapData: Bool = false
    public var isWeatherLiveData: Bool = false

    private var lastWeatherFetch: TimeInterval = 0
    private var lastMapFetch: TimeInterval = 0
    private var lastLocation: CLLocationCoordinate2D?

    // Add your OpenWeather API key here if you have one, or leave as is to safely default to CLEAR.
    private let weatherApiKey = "YOUR_OPENWEATHER_API_KEY_HERE"

    public func updateLocation(lat: Double, lon: Double) {
        let now = Date().timeIntervalSince1970

        // 1. Weather Update (Every 5 minutes)
        if now - lastWeatherFetch > 300 {
            lastWeatherFetch = now
            fetchWeather(lat: lat, lon: lon)
        }

        // 2. Map Update (Every 10 seconds or significant movement)
        var movedFar = false
        if let last = lastLocation {
            // Rough distance check (1 degree is ~111km)
            let dLat = (lat - last.latitude) * 111000
            let dLon = (lon - last.longitude) * 111000
            let dist = sqrt(dLat*dLat + dLon*dLon)
            if dist > 50 { movedFar = true }
        } else {
            movedFar = true
        }

        if movedFar || (now - lastMapFetch > 10) {
            lastMapFetch = now
            lastLocation = CLLocationCoordinate2D(latitude: lat, longitude: lon)
            fetchMap(lat: lat, lon: lon)
        }
    }

    private func fetchWeather(lat: Double, lon: Double) {
        // No API key configured -- this is the common case (the key is
        // still the placeholder string by default). Previously this left
        // weatherConditionCode at whatever it was (its own default, 6.0)
        // with no way to tell that from a real "cloudy" observation. Now
        // explicit: isWeatherLiveData stays false until a real fetch
        // actually succeeds.
        guard weatherApiKey != "YOUR_OPENWEATHER_API_KEY_HERE" else { return }
        let urlStr = "https://api.openweathermap.org/data/2.5/weather?lat=\(lat)&lon=\(lon)&appid=\(weatherApiKey)"
        guard let url = URL(string: urlStr) else { return }

        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weatherArray = json["weather"] as? [[String: Any]],
                  let main = weatherArray.first?["main"] as? String else { return }

            let result = classifyWeatherCondition(main)
            DispatchQueue.main.async {
                self.weatherMultiplier = result.multiplier
                self.weatherConditionCode = result.conditionCode
                self.isWeatherLiveData = result.isLiveData
            }
        }.resume()
    }

    private func fetchMap(lat: Double, lon: Double) {
        let query = "[out:json][timeout:2];(way(around:80,\(lat),\(lon))[\"highway\"];);out tags;"
        guard let urlStr = "https://overpass-api.de/api/interpreter?data=\(query)".addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: urlStr) else { return }

        var request = URLRequest(url: url)
        request.timeoutInterval = 2.0

        URLSession.shared.dataTask(with: request) { data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let elements = json["elements"] as? [[String: Any]] else { return }

            let tagDicts: [[String: String]] = elements.compactMap { $0["tags"] as? [String: String] }
            let result = parseRoadSpeedFromOverpassTags(tagDicts)

            DispatchQueue.main.async {
                self.isRoundabout = result.isRoundabout
                self.designSpeed = result.designSpeedKmh
                self.isDesignSpeedFromMapData = result.isFromMapData
            }
        }.resume()
    }
}

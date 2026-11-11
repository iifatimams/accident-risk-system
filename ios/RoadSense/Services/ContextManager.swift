import Foundation
import CoreLocation

public class ContextManager {
    public static let shared = ContextManager()
    
    // Cached State
    public var weatherMultiplier: Double = 1.0
    public var weatherConditionCode: Double = 6.0 // Default: 6.0 (Cloudy)
    public var isRoundabout: Bool = false
    public var designSpeed: Double = 80.0
    
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
        guard weatherApiKey != "YOUR_OPENWEATHER_API_KEY_HERE" else { return }
        let urlStr = "https://api.openweathermap.org/data/2.5/weather?lat=\(lat)&lon=\(lon)&appid=\(weatherApiKey)"
        guard let url = URL(string: urlStr) else { return }
        
        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weatherArray = json["weather"] as? [[String: Any]],
                  let main = weatherArray.first?["main"] as? String else { return }
            
            let condition = main.uppercased()
            var mult = 1.0
            var code = 6.0 // Cloudy
            
            if condition == "RAIN" || condition == "DRIZZLE" { mult = 1.25; code = 8.0 }
            else if condition == "THUNDERSTORM" { mult = 1.40; code = 8.0 }
            else if ["MIST", "FOG", "SAND", "DUST", "HAZE"].contains(condition) { mult = 1.15; code = 7.0 }
            else if condition == "CLEAR" { code = 0.0 }
            
            DispatchQueue.main.async {
                self.weatherMultiplier = mult
                self.weatherConditionCode = code
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
            
            var rbt = false
            var spd = 80.0
            
            for el in elements {
                if let tags = el["tags"] as? [String: String] {
                    if tags["junction"] == "roundabout" || tags["highway"] == "roundabout" { rbt = true }
                    if let maxspeedStr = tags["maxspeed"], let s = Double(maxspeedStr.filter(\.isNumber)) { spd = s }
                }
            }
            
            DispatchQueue.main.async {
                self.isRoundabout = rbt
                self.designSpeed = spd
            }
        }.resume()
    }
}
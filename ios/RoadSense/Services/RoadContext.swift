import Foundation

/// Phase 10: pure, testable pieces of ContextManager's map/weather logic,
/// pulled out specifically to fix and pin down two real bugs found while
/// auditing it:
///
/// 1. ContextManager previously defaulted/fell back `designSpeed` to 80.0
///    km/h whenever no OSM `maxspeed` tag was found -- but
///    scripts/train_polidriving.py's own missing-value imputation for
///    design_speed is 70.0, not 80.0 (confirmed by reading that script).
///    RoadSenseOnDeviceRiskEngine.swift's own comment on its OTHER
///    missing-feature defaults already warns about exactly this: "any
///    mismatch here biases every live prediction away from what the
///    model was actually trained to expect for 'unknown.'" 80.0 was that
///    mismatch, just undiscovered until now.
/// 2. There was no way to tell a REAL OSM-sourced speed limit apart from
///    a guessed/fallback one at all -- the project directive explicitly
///    warns against silently labeling a guessed fallback speed as an
///    official legal limit. `RoadSpeedResult.isFromMapData` exists so a
///    dashboard/export can show this honestly; it does not change what
///    value is fed to the model (changing the number itself, rather than
///    just labeling its confidence, would risk moving it further from
///    what training expects).
public struct RoadSpeedResult: Equatable {
  public var designSpeedKmh: Double
  public var isRoundabout: Bool
  /// True only when an actual `maxspeed` tag was found on a nearby way.
  /// False for "no tag found" and "request failed/not yet fetched" alike
  /// -- both are a guess, just via different mechanisms.
  public var isFromMapData: Bool

  public init(designSpeedKmh: Double, isRoundabout: Bool, isFromMapData: Bool) {
    self.designSpeedKmh = designSpeedKmh
    self.isRoundabout = isRoundabout
    self.isFromMapData = isFromMapData
  }

  /// Matches scripts/train_polidriving.py's design_speed missing-value
  /// default exactly (70.0, not 80.0) -- see this file's own doc comment.
  public static let trainingDefaultDesignSpeedKmh = 70.0

  public static let unknown = RoadSpeedResult(
    designSpeedKmh: trainingDefaultDesignSpeedKmh, isRoundabout: false, isFromMapData: false
  )
}

/// Parses the Overpass "elements" response used by ContextManager.fetchMap
/// into a RoadSpeedResult. Pure/testable: takes already-decoded tag
/// dictionaries, not raw JSON or a network call.
public func parseRoadSpeedFromOverpassTags(_ elements: [[String: String]]) -> RoadSpeedResult {
  var isRoundabout = false
  var designSpeed: Double?
  for tags in elements {
    if tags["junction"] == "roundabout" || tags["highway"] == "roundabout" {
      isRoundabout = true
    }
    if let maxspeedStr = tags["maxspeed"] {
      let digits = maxspeedStr.filter { $0.isNumber }
      if let parsed = Double(digits) {
        designSpeed = parsed
      }
    }
  }
  if let designSpeed {
    return RoadSpeedResult(designSpeedKmh: designSpeed, isRoundabout: isRoundabout, isFromMapData: true)
  }
  // No maxspeed tag found on any nearby way -- NOT a real limit. Use the
  // same missing-value convention the model was trained with, not an
  // arbitrary guess, and flag it as not map-sourced.
  return RoadSpeedResult(
    designSpeedKmh: RoadSpeedResult.trainingDefaultDesignSpeedKmh, isRoundabout: isRoundabout, isFromMapData: false
  )
}

public struct WeatherResult: Equatable {
  public var multiplier: Double
  /// Matches scripts/train_polidriving.py's WEATHER_MAP code for the
  /// condition, or its own missing-value default (6.0, "cloudy") when
  /// the condition string isn't one ContextManager recognizes.
  public var conditionCode: Double
  public var isLiveData: Bool

  public init(multiplier: Double, conditionCode: Double, isLiveData: Bool) {
    self.multiplier = multiplier
    self.conditionCode = conditionCode
    self.isLiveData = isLiveData
  }

  /// Matches scripts/train_polidriving.py's current_weather missing-value
  /// default ("cloudy" -> WEATHER_MAP["cloudy"] == 6.0) -- used both when
  /// no API key is configured (fetchWeather never even tries) and when a
  /// request fails, so the live feature never silently uses a different
  /// "unknown" convention than the one the model was trained with.
  public static let unknown = WeatherResult(multiplier: 1.0, conditionCode: 6.0, isLiveData: false)
}

/// Maps a provider weather condition string (e.g. OpenWeatherMap's
/// `weather[0].main`) to the risk multiplier and training-aligned
/// WEATHER_MAP code ContextManager.fetchWeather previously computed
/// inline. Pure/testable; the provider-specific parsing (pulling `main`
/// out of the JSON body) stays in ContextManager.
public func classifyWeatherCondition(_ condition: String) -> WeatherResult {
  switch condition.uppercased() {
  case "RAIN", "DRIZZLE":
    return WeatherResult(multiplier: 1.25, conditionCode: 8.0, isLiveData: true)
  case "THUNDERSTORM":
    return WeatherResult(multiplier: 1.40, conditionCode: 8.0, isLiveData: true)
  case "MIST", "FOG", "SAND", "DUST", "HAZE":
    return WeatherResult(multiplier: 1.15, conditionCode: 7.0, isLiveData: true)
  case "CLEAR":
    return WeatherResult(multiplier: 1.0, conditionCode: 0.0, isLiveData: true)
  default:
    // Any other real, observed condition (e.g. "CLOUDS") reads as the
    // same "cloudy" code the model was trained to treat as unremarkable
    // -- this IS live data, just an unmapped condition string, which is
    // different from never having fetched anything at all (WeatherResult.unknown).
    return WeatherResult(multiplier: 1.0, conditionCode: 6.0, isLiveData: true)
  }
}

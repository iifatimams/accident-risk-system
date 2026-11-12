import XCTest
@testable import RoadSenseAuth

final class RoadContextTests: XCTestCase {
    func testNoElementsFallsBackToTrainingDefaultNotEighty() {
        // The actual bug this session found: the old code fell back to
        // 80.0 km/h, but scripts/train_polidriving.py's own missing-value
        // default for design_speed is 70.0.
        let result = parseRoadSpeedFromOverpassTags([])
        XCTAssertEqual(result.designSpeedKmh, 70.0, accuracy: 1e-9)
        XCTAssertFalse(result.isFromMapData, "no tag found must not be reported as real map data")
    }

    func testRealMaxspeedTagIsUsedAndFlaggedAsMapData() {
        let result = parseRoadSpeedFromOverpassTags([["highway": "primary", "maxspeed": "100"]])
        XCTAssertEqual(result.designSpeedKmh, 100.0, accuracy: 1e-9)
        XCTAssertTrue(result.isFromMapData)
    }

    func testMaxspeedTagWithUnitsStillParsesTheDigits() {
        let result = parseRoadSpeedFromOverpassTags([["maxspeed": "60 mph"]])
        XCTAssertEqual(result.designSpeedKmh, 60.0, accuracy: 1e-9)
        XCTAssertTrue(result.isFromMapData)
    }

    func testRoundaboutJunctionTagIsDetectedRegardlessOfMaxspeed() {
        let result = parseRoadSpeedFromOverpassTags([["junction": "roundabout"]])
        XCTAssertTrue(result.isRoundabout)
        XCTAssertFalse(result.isFromMapData, "a roundabout tag alone, with no maxspeed, is still a guessed speed")
    }

    func testHighwayRoundaboutTagIsAlsoDetected() {
        let result = parseRoadSpeedFromOverpassTags([["highway": "roundabout"]])
        XCTAssertTrue(result.isRoundabout)
    }

    func testUnknownConstantMatchesTheNoElementsCase() {
        XCTAssertEqual(RoadSpeedResult.unknown, parseRoadSpeedFromOverpassTags([]))
    }
}

final class ClassifyWeatherConditionTests: XCTestCase {
    func testClearMapsToTrainingSunnyCode() {
        let result = classifyWeatherCondition("Clear")
        XCTAssertEqual(result.conditionCode, 0.0)
        XCTAssertEqual(result.multiplier, 1.0, accuracy: 1e-9)
        XCTAssertTrue(result.isLiveData)
    }

    func testRainAndDrizzleBothMapToTheRainCodeWithTheSameMultiplier() {
        let rain = classifyWeatherCondition("RAIN")
        let drizzle = classifyWeatherCondition("drizzle")
        XCTAssertEqual(rain.conditionCode, 8.0)
        XCTAssertEqual(drizzle.conditionCode, 8.0)
        XCTAssertEqual(rain.multiplier, drizzle.multiplier, accuracy: 1e-9)
    }

    func testThunderstormHasTheHighestMultiplier() {
        let thunder = classifyWeatherCondition("Thunderstorm")
        let rain = classifyWeatherCondition("Rain")
        XCTAssertGreaterThan(thunder.multiplier, rain.multiplier)
    }

    func testLowVisibilityConditionsMapToTheFogCode() {
        for condition in ["Mist", "Fog", "Sand", "Dust", "Haze"] {
            let result = classifyWeatherCondition(condition)
            XCTAssertEqual(result.conditionCode, 7.0, "\(condition) should map to the fog code")
            XCTAssertTrue(result.isLiveData)
        }
    }

    func testUnrecognizedConditionIsStillLiveDataAtTheCloudyCode() {
        // "Clouds" from a real provider response -- not in the explicit
        // switch, but it WAS a real fetch, unlike WeatherResult.unknown.
        let result = classifyWeatherCondition("Clouds")
        XCTAssertEqual(result.conditionCode, 6.0)
        XCTAssertTrue(result.isLiveData, "an unmapped but real condition string is still live data")
    }

    func testUnknownConstantIsNotLiveData() {
        XCTAssertFalse(WeatherResult.unknown.isLiveData)
        XCTAssertEqual(WeatherResult.unknown.conditionCode, 6.0)
        XCTAssertEqual(WeatherResult.unknown.multiplier, 1.0, accuracy: 1e-9)
    }
}

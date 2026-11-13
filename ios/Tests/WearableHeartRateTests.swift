import XCTest
@testable import RoadSenseAuth

/// Phase 15: "test with mocked BLE packets first" -- these ARE those
/// mocked packets. No physical WHOOP or Amazfit strap was used; see
/// AmazfitClient.swift for exactly what remains unverified on real
/// hardware.
final class WearableHeartRateTests: XCTestCase {
    func testUInt8BpmWithNoOptionalFields() {
        // flags=0x00 (UInt8 BPM, no energy, no RR), bpm=72.
        let packet: [UInt8] = [0x00, 72]
        let reading = WearableHeartRate.parseHeartRateMeasurement(packet)
        XCTAssertEqual(reading?.heartRateBpm, 72)
        XCTAssertNil(reading?.rrIntervalMs)
    }

    func testUInt16BpmAboveTwoFiftyFive() {
        // A UInt8 field could never represent 300 bpm -- confirms the
        // UInt16 flag bit is actually honored, not just accepted.
        let packet = WearableHeartRate.mockPacket(bpm: 300)
        let reading = WearableHeartRate.parseHeartRateMeasurement(packet)
        XCTAssertEqual(reading?.heartRateBpm, 300)
    }

    func testRRIntervalRoundTripsThroughTheMockPacketBuilder() {
        let packet = WearableHeartRate.mockPacket(bpm: 65, rrIntervalMs: 920.0)
        let reading = WearableHeartRate.parseHeartRateMeasurement(packet)
        XCTAssertEqual(reading?.heartRateBpm, 65)
        XCTAssertEqual(reading?.rrIntervalMs ?? 0, 920.0, accuracy: 2.0)  // 1/1024s quantization
    }

    func testEnergyExpendedFieldIsSkippedNotMisreadAsRRInterval() {
        // flags = UInt16 BPM (0x01) + Energy Expended present (0x08), NO RR
        // bit set. Byte layout: [flags, bpmLo, bpmHi, energyLo, energyHi].
        let packet: [UInt8] = [0x09, 80, 0x00, 0xFF, 0xFF]
        let reading = WearableHeartRate.parseHeartRateMeasurement(packet)
        XCTAssertEqual(reading?.heartRateBpm, 80)
        XCTAssertNil(reading?.rrIntervalMs, "RR bit was not set -- energy-expended bytes must not be misread as RR")
    }

    func testEnergyExpendedThenRRIntervalBothPresentAreReadCorrectly() {
        // flags = UInt16 BPM (0x01) + Energy (0x08) + RR (0x10).
        // [flags, bpmLo, bpmHi, energyLo, energyHi, rrLo, rrHi]
        let rrRaw: UInt16 = 1024  // exactly 1.000s
        let packet: [UInt8] = [0x19, 70, 0x00, 0, 0, UInt8(rrRaw & 0xFF), UInt8(rrRaw >> 8)]
        let reading = WearableHeartRate.parseHeartRateMeasurement(packet)
        XCTAssertEqual(reading?.heartRateBpm, 70)
        XCTAssertEqual(reading?.rrIntervalMs ?? 0, 1000.0, accuracy: 1e-6)
    }

    func testEmptyPacketReturnsNilNotAFabricatedReading() {
        XCTAssertNil(WearableHeartRate.parseHeartRateMeasurement([]))
    }

    func testTruncatedUInt16PacketReturnsNil() {
        // Flags claim UInt16 BPM, but only one more byte is actually present.
        XCTAssertNil(WearableHeartRate.parseHeartRateMeasurement([0x01, 72]))
    }

    func testZeroBpmIsTreatedAsNoReadingNotAFabricatedZero() {
        XCTAssertNil(WearableHeartRate.parseHeartRateMeasurement([0x00, 0]))
    }
}

import Foundation

/// Phase 15: parsing for the standard Bluetooth GATT Heart Rate Measurement
/// characteristic (service 0x180D, characteristic 0x2A37) -- shared by
/// WhoopClient.swift and AmazfitClient.swift, since both devices are
/// expected to expose the SAME standard service (the directive's own
/// instruction: "investigate the Helio Strap's Heart Rate Push setting and
/// compatibility with standard Bluetooth Heart Rate Service broadcasts...
/// use supported device functionality, not undocumented cloud-password
/// scraping"). Pure/testable with synthetic byte packets -- "test with
/// mocked BLE packets first," per the same directive -- since this file has
/// no CoreBluetooth dependency of its own.
///
/// Format (Bluetooth SIG GATT Heart Rate Measurement, 0x2A37): byte 0 is a
/// flags bitfield; bit 0 selects UInt8 vs UInt16 BPM; bit 3 marks optional
/// Energy Expended (UInt16, skipped, never used here); bit 4 marks one or
/// more optional RR-Interval fields (UInt16 each, 1/1024 second units) --
/// only the FIRST is read, matching WhoopClient's pre-existing behavior.
public enum WearableHeartRate {
  public struct Reading: Equatable {
    public var heartRateBpm: Double
    public var rrIntervalMs: Double?

    public init(heartRateBpm: Double, rrIntervalMs: Double?) {
      self.heartRateBpm = heartRateBpm
      self.rrIntervalMs = rrIntervalMs
    }
  }

  /// Returns nil for an empty or too-short packet; never invents a 0 BPM
  /// reading or a fabricated RR interval that bit 4 didn't actually signal.
  public static func parseHeartRateMeasurement(_ bytes: [UInt8]) -> Reading? {
    guard let flags = bytes.first else {
      return nil
    }
    let isUInt16 = (flags & 0x01) != 0
    var offset = 1
    var bpm: Double

    if isUInt16 {
      guard bytes.count >= 3 else {
        return nil
      }
      bpm = Double(UInt16(bytes[1]) | (UInt16(bytes[2]) << 8))
      offset = 3
    } else {
      guard bytes.count >= 2 else {
        return nil
      }
      bpm = Double(bytes[1])
      offset = 2
    }

    // Energy Expended present (bit 3): a UInt16 field to skip over, not used.
    if (flags & 0x08) != 0 {
      offset += 2
    }

    var rrIntervalMs: Double?
    // RR-Interval present (bit 4): one or more UInt16 values in 1/1024s
    // units; only the first is read.
    if (flags & 0x10) != 0, bytes.count >= offset + 2 {
      let raw = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
      rrIntervalMs = (Double(raw) / 1024.0) * 1000.0
    }

    guard bpm > 0 else {
      return nil
    }
    return Reading(heartRateBpm: bpm, rrIntervalMs: rrIntervalMs)
  }

  // MARK: - Test fixtures (mocked BLE packets)
  // Builds a real, spec-correct Heart Rate Measurement payload -- used by
  // this file's own tests, and reusable by any future test that needs a
  // believable packet without a real strap. Never used by production code.
  public static func mockPacket(bpm: UInt16, rrIntervalMs: Double? = nil) -> [UInt8] {
    var flags: UInt8 = 0x01  // UInt16 BPM, for round-trip-safe values > 255
    var bytes: [UInt8] = []
    if let rrIntervalMs {
      flags |= 0x10
      let raw = UInt16((rrIntervalMs / 1000.0) * 1024.0)
      bytes = [flags, UInt8(bpm & 0xFF), UInt8(bpm >> 8), UInt8(raw & 0xFF), UInt8(raw >> 8)]
    } else {
      bytes = [flags, UInt8(bpm & 0xFF), UInt8(bpm >> 8)]
    }
    return bytes
  }
}

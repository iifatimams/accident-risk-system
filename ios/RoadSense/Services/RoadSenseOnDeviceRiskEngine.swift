// RoadSense/Services/RoadSenseOnDeviceRiskEngine.swift
import CoreML
import Foundation

public final class RoadSenseOnDeviceRiskEngine {
    private let vzcrashModel: VZCrashModel?
    private let fusionModel: FusionModel?
    // History buffers for proactive kinematics
    private var accelHistory: [(time: Double, val: Double)] = []
    private var hrHistory: [(time: Double, val: Double)] = []

    public init() {
        let config = MLModelConfiguration()
        self.vzcrashModel = try? VZCrashModel(configuration: config)
        self.fusionModel = try? FusionModel(configuration: config)

        #if DEBUG
        if vzcrashModel == nil {
            print("[RoadSense] Failed to load VZCrashModel")
        }
        if fusionModel == nil {
            print("[RoadSense] Failed to load FusionModel")
        }
        #endif
    }

    public func predictRisk(
        effectiveAccel: Double,
        gyroMag: Double,
        speedKmh: Double,
        accelObd: Double,
        rpm: Double,
        throttle: Double,
        load: Double,
        hr: Double,
        eyesOnRoad: Bool,
        eyesClosedSec: Double,
        trafficLevel: String,
        pedestriansDetected: Bool,
        leadClosingFast: Bool,
        inRoundabout: Bool,
        weatherMult: Double,
        weatherCode: Double,
        designSpeed: Double,
        isDriving: Bool
    ) -> Double {
        guard isDriving else {
            accelHistory.removeAll()
            hrHistory.removeAll()
            return 0.0
        }
        let ml = mlComponents(
            effectiveAccel: effectiveAccel, gyroMag: gyroMag, speedKmh: speedKmh,
            accelObd: accelObd, rpm: rpm, throttle: throttle, load: load, hr: hr,
            weatherCode: weatherCode, designSpeed: designSpeed
        )
        return breakdown(for: ml, speedKmh: speedKmh, eyesOnRoad: eyesOnRoad,
                          eyesClosedSec: eyesClosedSec, trafficLevel: trafficLevel,
                          pedestriansDetected: pedestriansDetected, leadClosingFast: leadClosingFast,
                          gyroMag: gyroMag, effectiveAccel: effectiveAccel,
                          inRoundabout: inRoundabout, weatherMult: weatherMult).total
    }

    /// Same computation as predictRisk, but returns every component
    /// (Phase 9.1's "explainable risk breakdown") instead of only the
    /// final clamped number -- for a dashboard/debug view to show WHY a
    /// score is what it is, not just the number itself. Shares the exact
    /// same ML-inference and history-buffer code path as predictRisk (via
    /// mlComponents/breakdown below) -- there is no second, divergeable
    /// copy of that logic.
    public func predictRiskBreakdown(
        effectiveAccel: Double,
        gyroMag: Double,
        speedKmh: Double,
        accelObd: Double,
        rpm: Double,
        throttle: Double,
        load: Double,
        hr: Double,
        eyesOnRoad: Bool,
        eyesClosedSec: Double,
        trafficLevel: String,
        pedestriansDetected: Bool,
        leadClosingFast: Bool,
        inRoundabout: Bool,
        weatherMult: Double,
        weatherCode: Double,
        designSpeed: Double,
        isDriving: Bool
    ) -> RiskBreakdown {
        guard isDriving else {
            accelHistory.removeAll()
            hrHistory.removeAll()
            return RiskBreakdown()
        }
        let ml = mlComponents(
            effectiveAccel: effectiveAccel, gyroMag: gyroMag, speedKmh: speedKmh,
            accelObd: accelObd, rpm: rpm, throttle: throttle, load: load, hr: hr,
            weatherCode: weatherCode, designSpeed: designSpeed
        )
        return breakdown(for: ml, speedKmh: speedKmh, eyesOnRoad: eyesOnRoad,
                          eyesClosedSec: eyesClosedSec, trafficLevel: trafficLevel,
                          pedestriansDetected: pedestriansDetected, leadClosingFast: leadClosingFast,
                          gyroMag: gyroMag, effectiveAccel: effectiveAccel,
                          inRoundabout: inRoundabout, weatherMult: weatherMult)
    }

    private struct MLComponents {
        var jerk: Double
        var hrVelocity: Double
        var vzRisk: Double
        var poliRisk: Double
        var classPred: Int
    }

    /// Steps 1-4 of the original monolithic predictRisk: maintain the
    /// rolling history buffers, compute the jerk/hrVelocity derivatives,
    /// and run both CoreML models. Mutates accelHistory/hrHistory as a
    /// side effect (unchanged from the original behavior) -- called
    /// exactly once per predictRisk/predictRiskBreakdown invocation, never
    /// twice, so history/inference never double-advances for one frame.
    private func mlComponents(
        effectiveAccel: Double, gyroMag: Double, speedKmh: Double, accelObd: Double,
        rpm: Double, throttle: Double, load: Double, hr: Double,
        weatherCode: Double, designSpeed: Double
    ) -> MLComponents {
        let now = Date().timeIntervalSince1970
        accelHistory.append((time: now, val: effectiveAccel))
        hrHistory.append((time: now, val: hr))
        accelHistory.removeAll { now - $0.time > 30.0 }
        hrHistory.removeAll { now - $0.time > 30.0 }

        // jerk/hrVelocity are (last-first)/(last.time-first.time) across
        // the whole rolling buffer. That's only a meaningful derivative
        // once the buffer spans a real window: snapshots arrive roughly
        // every 0.3s, so right after the buffer is rebuilt (trip start,
        // or any gap that caused a reset) it can contain just 1-2 samples
        // spanning a fraction of a second. Dividing an ordinary braking
        // delta by a near-zero denominator produces a huge spurious
        // jerk/hrVelocity reading -- this is what caused a gentle brake
        // to register as a ~40% risk spike even though the underlying
        // ML fusion score was only ~15. Requiring a real minimum span
        // means we only trust the derivative once there's enough history
        // for it to actually mean something; until then it stays 0.
        let minDerivativeSpanSeconds = 1.5
        let jerk = Self.derivative(accelHistory, minSpan: minDerivativeSpanSeconds)
        let hrVelocity = Self.derivative(hrHistory, minSpan: minDerivativeSpanSeconds)

        // VZCrash ML prediction (4 features)
        var vzRisk = 0.0
        var classPred = 2
        if let vzArray = try? MLMultiArray(shape: [4], dataType: .double) {
            vzArray[0] = NSNumber(value: effectiveAccel)
            vzArray[1] = NSNumber(value: gyroMag)
            vzArray[2] = NSNumber(value: speedKmh)
            vzArray[3] = NSNumber(value: abs(accelObd))
            if let prediction = try? vzcrashModel?.prediction(features: vzArray) {
                let pCrash = prediction.classProbability[0 as Int64] ?? 0.0
                let pNearMiss = prediction.classProbability[1 as Int64] ?? 0.0
                let pNormal = prediction.classProbability[2 as Int64] ?? 0.0
                vzRisk = (1.0 * pCrash) + (0.6 * pNearMiss)
                if pCrash > pNearMiss && pCrash > pNormal {
                    classPred = 0
                } else if pNearMiss > pCrash && pNearMiss > pNormal {
                    classPred = 1
                }
            }
        }

        // POLIDriving ML prediction (14 features)
        var poliRisk = 0.0
        if let poliArray = try? MLMultiArray(shape: [14], dataType: .double) {
            let hour = Double(Calendar.current.component(.hour, from: Date()))
            // Indices 5, 7, 9, 10, 11, 13 have no live sensor source on an
            // iPhone, so they fall back to the same defaults train_polidriving.py
            // substitutes when a training row is missing that column (engine
            // temperature, heart rate, visibility, precipitation, and the two
            // local-accident-history counters) -- NOT arbitrary guesses, since
            // any mismatch here biases every live prediction away from what the
            // model was actually trained to expect for "unknown".
            let features: [Double] = [
                hour, speedKmh, rpm, accelObd, throttle, 90.0, load,
                hr > 0 ? hr : 75.0, weatherCode, 10.0, 0.0, 0.0, designSpeed, 0.0,
            ]
            for (index, value) in features.enumerated() {
                poliArray[index] = NSNumber(value: value)
            }
            if let prediction = try? fusionModel?.prediction(features: poliArray) {
                let pMedium = prediction.classProbability[1 as Int64] ?? 0.0
                let pHigh = prediction.classProbability[2 as Int64] ?? 0.0
                let pVeryHigh = prediction.classProbability[3 as Int64] ?? 0.0
                poliRisk = (0.25 * pMedium) + (0.65 * pHigh) + (1.0 * pVeryHigh)
            }
        }
        return MLComponents(jerk: jerk, hrVelocity: hrVelocity, vzRisk: vzRisk, poliRisk: poliRisk, classPred: classPred)
    }

    /// Steps 5-10: delegates to the independently-testable
    /// RiskFusionHeuristics.swift (Phase 9.1) -- see that file for why
    /// this split exists and the full per-component breakdown.
    private func breakdown(
        for ml: MLComponents, speedKmh: Double, eyesOnRoad: Bool, eyesClosedSec: Double,
        trafficLevel: String, pedestriansDetected: Bool, leadClosingFast: Bool,
        gyroMag: Double, effectiveAccel: Double, inRoundabout: Bool, weatherMult: Double
    ) -> RiskBreakdown {
        // poliRisk/vzRisk are already weighted sums of CoreML
        // classProbability values, which for these exported
        // RandomForestClassifier models are vote percentages on a
        // 0-100 scale (not 0-1 fractions) -- confirmed by probing
        // the bundled .mlpackage files directly. Do not re-multiply
        // by 100 here or the fused score saturates past the 99.9
        // clamp almost immediately on any nonzero medium/high class
        // probability.
        let mlFusionBase = ((0.5 * ml.poliRisk) + (0.5 * ml.vzRisk)) * weatherMult
        return applyRiskHeuristics(
            mlFusionBase: mlFusionBase, jerk: ml.jerk, hrVelocity: ml.hrVelocity,
            trafficLevel: trafficLevel, pedestriansDetected: pedestriansDetected,
            eyesClosedSec: eyesClosedSec, eyesOnRoad: eyesOnRoad, speedKmh: speedKmh,
            leadClosingFast: leadClosingFast, gyroMag: gyroMag, effectiveAccel: effectiveAccel,
            inRoundabout: inRoundabout, classPred: ml.classPred, weatherMult: weatherMult
        )
    }

    private static func derivative(_ history: [(time: Double, val: Double)], minSpan: Double) -> Double {
        guard let first = history.first, let last = history.last,
              last.time - first.time > minSpan
        else {
            return 0.0
        }
        return (last.val - first.val) / (last.time - first.time)
    }
}

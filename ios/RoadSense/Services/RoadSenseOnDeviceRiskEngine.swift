// RoadSense/Services/RoadSenseOnDeviceRiskEngine.swift
import CoreML
import Foundation

public final class RoadSenseOnDeviceRiskEngine {
    /// Used ONLY for collision-support evidence (classPred, surfaced as
    /// `collisionSupportEvidence`/`vzClassPred` below) -- never blended
    /// into the predictive risk magnitude. The capstone directive is
    /// explicit that VZCrash "must not be the principal contributor to
    /// the predictive risk score"; before this change `mlFusionBase` was
    /// `0.5*poliRisk + 0.5*vzRisk`, an EQUAL, not merely supporting,
    /// weight. See RiskFusionHeuristics.swift's mlFusionBase doc comment
    /// for the corrected formula.
    private let vzcrashModel: VZCrashModel?
    private let fusionModel: FusionModel?
    /// POLIDriving's separately-trained 5-second-ahead hazard model
    /// (scripts/train_polidriving_5s_ahead.py) -- same 14-feature input
    /// contract as `fusionModel`, but trained to forecast risk_level 5
    /// seconds ahead rather than classify the current instant. Reported
    /// as its own `fiveSecondAheadRisk` output field, never blended into
    /// `currentRisk`.
    private let fiveSecondModel: PolidrivingFiveSecondModel?
    // History buffers for proactive kinematics
    private var accelHistory: [(time: Double, val: Double)] = []
    private var hrHistory: [(time: Double, val: Double)] = []
    private var rollingRiskHistory = RollingRiskHistory()

    public init() {
        let config = MLModelConfiguration()
        self.vzcrashModel = try? VZCrashModel(configuration: config)
        self.fusionModel = try? FusionModel(configuration: config)
        self.fiveSecondModel = try? PolidrivingFiveSecondModel(configuration: config)

        #if DEBUG
        if vzcrashModel == nil {
            print("[RoadSense] Failed to load VZCrashModel")
        }
        if fusionModel == nil {
            print("[RoadSense] Failed to load FusionModel")
        }
        if fiveSecondModel == nil {
            print("[RoadSense] Failed to load PolidrivingFiveSecondModel")
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

    /// VZCrash's collision-support classification, independent of the
    /// predictive risk score (section 7 of the master directive: "the
    /// collision pathway must never depend on the predictive score
    /// crossing a threshold"). CollisionDetector.swift's pure
    /// acceleration-magnitude threshold is, and remains, the sole
    /// AUTHORITATIVE trigger for a suspected collision -- this struct is
    /// only ADDITIONAL discriminating evidence attached to an incident
    /// that already exists, never a gate on whether one is created.
    public enum CollisionEventClass: String, Codable {
        case crash
        case nearMiss = "near_miss"
        case normal
    }

    public struct CollisionEvidence: Equatable {
        public var predictedClass: CollisionEventClass
        /// The predicted class's own probability, normalized to [0, 1]
        /// -- an estimated impact-event CATEGORY confidence, not a
        /// medically or physically verified injury-severity
        /// classification.
        public var confidence: Double
        public var pCrash: Double
        public var pNearMiss: Double
        public var pNormal: Double
    }

    /// Runs VZCrash on the exact 4-feature snapshot at the moment
    /// CollisionDetector's threshold fired, to help discriminate a real
    /// collision from harsh braking, a pothole/speed bump, a dropped
    /// phone, rapid acceleration, or an ordinary turn (master directive
    /// section 7). Returns nil only if the model failed to load.
    public func classifyCollisionEvidence(
        effectiveAccel: Double, gyroMag: Double, speedKmh: Double, accelObd: Double
    ) -> CollisionEvidence? {
        guard let vzArray = try? MLMultiArray(shape: [4], dataType: .double) else {
            return nil
        }
        vzArray[0] = NSNumber(value: effectiveAccel)
        vzArray[1] = NSNumber(value: gyroMag)
        vzArray[2] = NSNumber(value: speedKmh)
        vzArray[3] = NSNumber(value: abs(accelObd))
        guard let prediction = try? vzcrashModel?.prediction(features: vzArray) else {
            return nil
        }
        // classProbability is a 0-100 vote-percentage scale for this
        // exported RandomForestClassifier (confirmed directly: `.venv/bin/python
        // -c "import coremltools as ct; ... m.predict(...)"` on
        // VZCrashModel.mlpackage returns probabilities summing to ~100,
        // e.g. {0: 4.77, 1: 25.95, 2: 69.28}) -- normalize to [0, 1] here
        // so CollisionEvidence.confidence is an actual probability.
        let pCrash = (prediction.classProbability[0 as Int64] ?? 0.0) / 100.0
        let pNearMiss = (prediction.classProbability[1 as Int64] ?? 0.0) / 100.0
        let pNormal = (prediction.classProbability[2 as Int64] ?? 0.0) / 100.0
        let predictedClass: CollisionEventClass
        let confidence: Double
        if pCrash >= pNearMiss && pCrash >= pNormal {
            predictedClass = .crash
            confidence = pCrash
        } else if pNearMiss >= pCrash && pNearMiss >= pNormal {
            predictedClass = .nearMiss
            confidence = pNearMiss
        } else {
            predictedClass = .normal
            confidence = pNormal
        }
        return CollisionEvidence(
            predictedClass: predictedClass, confidence: confidence,
            pCrash: pCrash, pNearMiss: pNearMiss, pNormal: pNormal
        )
    }

    private struct MLComponents {
        var jerk: Double
        var hrVelocity: Double
        var vzRisk: Double
        var poliRisk: Double
        var classPred: Int
        /// nil when the 5-second-ahead model failed to load or produced
        /// no prediction -- never defaulted to 0, which would silently
        /// read as "confidently not hazardous."
        var fiveSecondHazardProb: Double?
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

        // POLIDriving ML prediction (14 features) -- current-instant AND
        // 5-second-ahead models share the exact same feature vector,
        // since scripts/train_polidriving_5s_ahead.py was trained on the
        // identical 14-column POLIDRIVING_FEATURES order/defaults as
        // scripts/train_polidriving.py (same dataset, same loader
        // conventions, only the TARGET differs -- current risk_level vs
        // risk_level 5 seconds later).
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

        var poliRisk = 0.0
        if let poliArray = try? MLMultiArray(shape: [14], dataType: .double) {
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

        var fiveSecondHazardProb: Double?
        if let fiveSecondArray = try? MLMultiArray(shape: [14], dataType: .double) {
            for (index, value) in features.enumerated() {
                fiveSecondArray[index] = NSNumber(value: value)
            }
            if let prediction = try? fiveSecondModel?.prediction(features: fiveSecondArray) {
                // Trained classes are [0, 1] -- 1 == "hazardous_in_5s".
                fiveSecondHazardProb = prediction.classProbability[1 as Int64]
            }
        }

        return MLComponents(
            jerk: jerk, hrVelocity: hrVelocity, vzRisk: vzRisk, poliRisk: poliRisk,
            classPred: classPred, fiveSecondHazardProb: fiveSecondHazardProb
        )
    }

    /// Steps 5-10: delegates to the independently-testable
    /// RiskFusionHeuristics.swift (Phase 9.1) -- see that file for why
    /// this split exists and the full per-component breakdown.
    private func breakdown(
        for ml: MLComponents, speedKmh: Double, eyesOnRoad: Bool, eyesClosedSec: Double,
        trafficLevel: String, pedestriansDetected: Bool, leadClosingFast: Bool,
        gyroMag: Double, effectiveAccel: Double, inRoundabout: Bool, weatherMult: Double
    ) -> RiskBreakdown {
        // poliRisk is already a weighted sum of CoreML classProbability
        // values, which for this exported RandomForestClassifier are
        // vote percentages on a 0-100 scale (not 0-1 fractions) --
        // confirmed by probing the bundled .mlpackage directly. Do not
        // re-multiply by 100 here or the fused score saturates past the
        // 99.9 clamp almost immediately on any nonzero medium/high class
        // probability.
        //
        // vzRisk is DELIBERATELY NOT included in mlFusionBase. Before
        // this change the formula was `0.5*poliRisk + 0.5*vzRisk` -- an
        // EQUAL weight that made VZCrash a PRINCIPAL contributor to the
        // predictive score, directly contradicting the capstone
        // directive's explicit "VZCrash must not be the principal
        // contributor to the predictive risk score. Its primary
        // responsibility is detecting collision-related motion events
        // and supporting the post-impact safety system." VZCrash's
        // classPred is still used below (inside applyRiskHeuristics'
        // false-positive-suppression gate) as a SUPPORTING signal, which
        // is consistent with that instruction -- it just no longer
        // contributes any MAGNITUDE to the predictive total. ml.vzRisk
        // is still returned in the breakdown (as `collisionSupportEvidence`)
        // purely for observability/debugging and for BackgroundCaptureService's
        // independent collision-discrimination use, never summed in.
        let mlFusionBase = ml.poliRisk * weatherMult
        var result = applyRiskHeuristics(
            mlFusionBase: mlFusionBase, jerk: ml.jerk, hrVelocity: ml.hrVelocity,
            trafficLevel: trafficLevel, pedestriansDetected: pedestriansDetected,
            eyesClosedSec: eyesClosedSec, eyesOnRoad: eyesOnRoad, speedKmh: speedKmh,
            leadClosingFast: leadClosingFast, gyroMag: gyroMag, effectiveAccel: effectiveAccel,
            inRoundabout: inRoundabout, classPred: ml.classPred, weatherMult: weatherMult
        )
        result.collisionSupportEvidence = ml.vzRisk
        result.fiveSecondAheadVehicleHazardProb = ml.fiveSecondHazardProb
        return result
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

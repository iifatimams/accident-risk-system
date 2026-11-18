// RoadSense/Services/RoadSenseOnDeviceRiskEngine.swift
import CoreML
import Foundation

public final class RoadSenseOnDeviceRiskEngine {
    /// Used ONLY for collision-support evidence (classPred, surfaced as
    /// `collisionSupportEvidence`/`vzClassPred` below) -- never blended
    /// into the predictive risk magnitude. The capstone directive is
    /// explicit that VZCrash "must not be the principal contributor to
    /// the predictive risk score". See RiskFusionHeuristics.swift's
    /// mlFusionBase doc comment for the current formula.
    private let vzcrashModel: VZCrashModel?
    /// POLIDriving's 5-second-ahead hazard model
    /// (scripts/train_polidriving_5s_ahead.py), driver-held-out tested
    /// (0.918 ROC-AUC) -- the sole POLIDriving model deployed on-device.
    /// The earlier current-instant classifier (scripts/train_polidriving.py
    /// -> FusionModel.mlpackage) was retired: it was trained with a plain
    /// random train_test_split, not a driver-held-out one, so its accuracy
    /// could not be honestly compared to this one, and keeping two
    /// separately-reported scores (an unvalidated "now" number and a
    /// validated-but-ignored "+5s" number) was actively misleading. This
    /// model's probability now feeds `mlFusionBase` directly, and
    /// `MultimodalRiskFusion.fuseMultimodalRisk`'s `fiveSecondAheadRisk`
    /// is the one authoritative live score (see BackgroundCaptureService.swift).
    private let fiveSecondModel: PolidrivingFiveSecondModel?
    // History buffers for proactive kinematics
    private var accelHistory: [(time: Double, val: Double)] = []
    private var hrHistory: [(time: Double, val: Double)] = []

    public init() {
        let config = MLModelConfiguration()
        self.vzcrashModel = try? VZCrashModel(configuration: config)
        self.fiveSecondModel = try? PolidrivingFiveSecondModel(configuration: config)

        #if DEBUG
        if vzcrashModel == nil {
            print("[RoadSense] Failed to load VZCrashModel")
        }
        if fiveSecondModel == nil {
            print("[RoadSense] Failed to load PolidrivingFiveSecondModel")
        }
        #endif
    }

    /// Returns every component of the risk computation
    /// (Phase 9.1's "explainable risk breakdown") instead of only the
    /// final clamped number -- for a dashboard/debug view to show WHY a
    /// score is what it is, not just the number itself.
    public func predictRiskBreakdown(
        effectiveAccel: Double,
        gyroMag: Double,
        speedKmh: Double,
        accelObd: Double,
        rpm: Double?,
        throttle: Double?,
        load: Double?,
        hr: Double?,
        eyesOnRoad: Bool,
        eyesClosedSec: Double,
        trafficLevel: String,
        pedestriansDetected: Bool,
        leadClosingFast: Bool,
        inRoundabout: Bool,
        weatherMult: Double,
        weatherCode: Double,
        designSpeed: Double,
        isDriving: Bool,
        postureCollapse: Bool = false
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
                          inRoundabout: inRoundabout, weatherMult: weatherMult,
                          postureCollapse: postureCollapse)
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
        rpm: Double?, throttle: Double?, load: Double?, hr: Double?,
        weatherCode: Double, designSpeed: Double
    ) -> MLComponents {
        let now = Date().timeIntervalSince1970
        accelHistory.append((time: now, val: effectiveAccel))
        // A disconnected wearable must not be recorded as a 0 bpm reading --
        // that would plant a fake "heart rate just dropped to zero" sample
        // in the history, which then reads as a huge spurious hrVelocity
        // the moment a real reading returns and the derivative spans that
        // fake zero. Simply not appending anything this frame means the
        // derivative is computed across a real gap in time instead, which
        // Self.derivative's own minimum-span guard already handles.
        if let hr {
            hrHistory.append((time: now, val: hr))
        }
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

        // POLIDriving ML prediction (14 features) -- the 5-second-ahead
        // model's feature vector, same POLIDRIVING_FEATURES order/defaults
        // scripts/train_polidriving_5s_ahead.py was trained on.
        let hour = Double(Calendar.current.component(.hour, from: Date()))
        // Indices 5, 7, 9, 10, 11, 13 have no live sensor source on an
        // iPhone, so they fall back to the same defaults train_polidriving.py
        // substitutes when a training row is missing that column (engine
        // temperature, heart rate, visibility, precipitation, and the two
        // local-accident-history counters) -- NOT arbitrary guesses, since
        // any mismatch here biases every live prediction away from what the
        // model was actually trained to expect for "unknown". Heart rate is
        // the one column of that set that CAN be live (via a connected
        // wearable); the same 75 bpm training-time default is used only
        // when it genuinely is not, never a fabricated 0.
        //
        // rpm/throttle/load have no such "no live sensor source" status --
        // they ARE live OBD-II readings whenever the adapter is connected,
        // so unlike the six indices above, there is no defensible trained
        // default to substitute for them: an adapter that never connects
        // (e.g. a vehicle whose OBD-II port does not expose these
        // parameters at all) would otherwise silently look to the model
        // like an engine reading 0 RPM/0% throttle/0% load while the car is
        // actually moving at highway speed, a combination the model never
        // saw in training and that is not an honest "unknown" either. When
        // any of the three is unavailable, the POLIDriving model is
        // skipped entirely for this frame rather than fed a fabricated
        // reading -- fiveSecondHazardProb stays at its "unavailable" value
        // (nil), the same convention the rest of this engine already uses
        // for a component that cannot be scored.
        var fiveSecondHazardProb: Double?
        if let rpm, let throttle, let load {
            let hrForModel = hr.map { $0 > 0 ? $0 : 75.0 } ?? 75.0
            let features: [Double] = [
                hour, speedKmh, rpm, accelObd, throttle, 90.0, load,
                hrForModel, weatherCode, 10.0, 0.0, 0.0, designSpeed, 0.0,
            ]

            if let fiveSecondArray = try? MLMultiArray(shape: [14], dataType: .double) {
                for (index, value) in features.enumerated() {
                    fiveSecondArray[index] = NSNumber(value: value)
                }
                if let prediction = try? fiveSecondModel?.prediction(features: fiveSecondArray) {
                    // Trained classes are [0, 1] -- 1 == "hazardous_in_5s".
                    fiveSecondHazardProb = prediction.classProbability[1 as Int64]
                }
            }
        }

        return MLComponents(
            jerk: jerk, hrVelocity: hrVelocity, vzRisk: vzRisk,
            classPred: classPred, fiveSecondHazardProb: fiveSecondHazardProb
        )
    }

    /// Steps 5-10: delegates to the independently-testable
    /// RiskFusionHeuristics.swift (Phase 9.1) -- see that file for why
    /// this split exists and the full per-component breakdown.
    private func breakdown(
        for ml: MLComponents, speedKmh: Double, eyesOnRoad: Bool, eyesClosedSec: Double,
        trafficLevel: String, pedestriansDetected: Bool, leadClosingFast: Bool,
        gyroMag: Double, effectiveAccel: Double, inRoundabout: Bool, weatherMult: Double,
        postureCollapse: Bool = false
    ) -> RiskBreakdown {
        // fiveSecondHazardProb is a true [0, 1] probability (confirmed by
        // probing PolidrivingFiveSecondModel.mlpackage directly -- this
        // model's classProbability output sums to ~1.0, unlike the retired
        // current-instant FusionModel, which summed to ~100). Scaling by
        // 100 here keeps it on the same 0-99.9-ish scale the rest of this
        // engine's heuristic bonuses already use.
        //
        // vzRisk is DELIBERATELY NOT included in mlFusionBase. The capstone
        // directive is explicit that "VZCrash must not be the principal
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
        let mlFusionBase = (ml.fiveSecondHazardProb ?? 0.0) * 100.0 * weatherMult
        var result = applyRiskHeuristics(
            mlFusionBase: mlFusionBase, jerk: ml.jerk, hrVelocity: ml.hrVelocity,
            trafficLevel: trafficLevel, pedestriansDetected: pedestriansDetected,
            eyesClosedSec: eyesClosedSec, eyesOnRoad: eyesOnRoad, speedKmh: speedKmh,
            leadClosingFast: leadClosingFast, gyroMag: gyroMag, effectiveAccel: effectiveAccel,
            inRoundabout: inRoundabout, classPred: ml.classPred, weatherMult: weatherMult,
            postureCollapse: postureCollapse
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

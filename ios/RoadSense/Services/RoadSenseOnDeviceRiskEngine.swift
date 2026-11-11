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
        // 0. Driving state validation
        guard isDriving else {
            accelHistory.removeAll()
            hrHistory.removeAll()
            return 0.0
        }

        let now = Date().timeIntervalSince1970

        // 1. Maintain history buffers (30 seconds)
        accelHistory.append((
            time: now,
            val: effectiveAccel
        ))
        hrHistory.append((
            time: now,
            val: hr
        ))

        accelHistory.removeAll {
            now - $0.time > 30.0
        }
        hrHistory.removeAll {
            now - $0.time > 30.0
        }

        // 2. Compute predictive metrics
        //
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
        // (matching distractionThresholdSeconds-scale timing) means we
        // only trust the derivative once there's enough history for it
        // to actually mean something; until then it stays 0 rather than
        // amplifying noise.
        let minDerivativeSpanSeconds = 1.5
        var jerk = 0.0
        var hrVelocity = 0.0

        if let firstAccel = accelHistory.first,
           let lastAccel = accelHistory.last,
           lastAccel.time - firstAccel.time > minDerivativeSpanSeconds {
            jerk = (
                lastAccel.val - firstAccel.val
            ) / (
                lastAccel.time - firstAccel.time
            )
        }

        if let firstHr = hrHistory.first,
           let lastHr = hrHistory.last,
           lastHr.time - firstHr.time > minDerivativeSpanSeconds {
            hrVelocity = (
                lastHr.val - firstHr.val
            ) / (
                lastHr.time - firstHr.time
            )
        }

        // 3. VZCrash ML prediction (4 features)
        var vzRisk = 0.0
        var classPred = 2

        if let vzArray = try? MLMultiArray(
            shape: [4],
            dataType: .double
        ) {
            vzArray[0] = NSNumber(value: effectiveAccel)
            vzArray[1] = NSNumber(value: gyroMag)
            vzArray[2] = NSNumber(value: speedKmh)
            vzArray[3] = NSNumber(value: abs(accelObd))

            if let prediction = try? vzcrashModel?.prediction(
                features: vzArray
            ) {
                let pCrash =
                    prediction.classProbability[0 as Int64] ?? 0.0
                let pNearMiss =
                    prediction.classProbability[1 as Int64] ?? 0.0
                let pNormal =
                    prediction.classProbability[2 as Int64] ?? 0.0

                vzRisk =
                    (1.0 * pCrash)
                    + (0.6 * pNearMiss)

                if pCrash > pNearMiss &&
                   pCrash > pNormal {
                    classPred = 0
                } else if pNearMiss > pCrash &&
                          pNearMiss > pNormal {
                    classPred = 1
                }
            }
        }

        // 4. POLIDriving ML prediction (14 features)
        var poliRisk = 0.0

        if let poliArray = try? MLMultiArray(
            shape: [14],
            dataType: .double
        ) {
            let hour = Double(
                Calendar.current.component(
                    .hour,
                    from: Date()
                )
            )

            let features: [Double] = [
                hour,
                speedKmh,
                rpm,
                accelObd,
                throttle,
                92.0,
                load,
                hr > 0 ? hr : 75.0,
                weatherCode,
                10.0,
                0.0,
                2.0,
                designSpeed,
                1.0
            ]

            for (index, value) in features.enumerated() {
                poliArray[index] = NSNumber(value: value)
            }

            if let prediction = try? fusionModel?.prediction(
                features: poliArray
            ) {
                let pMedium =
                    prediction.classProbability[1 as Int64] ?? 0.0
                let pHigh =
                    prediction.classProbability[2 as Int64] ?? 0.0
                let pVeryHigh =
                    prediction.classProbability[3 as Int64] ?? 0.0

                poliRisk =
                    (0.25 * pMedium)
                    + (0.65 * pHigh)
                    + (1.0 * pVeryHigh)
            }
        }

        // 5. Base ML fusion and predictive boosts
        //
        // poliRisk/vzRisk are already weighted sums of CoreML
        // classProbability values, which for these exported
        // RandomForestClassifier models are vote percentages on a
        // 0-100 scale (not 0-1 fractions) -- confirmed by probing
        // the bundled .mlpackage files directly. Do not re-multiply
        // by 100 here or the fused score saturates past the 99.9
        // clamp almost immediately on any nonzero medium/high class
        // probability.
        var riskScore = (
            (0.5 * poliRisk)
            + (0.5 * vzRisk)
        ) * weatherMult

        if jerk > 12.0 {
            riskScore += min(
                25.0,
                jerk * 1.5
            )
        }

        if hrVelocity > 3.0 {
            riskScore += min(
                15.0,
                hrVelocity * 2.5
            )
        }

        // 6. Road context (traffic density + pedestrians)
        //
        // trafficLevel/pedestrians previously only mattered inside the
        // "eyes off the road" branch below, so a crowded street or a
        // pedestrian standing right in frame had zero effect on risk as
        // long as the driver was technically looking forward. Give road
        // context its own always-on contribution here, then layer a much
        // larger compounding penalty on top below when the driver also
        // isn't watching it.
        let trafficBase: Double
        switch trafficLevel {
        case "empty":
            trafficBase = 0.0
        case "light":
            trafficBase = 2.0
        case "moderate":
            trafficBase = 6.0
        case "crowded":
            trafficBase = 12.0
        default:
            trafficBase = 0.0
        }
        riskScore += trafficBase

        if pedestriansDetected {
            riskScore += 8.0
        }

        // 7. Computer vision context penalties
        //
        // eyesClosedSec previously jumped straight from +0 to a flat +40
        // the instant it crossed 1.0s, then sat there until the 99.0 hard
        // override at 3.0s -- a step function, not a gradual increase.
        // Ramp it continuously instead (after a short grace period so an
        // ordinary blink, a sub-second EAR dip, never registers at all),
        // so the score keeps climbing the whole time the eyes stay closed
        // instead of jumping in one discrete step.
        let eyesClosedGraceSeconds = 0.5
        let eyesClosedRampCeilingSeconds = 3.0
        if eyesClosedSec > eyesClosedGraceSeconds {
            let rampProgress = min(
                1.0,
                (eyesClosedSec - eyesClosedGraceSeconds)
                    / (eyesClosedRampCeilingSeconds - eyesClosedGraceSeconds)
            )
            riskScore += rampProgress * 60.0
        }

        if !eyesOnRoad && speedKmh > 10.0 {
            riskScore += 15.0 + (trafficBase * 2.0)
            if pedestriansDetected {
                riskScore += 20.0
            }
        }

        // 8. False-positive suppressor
        let isSwerving = (
            gyroMag > 0.8 &&
            effectiveAccel < 24.0
        )

        if (inRoundabout || isSwerving) &&
           (classPred == 0 || classPred == 1) &&
           effectiveAccel < 24.0 {
            riskScore = min(
                riskScore,
                18.0 * weatherMult
            )
        }

        // 9. High-risk computer vision conditions
        if eyesClosedSec >= 3.0 {
            riskScore = 99.0
        } else if !eyesOnRoad &&
                  leadClosingFast &&
                  speedKmh >= 20.0 {
            riskScore += 25.0
        }

        // 10. Clamp final risk index
        return min(
            99.9,
            max(0.0, riskScore)
        )
    }
}
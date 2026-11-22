"""Turn the live inference stream into persisted sessions, tracks and alerts.

The recorder sits between ingestion and the database. It never raises into the
ingestion path: a storage failure must not stop a frame from being processed,
because losing telemetry is worse than losing a history row.
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any

from accident_risk.api.db import Database
from accident_risk.processing import collision_detector
from accident_risk.processing.geo import trip_distance_km
from accident_risk.schemas import PredictionResult, SensorFrame

LOGGER = logging.getLogger("accident_risk")

# One track point per second is enough for history and the live map; raw frames
# arrive far faster than that and belong in the research pipeline, not SQLite.
POINT_INTERVAL_SECONDS = 1.0
# A required source not observed for longer than this is a real interruption,
# not ordinary inter-frame jitter -- opens a session_gaps row for that source.
# Separate from PipelineSettings.max_age_seconds (that drives the non-edge
# InferenceEngine's own staleness flags); this one applies uniformly on the
# single path every frame -- edge or not -- actually flows through.
REQUIRED_SOURCE_GAP_SECONDS = 10.0
# Do not raise a second high-risk alert for the same driver inside this window.
ALERT_COOLDOWN_SECONDS = 60.0
# Do not flag a second automatic collision incident for the same stream
# inside this window -- the duplicate-suppression the roadmap asks for.
# Deliberately longer than ALERT_COOLDOWN_SECONDS: a real impact's kinematic
# signature can keep exceeding the threshold for several seconds (rollover,
# secondary impact, airbag deployment), and that must stay ONE incident.
COLLISION_DETECTION_COOLDOWN_SECONDS = 120.0
# An incident nobody confirms or dismisses within this window times out
# rather than sitting open forever.
UNCONFIRMED_TIMEOUT_SECONDS = 180.0


class SessionRecorder:
    """Persist track points and raise alerts for frames belonging to open sessions."""

    # Required sources tracked for interruption/availability recording. Not
    # every sensor RoadSense reads -- just the ones whose loss is meant to be
    # visible as a session_gaps row rather than silently absent values. Rear
    # (road-facing) camera is deliberately not tracked here: nothing in the
    # current wire protocol distinguishes "rear camera off" from "rear camera
    # sees no relevant object right now", so a presence signal for it would
    # be guessed, not measured (see docs/MODEL_LIMITATIONS.md).
    REQUIRED_SOURCES: tuple[str, ...] = ("vehicle", "gps", "front_camera", "wearable")

    def __init__(self, database: Database) -> None:
        self.db = database
        self._lock = RLock()
        self._sessions: dict[str, int] = {}
        self._last_point: dict[str, float] = {}
        self._last_alert: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._last_collision_detection: dict[str, float] = {}
        self._session_started: dict[str, float] = {}
        self._source_last_seen: dict[tuple[str, str], float] = {}
        self._open_gaps: dict[tuple[str, str], int] = {}

    def register(self, stream_id: str, session_id: int) -> None:
        """Begin recording frames for a stream."""
        with self._lock:
            self._sessions[stream_id] = session_id
            self._last_point.pop(stream_id, None)
            self._last_alert.pop(stream_id, None)
            self._last_seen.pop(stream_id, None)
            self._last_collision_detection.pop(stream_id, None)
            session = self.db.get_session(session_id)
            self._session_started[stream_id] = session["started_at"] if session else time.time()
            for source in self.REQUIRED_SOURCES:
                self._source_last_seen.pop((stream_id, source), None)
                self._open_gaps.pop((stream_id, source), None)

    def release(self, stream_id: str) -> None:
        """Stop recording frames for a stream."""
        with self._lock:
            for store in (
                self._sessions, self._last_point, self._last_alert,
                self._last_seen, self._last_collision_detection, self._session_started,
            ):
                store.pop(stream_id, None)
            for source in self.REQUIRED_SOURCES:
                self._source_last_seen.pop((stream_id, source), None)
                self._open_gaps.pop((stream_id, source), None)

    @staticmethod
    def _source_presence(frame: SensorFrame, camera: dict[str, Any]) -> dict[str, bool]:
        """Whether each required source has a real reading on this frame."""
        return {
            "vehicle": frame.vehicle_speed is not None,
            "gps": frame.gps_latitude is not None and frame.gps_longitude is not None,
            "front_camera": camera.get("camera_status") not in (None, "idle"),
            "wearable": frame.heart_rate is not None,
        }

    def _track_gaps(
        self, session_id: int, stream_id: str, now: float, frame: SensorFrame, camera: dict[str, Any]
    ) -> None:
        """Open a session_gaps row the first time a required source has gone
        unseen for longer than REQUIRED_SOURCE_GAP_SECONDS, and close it the
        next time that source is seen again. Runs on every observed frame,
        not just the throttled ~1Hz ones, so the gap boundary is timed by
        the frame that actually ended/restored it."""
        presence = self._source_presence(frame, camera)
        for source, present in presence.items():
            key = (stream_id, source)
            if present:
                self._source_last_seen[key] = now
                gap_id = self._open_gaps.pop(key, None)
                if gap_id is not None:
                    self.db.record_gap_end(gap_id, now)
                continue
            baseline = self._source_last_seen.get(key, self._session_started.get(stream_id, now))
            if key not in self._open_gaps and now - baseline > REQUIRED_SOURCE_GAP_SECONDS:
                self._open_gaps[key] = self.db.record_gap_start(session_id, source, baseline)

    def session_for(self, stream_id: str) -> int | None:
        """Return the open session recording this stream, if any."""
        with self._lock:
            return self._sessions.get(stream_id)

    def check_collision(self, frame: SensorFrame) -> dict[str, Any] | None:
        """Independent collision detection: runs on this frame's raw
        accelerometer data ONLY -- never reads a PredictionResult, risk
        score, or anything from the (separate, still-unvalidated)
        future-risk research. Call this unconditionally for every frame,
        regardless of whether a prediction exists yet for this stream.
        Never raises into ingestion, same principle as observe()."""
        try:
            return self._check_collision(frame)
        except Exception:  # noqa: BLE001 - detection must never break ingestion
            LOGGER.exception("collision_detector_failed", extra={"context": {"stream": frame.stream_id}})
            return None

    def _check_collision(self, frame: SensorFrame) -> dict[str, Any] | None:
        """Returns the new/escalated/matched incident dict, else None (no
        trigger, no active session, or suppressed as a duplicate).

        client_incident_id (when present, from the on-device detector) is
        the PRIMARY dedup key: a retried or offline-queued re-upload of the
        exact same detected event is recognized and returns the EXISTING
        alert's current state (which may have moved on from
        awaiting_confirmation by now), never creating a second one,
        regardless of how much time has passed OR whether this process has
        restarted since the original detection -- the dedup key lives in
        the database (Database.get_or_create_collision_alert), not an
        in-process cache. The time-window cooldown below remains as a
        fallback for frames with no id (legacy/non-iOS sources) and as
        defense in depth.
        """
        suspicion = collision_detector.check(frame)
        if suspicion is None or not suspicion.triggered:
            return None

        if frame.client_incident_id is not None:
            return self._get_or_create_known_incident(frame, suspicion)

        with self._lock:
            session_id = self._sessions.get(frame.stream_id)
            if session_id is None:
                return None
            now = frame.timestamp
            suppressed = (
                now - self._last_collision_detection.get(frame.stream_id, 0.0)
                < COLLISION_DETECTION_COOLDOWN_SECONDS
            )
            if suppressed:
                return None
            self._last_collision_detection[frame.stream_id] = now

        session = self.db.get_session(session_id)
        if session is None:
            return None

        detail = {
            "source": "automatic_detector",
            "model_version": suspicion.model_version,
            "onset_estimate": suspicion.onset_estimate,
            "detection_score": suspicion.score,
            "accel_magnitude_ms2": suspicion.accel_magnitude_ms2,
        }
        incident = self.db.create_alert(
            session["user_id"], "collision", session_id=session_id,
            latitude=frame.gps_latitude, longitude=frame.gps_longitude,
            status="suspected", detail=detail,
        )
        # Immediately promote suspected -> awaiting_confirmation: the driver
        # is now being asked. Kept as two steps (not created directly in the
        # later state) so the lifecycle transition itself is exercised and
        # audited, not merely asserted.
        return self.db.set_alert_status(
            incident["id"], "awaiting_confirmation", enforce_lifecycle=True
        )

    def _get_or_create_known_incident(
        self, frame: SensorFrame, suspicion: Any
    ) -> dict[str, Any] | None:
        """client_incident_id path: durable, atomic dedup via the database
        (see get_or_create_collision_alert's own docstring). No session
        lookup is required to RECOGNIZE a duplicate -- only to create a new
        incident -- so a retry that outlives its session (app relaunched,
        stream re-registered under the same still-pending client_incident_id)
        still correctly returns the existing incident rather than silently
        dropping it for lack of an active session."""
        with self._lock:
            session_id = self._sessions.get(frame.stream_id)
        session = self.db.get_session(session_id) if session_id is not None else None

        if session is None:
            # No active session to attach a NEW incident to (alerts.user_id
            # is NOT NULL) -- but a RETRY of an incident already recorded
            # under an earlier session (app relaunched and re-registered the
            # stream, or this arrives after the session ended / after a
            # server restart) must still resolve to that existing incident,
            # matching the dedup contract regardless of session lifetime.
            # Read-only: never attempts to create one without a user_id.
            return self.db.get_alert_by_client_incident_id(frame.client_incident_id)

        detail = {
            "source": "automatic_detector",
            "model_version": suspicion.model_version,
            "onset_estimate": suspicion.onset_estimate,
            "detection_score": suspicion.score,
            "accel_magnitude_ms2": suspicion.accel_magnitude_ms2,
            "client_incident_id": frame.client_incident_id,
        }
        incident, created = self.db.get_or_create_collision_alert(
            frame.client_incident_id, session["user_id"], session_id,
            latitude=frame.gps_latitude, longitude=frame.gps_longitude,
            status="suspected", detail=detail,
        )
        if not created:
            # Someone (this process earlier, or another concurrent request
            # right now) already created this exact incident -- return its
            # CURRENT state as-is. It may have moved on from "suspected"
            # already; never re-run the promotion below for an incident
            # that already exists.
            return incident
        return self.db.set_alert_status(
            incident["id"], "awaiting_confirmation", enforce_lifecycle=True
        )

    def observe(
        self,
        frame: SensorFrame,
        prediction: PredictionResult | None,
        camera: dict[str, Any] | None = None,
    ) -> None:
        """Record a frame against its session. Never raises into ingestion.

        camera carries front/rear-camera-derived fields (eyes_on_road, traffic
        level, pedestrian detections, ...) when the caller has them -- these live
        in the ingestion layer's per-stream UI state, not on SensorFrame itself.
        """
        try:
            self._observe(frame, prediction, camera or {})
        except Exception:  # noqa: BLE001 - storage must never break acquisition
            LOGGER.exception("recorder_failed", extra={"context": {"stream": frame.stream_id}})

    def _observe(
        self, frame: SensorFrame, prediction: PredictionResult | None, camera: dict[str, Any]
    ) -> None:
        """Append a throttled track point and raise alerts when thresholds are crossed."""
        with self._lock:
            session_id = self._sessions.get(frame.stream_id)
            if session_id is None:
                return
            now = frame.timestamp
            self._last_seen[frame.stream_id] = now
            due = now - self._last_point.get(frame.stream_id, 0.0) >= POINT_INTERVAL_SECONDS
            if due:
                self._last_point[frame.stream_id] = now
            alert_due = now - self._last_alert.get(frame.stream_id, 0.0) >= ALERT_COOLDOWN_SECONDS

        self._track_gaps(session_id, frame.stream_id, now, frame, camera)

        if due:
            # SensorFrame carries m/s; the interface shows km/h.
            speed_kmh = frame.vehicle_speed * 3.6 if frame.vehicle_speed is not None else None
            self.db.add_point(
                session_id,
                {
                    "ts": now,
                    "risk_index": prediction.overall_risk_index if prediction else None,
                    "risk_level": prediction.risk_level if prediction else None,
                    "speed_kmh": speed_kmh,
                    "rpm": frame.rpm,
                    "engine_load": frame.engine_load,
                    "throttle": frame.throttle_position,
                    "heart_rate": frame.heart_rate,
                    "latitude": frame.gps_latitude,
                    "longitude": frame.gps_longitude,
                },
            )
            self._record_full_sample(session_id, frame, prediction, camera, speed_kmh)

        if prediction is None or prediction.overall_risk_index is None or not alert_due:
            return
        thresholds = self.db.get_settings()
        if prediction.overall_risk_index < thresholds["risk_high_threshold"]:
            return
        session = self.db.get_session(session_id)
        if session is None:
            return
        with self._lock:
            self._last_alert[frame.stream_id] = now
        self.db.create_alert(
            session["user_id"],
            "high_risk",
            session_id=session_id,
            risk_index=prediction.overall_risk_index,
            latitude=frame.gps_latitude,
            longitude=frame.gps_longitude,
        )

    # Sources that plausibly originate from an assigned external device, by
    # devices.kind. The wire protocol carries no explicit device id field, so
    # this is the best available attribution at observation time -- not a
    # reconstruction from a later snapshot, but still best-effort, and marked
    # explicitly unresolved rather than silently guessed when ambiguous.
    _DEVICE_KIND_BY_SOURCE = {
        "obd": "adapter",
        "whoop_ble": "wearable",
        "wearable": "wearable",
    }

    def _resolve_device_id(self, user_id: int, source: str) -> str:
        """Device attribution at the moment this sample is recorded. Never
        reconstructed later from whatever happens to be assigned at export
        time -- this is evaluated synchronously as each frame arrives."""
        kind = self._DEVICE_KIND_BY_SOURCE.get(source)
        if kind is None:
            return "not_applicable:source_has_no_external_device"
        candidates = [d for d in self.db.kit_for_user(user_id) if d["kind"] == kind]
        if len(candidates) == 1:
            return candidates[0]["identifier"]
        if not candidates:
            return "unresolved:none_assigned"
        return "unresolved:ambiguous(" + ",".join(sorted(d["identifier"] for d in candidates)) + ")"

    def _record_full_sample(
        self,
        session_id: int,
        frame: SensorFrame,
        prediction: PredictionResult | None,
        camera: dict[str, Any],
        speed_kmh: float | None,
    ) -> None:
        """Append the full per-frame record alongside the throttled track point.
        Unconditional -- every session gets this, there is no consent gate.
        Missing readings stay None/absent -- never coerced to zero -- and
        sensor_timestamps preserves each sensor's original observation time,
        independent of when this frame itself arrived."""
        session = self.db.get_session(session_id)
        if session is None:
            return
        device_id = self._resolve_device_id(session["user_id"], frame.source)

        # The threshold actually in effect at THIS frame's own timestamp, not
        # whatever the settings happen to be right now -- see
        # Database.effective_thresholds_at and threshold_history.
        thresholds = self.db.effective_thresholds_at(frame.timestamp)
        risk_index = prediction.overall_risk_index if prediction else None
        warning_fired = risk_index is not None and risk_index >= thresholds["risk_high_threshold"]

        multimodal = camera.get("multimodal_risk") or {}
        breakdown = camera.get("risk_breakdown") or {}

        payload: dict[str, Any] = {
            "speed_kmh": speed_kmh,
            "rpm": frame.rpm,
            "throttle_position": frame.throttle_position,
            "engine_load": frame.engine_load,
            "accel_x": frame.accel_x, "accel_y": frame.accel_y, "accel_z": frame.accel_z,
            "gyro_x": frame.gyro_x, "gyro_y": frame.gyro_y, "gyro_z": frame.gyro_z,
            "gps_latitude": frame.gps_latitude,
            "gps_longitude": frame.gps_longitude,
            "gps_heading": frame.gps_heading,
            "is_roundabout": frame.is_roundabout,
            "heart_rate": frame.heart_rate,
            "rr_interval_ms": frame.rr_interval_ms,
            "sensor_timestamps": frame.sensor_timestamps,
            # Weather context -- see ContextManager.swift. weather_is_live is
            # always False right now: no real OpenWeather API key is
            # configured, so this is always the CLEAR/1.0 "unknown" default,
            # never fabricated as an observed reading.
            "weather_multiplier": camera.get("weather_multiplier"),
            "weather_condition_code": camera.get("weather_condition_code"),
            "weather_is_live": camera.get("weather_is_live"),
            # Discrete states and raw pose diagnostics behind the booleans
            # above -- see DriverAttentionState.swift (Phase 2),
            # ForwardCollisionHazard.swift (Phase 6), PedestrianHazard.swift
            # (Phase 7). debug_pitch/yaw/roll/ear are the raw head-pose/eye-
            # aspect-ratio readings the attention classification is built
            # from, included so the reasoning behind an attention state is
            # reconstructable later, not just its final label.
            "eyes_on_road": camera.get("eyes_on_road"),
            "attention_state": camera.get("attention_state"),
            "eyes_closed_sec": camera.get("eyes_closed_sec"),
            "debug_pitch": camera.get("debug_pitch"),
            "debug_yaw": camera.get("debug_yaw"),
            "debug_roll": camera.get("debug_roll"),
            "debug_ear": camera.get("debug_ear"),
            # Without this, a pitch/yaw/roll of 0 is ambiguous: "looking
            # dead ahead" and "no face found this frame" are otherwise
            # indistinguishable when reviewing a session's pose timeline
            # after the fact.
            "debug_face_detected": camera.get("debug_face_detected"),
            "camera_status": camera.get("camera_status"),
            "traffic_level": camera.get("traffic_level"),
            "pedestrians_detected": camera.get("pedestrians_detected"),
            "pedestrian_count": camera.get("pedestrian_count"),
            "pedestrian_hazard_state": camera.get("pedestrian_hazard_state"),
            "lead_closing_fast": camera.get("lead_closing_fast"),
            "forward_hazard_state": camera.get("forward_hazard_state"),
            "warning_fired": warning_fired,
            "front_processing_latency_ms": camera.get("front_processing_latency_ms"),
            "back_processing_latency_ms": camera.get("back_processing_latency_ms"),
            # Vehicle-only vs camera-inclusive/fused, kept clearly distinct --
            # see MultimodalRiskFusion.swift. The vehicle-only prediction
            # itself is `prediction.vehicle_risk_score` below (set from
            # breakdown.fiveSecondAheadVehicleHazardProb in main.py), not a
            # separate field here; multimodal_risk.five_second_ahead_risk is
            # a DIFFERENT, already-blended forecast (vehicle+attention+road),
            # so it must never be read as the vehicle-only number. The three
            # *_contribution fields are each dimension's weighted share of
            # the fused overall_risk_index, so that number can be decomposed
            # back into vehicle/attention/road-hazard evidence later.
            "vehicle_contribution": multimodal.get("vehicle_contribution"),
            "attention_contribution": multimodal.get("attention_contribution"),
            "road_hazard_contribution": multimodal.get("road_hazard_contribution"),
            "fusion_evidence_quality": multimodal.get("evidence_quality"),
            "fusion_dominant_factor": multimodal.get("dominant_factor"),
            # VZCrash's own collision-support classification -- supporting
            # evidence for the independent impact pathway only, never a
            # contributor to the predictive fused score (see
            # RiskFusionHeuristics.swift's own doc comment on this field).
            "collision_support_evidence": breakdown.get("collision_support_evidence"),
        }
        if prediction is not None:
            payload.update({
                "event_detection_score": prediction.event_detection_score,
                "vehicle_risk_score": prediction.vehicle_risk_score,
                "driver_state_risk": prediction.driver_state_risk,
                "overall_risk_index": prediction.overall_risk_index,
                "risk_level": prediction.risk_level,
                "data_quality": prediction.data_quality,
                "model_version": prediction.model_version,
                "quality_flags": prediction.quality_flags,
                "sensor_age_seconds": prediction.sensor_age_seconds,
                "inference_latency_ms": prediction.inference_latency_ms,
            })
        self.db.add_sample(session_id, frame.timestamp, device_id, frame.source, payload)

    def summarise(self, session_id: int) -> dict[str, Any]:
        """Compute the stored session summary from its track points (FR13.7)."""
        points = self.db.session_points(session_id)
        session = self.db.get_session(session_id)
        if session is None:
            return {}
        speeds = [p["speed_kmh"] for p in points if p["speed_kmh"] is not None]
        risks = [p["risk_index"] for p in points if p["risk_index"] is not None]
        hearts = [p["heart_rate"] for p in points if p["heart_rate"] is not None]
        started = session["started_at"]
        ended = session["ended_at"] or (points[-1]["ts"] if points else started)
        duration = max(ended - started, 0.0)
        levels: dict[str, int] = {}
        for point in points:
            if point["risk_level"]:
                levels[point["risk_level"]] = levels.get(point["risk_level"], 0) + 1
        # Points are ~1 s apart, so counting them approximates seconds.
        idling = sum(1 for p in points if p["speed_kmh"] is not None and p["speed_kmh"] < 1)
        # GPS-derived distance -- see processing/geo.py for exclusion rules
        # (invalid fixes, GPS jumps, stationary drift). None, not 0, when
        # fewer than two valid GPS fixes exist for this session.
        distance_km = trip_distance_km(points)
        gaps = self.db.session_gaps(session_id)
        tolerance = self.db.get_settings()["session_gap_tolerance_seconds"]
        longest_gap = max(
            ((g["ended_at"] or ended) - g["started_at"] for g in gaps), default=0.0
        )
        # Sensor availability from ACTUAL recorded interruption time per
        # required source (session_gaps), not from counting nonempty cells --
        # a source that was simply never wired up for a trip (e.g. no
        # wearable worn) looks identical to one that silently stopped
        # reporting if you only count empty cells; gap tracking distinguishes
        # them by measuring observed presence over time instead.
        gap_seconds_by_source: dict[str, float] = {}
        for gap in gaps:
            gap_seconds_by_source[gap["source"]] = gap_seconds_by_source.get(gap["source"], 0.0) + max(
                (gap["ended_at"] or ended) - gap["started_at"], 0.0
            )
        sensor_availability_by_source = {
            source: round(1.0 - min(gap_seconds_by_source.get(source, 0.0) / duration, 1.0), 4)
            for source in SessionRecorder.REQUIRED_SOURCES
        } if duration > 0 else {source: None for source in SessionRecorder.REQUIRED_SOURCES}
        defined_availability = [v for v in sensor_availability_by_source.values() if v is not None]
        overall_sensor_availability = (
            round(sum(defined_availability) / len(defined_availability), 4)
            if defined_availability else None
        )
        return {
            "duration_seconds": round(duration, 1),
            "distance_km": round(distance_km, 2) if distance_km is not None else None,
            "distance_source": "gps",
            "distance_calculation_method": "haversine_excluding_jumps_and_drift_v1",
            "max_speed_kmh": round(max(speeds), 1) if speeds else None,
            "mean_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "idling_seconds": idling,
            "mean_risk_index": round(sum(risks) / len(risks), 3) if risks else None,
            "max_risk_index": round(max(risks), 3) if risks else None,
            "risk_level_seconds": levels,
            "mean_heart_rate": round(sum(hearts) / len(hearts), 1) if hearts else None,
            "points": len(points),
            "gap_count": len(gaps),
            "longest_gap_seconds": round(longest_gap, 1),
            "complete": longest_gap <= tolerance,
            "alerts": len(self.db.list_alerts(session_id=session_id, limit=1000)),
            "sensor_availability_by_source": sensor_availability_by_source,
            "overall_sensor_availability": overall_sensor_availability,
        }

    def finish(self, session_id: int, stream_id: str, reason: str) -> dict[str, Any]:
        """Close a session, store its summary and flag it incomplete if gaps were long."""
        now = time.time()
        with self._lock:
            for source in self.REQUIRED_SOURCES:
                gap_id = self._open_gaps.pop((stream_id, source), None)
                if gap_id is not None:
                    self.db.record_gap_end(gap_id, now)
        summary = self.summarise(session_id)
        self.db.save_summary(session_id, summary)
        self.db.end_session(session_id, reason, complete=bool(summary.get("complete", True)))
        self.release(stream_id)
        return summary

    def stale_streams(self, timeout_seconds: float) -> list[str]:
        """Return streams that have not produced a frame within the timeout (FR13.6)."""
        now = time.time()
        with self._lock:
            return [
                stream
                for stream, seen in self._last_seen.items()
                if now - seen > timeout_seconds
            ]

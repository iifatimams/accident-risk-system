"""Chapter 6 evaluation architecture: annotations, alert verification,
collision tests, threshold history, gap tracking, GPS distance, and the
TP/TN/FP/FN + ROC-AUC/PR-AUC join itself."""

import time

import pytest

from accident_risk.api.db import Database
from accident_risk.api.recorder import SessionRecorder
from accident_risk.processing.geo import trip_distance_km
from accident_risk.schemas import PredictionResult, SensorFrame
from accident_risk.training.chapter6_evaluation import (
    build_window_aggregates,
    evaluate_pathway,
    join_windows_with_annotations,
)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.sqlite3")


@pytest.fixture
def driver_and_admin(db):
    admin = db.create_user("+971500000001", "Admin", "administrator")
    driver = db.create_user("+971500000002", "Driver", "driver")
    return driver, admin


def frame(stream_id, ts, **values):
    return SensorFrame(timestamp=ts, stream_id=stream_id, source="synthetic_test", **values)


def prediction(ts, stream_id, risk_index):
    return PredictionResult(
        timestamp=ts, stream_id=stream_id, overall_risk_index=risk_index,
        risk_level="HIGH" if risk_index and risk_index >= 0.7 else "LOW",
        data_quality=1.0, model_version="test", inference_latency_ms=1.0,
    )


# ------------------------------------------------------- annotations


def test_annotation_persists_and_latest_annotations_returns_current_revision(db, driver_and_admin):
    driver, admin = driver_and_admin
    session = db.start_session(driver["id"], "s1")

    first = db.add_annotation(
        session["id"], 100.0, 110.0,
        ground_truth_hazard="safe", ground_truth_distracted="attentive",
        annotator_id=admin["id"],
    )
    assert first["revision"] == 1

    second = db.add_annotation(
        session["id"], 100.0, 110.0,
        ground_truth_hazard="hazardous", ground_truth_distracted="distracted",
        annotator_id=admin["id"], annotation_notes="corrected after review",
    )
    assert second["revision"] == 2

    latest = db.latest_annotations(session["id"])
    assert len(latest) == 1
    assert latest[0]["ground_truth_hazard"] == "hazardous"
    assert latest[0]["revision"] == 2

    history = db.annotation_history(session["id"], 100.0, 110.0)
    assert [h["revision"] for h in history] == [1, 2]
    assert history[0]["ground_truth_hazard"] == "safe"


def test_annotation_rejects_unknown_state(db, driver_and_admin):
    driver, admin = driver_and_admin
    session = db.start_session(driver["id"], "s1")
    with pytest.raises(ValueError):
        db.add_annotation(
            session["id"], 0.0, 10.0,
            ground_truth_hazard="definitely_bad", ground_truth_distracted="attentive",
            annotator_id=admin["id"],
        )


def test_safe_and_unknown_are_distinct_states(db, driver_and_admin):
    """An explicitly-reviewed-and-safe window must not collapse into the
    same state as a never-reviewed one."""
    driver, admin = driver_and_admin
    session = db.start_session(driver["id"], "s1")
    db.add_annotation(
        session["id"], 0.0, 10.0,
        ground_truth_hazard="safe", ground_truth_distracted="attentive",
        annotator_id=admin["id"],
    )
    db.add_annotation(
        session["id"], 10.0, 20.0,
        ground_truth_hazard="unknown", ground_truth_distracted="unknown",
        annotator_id=admin["id"],
    )
    latest = {(a["window_start_ts"]): a["ground_truth_hazard"] for a in db.latest_annotations(session["id"])}
    assert latest[0.0] == "safe"
    assert latest[10.0] == "unknown"


# --------------------------------------------------------- window aggregation


def test_window_aggregation_groups_by_fixed_width_aligned_to_session_start(db, driver_and_admin):
    driver, _ = driver_and_admin
    session = db.start_session(driver["id"], "s1", started_at=1000.0)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])

    # Two samples in window 0 ([1000, 1010)), one in window 1 ([1010, 1020)).
    for ts, risk in ((1001.0, 0.2), (1005.0, 0.4), (1012.0, 0.9)):
        recorder.observe(frame("s1", ts, vehicle_speed=10), prediction(ts, "s1", risk))

    aggregates = build_window_aggregates(db, session["id"], window_seconds=10.0)
    assert len(aggregates) == 2
    assert aggregates[0].window_start_ts == 1000.0
    assert aggregates[0].fused_risk_mean == pytest.approx(0.3)
    assert aggregates[1].window_start_ts == 1010.0
    assert aggregates[1].fused_risk_mean == pytest.approx(0.9)


# --------------------------------------------------------------- TP/TN/FP/FN


def test_tp_tn_fp_fn_computed_from_independent_annotations_not_from_predictions(db, driver_and_admin):
    driver, admin = driver_and_admin
    session = db.start_session(driver["id"], "s1", started_at=0.0)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])

    # Window 0: predicted hazardous (0.9 >= 0.7 default high threshold), truly hazardous -> TP.
    recorder.observe(frame("s1", 1.0, vehicle_speed=10), prediction(1.0, "s1", 0.9))
    # Window 1: predicted safe (0.1), truly safe -> TN.
    recorder.observe(frame("s1", 11.0, vehicle_speed=10), prediction(11.0, "s1", 0.1))
    # Window 2: predicted hazardous (0.9), truly safe -> FP.
    recorder.observe(frame("s1", 21.0, vehicle_speed=10), prediction(21.0, "s1", 0.9))
    # Window 3: predicted safe (0.1), truly hazardous -> FN.
    recorder.observe(frame("s1", 31.0, vehicle_speed=10), prediction(31.0, "s1", 0.1))

    db.add_annotation(session["id"], 0.0, 10.0, ground_truth_hazard="hazardous", ground_truth_distracted="attentive", annotator_id=admin["id"])
    db.add_annotation(session["id"], 10.0, 20.0, ground_truth_hazard="safe", ground_truth_distracted="attentive", annotator_id=admin["id"])
    db.add_annotation(session["id"], 20.0, 30.0, ground_truth_hazard="safe", ground_truth_distracted="attentive", annotator_id=admin["id"])
    db.add_annotation(session["id"], 30.0, 40.0, ground_truth_hazard="hazardous", ground_truth_distracted="attentive", annotator_id=admin["id"])

    joined = join_windows_with_annotations(db, session["id"], window_seconds=10.0)
    result = evaluate_pathway({session["id"]: joined}, "camera_inclusive")

    assert result.confusion_counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert result.evaluated_windows == 4
    assert result.excluded_unannotated == 0
    assert result.metrics is not None
    assert result.metrics["roc_auc"] is not None  # both classes present -> defined


def test_vehicle_score_field_override_and_positive_class_metrics(db, driver_and_admin):
    """A batch where vehicle_risk_score doesn't hold the real vehicle-only
    value must be evaluable by pointing at the field that does, and the
    reported metrics must clearly separate macro-averaged from
    positive-class-only precision/recall/f1 (a paper usually reports the
    latter)."""
    driver, admin = driver_and_admin
    session = db.start_session(driver["id"], "s1", started_at=0.0)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])

    # vehicle_risk_score is a duplicate of overall_risk_index here (like the
    # historical import); the REAL vehicle-only score sits elsewhere.
    for ts, truth, real_vehicle_score in (
        (1.0, "hazardous", 0.9), (11.0, "safe", 0.1), (21.0, "safe", 0.9), (31.0, "hazardous", 0.1),
    ):
        pred = prediction(ts, "s1", 0.5)  # overall_risk_index/vehicle_risk_score both 0.5, uninformative
        recorder.observe(frame("s1", ts, vehicle_speed=10), pred)
        # Patch in the real vehicle-only score under a different field name,
        # the way the historical importer does.
        with db.connect() as connection:
            row = connection.execute(
                "SELECT id, payload_json FROM session_samples WHERE session_id = ? AND ts = ?",
                (session["id"], ts),
            ).fetchone()
            import json as _json
            payload = _json.loads(row["payload_json"])
            payload["five_second_ahead_vehicle_hazard_prob"] = real_vehicle_score
            connection.execute(
                "UPDATE session_samples SET payload_json = ? WHERE id = ?",
                (_json.dumps(payload), row["id"]),
            )
        db.add_annotation(session["id"], ts - 1.0, ts + 9.0, ground_truth_hazard=truth, ground_truth_distracted="attentive", annotator_id=admin["id"])

    joined = join_windows_with_annotations(
        db, session["id"], window_seconds=10.0,
        vehicle_score_field="five_second_ahead_vehicle_hazard_prob",
    )
    result = evaluate_pathway({session["id"]: joined}, "vehicle_only")

    # Using the real field: 0.9>=0.5 hazardous (truth hazardous -> TP),
    # 0.1<0.5 safe (truth safe -> TN), 0.9>=0.5 hazardous (truth safe -> FP),
    # 0.1<0.5 safe (truth hazardous -> FN).
    assert result.confusion_counts == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert result.metrics["positive_class"] == "hazardous"
    assert result.metrics["positive_class_precision"] == pytest.approx(0.5)
    assert result.metrics["positive_class_recall"] == pytest.approx(0.5)
    assert result.metrics["positive_class_f1"] == pytest.approx(0.5)
    # Macro-averaged precision/recall/f1 must not reappear under any name --
    # it's the convention that caused the mismatch, not just a dupe to hide.
    assert "precision" not in result.metrics
    assert "recall" not in result.metrics
    assert "f1" not in result.metrics
    assert "macro_precision" not in result.metrics


def test_unreviewed_and_missing_windows_are_excluded_and_counted(db, driver_and_admin):
    driver, admin = driver_and_admin
    session = db.start_session(driver["id"], "s1", started_at=0.0)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])
    recorder.observe(frame("s1", 1.0, vehicle_speed=10), prediction(1.0, "s1", 0.2))
    recorder.observe(frame("s1", 11.0, vehicle_speed=10), prediction(11.0, "s1", 0.2))

    # Window 0 is annotated as "unknown" (reviewed, undecidable);
    # window 1 is never annotated at all.
    db.add_annotation(session["id"], 0.0, 10.0, ground_truth_hazard="unknown", ground_truth_distracted="unknown", annotator_id=admin["id"])

    joined = join_windows_with_annotations(db, session["id"], window_seconds=10.0)
    result = evaluate_pathway({session["id"]: joined}, "camera_inclusive")

    assert result.evaluated_windows == 0
    assert result.excluded_unknown == 1
    assert result.excluded_unannotated == 1
    assert result.metrics is None


# ---------------------------------------------------------------- thresholds


def test_prediction_stays_associated_with_the_threshold_active_when_it_was_recorded(db, driver_and_admin):
    # threshold_history is stamped with real wall-clock time (time.time()),
    # so this test uses real-world timestamps too, unlike the small
    # synthetic ones most other tests here use for frames.
    driver, admin = driver_and_admin
    early_ts = time.time()
    session = db.start_session(driver["id"], "s1", started_at=early_ts)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])

    recorder.observe(frame("s1", early_ts, vehicle_speed=10), prediction(early_ts, "s1", 0.72))
    time.sleep(0.05)
    db.update_settings({"risk_high_threshold": 0.80}, changed_by=admin["id"])
    # Recorder throttles session_samples to ~1Hz (POINT_INTERVAL_SECONDS);
    # the second frame must be spaced out enough to actually get recorded.
    time.sleep(1.1)
    late_ts = time.time()
    recorder.observe(frame("s1", late_ts, vehicle_speed=10), prediction(late_ts, "s1", 0.72))

    samples = db.session_samples(session["id"])
    early_thresholds = db.effective_thresholds_at(samples[0]["ts"])
    late_thresholds = db.effective_thresholds_at(samples[1]["ts"])
    assert early_thresholds["risk_high_threshold"] == 0.70
    assert late_thresholds["risk_high_threshold"] == 0.80
    # The early sample's own recorded warning_fired reflects the threshold
    # that was active when IT was recorded (0.72 >= 0.70), not the later one.
    assert samples[0]["warning_fired"] is True


# -------------------------------------------------------------- gap tracking


def test_required_source_gap_is_opened_and_closed(db, driver_and_admin):
    driver, _ = driver_and_admin
    session = db.start_session(driver["id"], "s1", started_at=0.0)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])

    recorder.observe(frame("s1", 1.0, vehicle_speed=10), prediction(1.0, "s1", 0.1))
    # GPS missing for well over the gap threshold.
    recorder.observe(frame("s1", 20.0, vehicle_speed=10), prediction(20.0, "s1", 0.1))
    # GPS returns.
    recorder.observe(frame("s1", 25.0, vehicle_speed=10, gps_latitude=25.0, gps_longitude=55.0), prediction(25.0, "s1", 0.1))

    gaps = db.session_gaps(session["id"])
    gps_gaps = [g for g in gaps if g["source"] == "gps"]
    assert len(gps_gaps) == 1
    # GPS was never present before this, so the gap is backdated to the
    # session's own start rather than to whenever it merely crossed the
    # staleness threshold.
    assert gps_gaps[0]["started_at"] == 0.0
    assert gps_gaps[0]["ended_at"] == 25.0


def test_still_open_gap_is_closed_when_session_finishes(db, driver_and_admin):
    driver, _ = driver_and_admin
    session = db.start_session(driver["id"], "s1", started_at=0.0)
    recorder = SessionRecorder(db)
    recorder.register("s1", session["id"])
    recorder.observe(frame("s1", 1.0, vehicle_speed=10), prediction(1.0, "s1", 0.1))
    recorder.observe(frame("s1", 20.0, vehicle_speed=10), prediction(20.0, "s1", 0.1))

    recorder.finish(session["id"], "s1", "driver_stopped")

    gaps = db.session_gaps(session["id"])
    gps_gaps = [g for g in gaps if g["source"] == "gps"]
    assert len(gps_gaps) == 1
    assert gps_gaps[0]["ended_at"] is not None


# ------------------------------------------------------------- GPS distance


def test_trip_distance_excludes_invalid_jumps_and_drift():
    points = [
        {"ts": 0, "latitude": 25.0, "longitude": 55.0},
        {"ts": 10, "latitude": 25.001, "longitude": 55.001},  # ~150m, real movement
        {"ts": 11, "latitude": 40.0, "longitude": 10.0},  # impossible jump in 1s -- excluded
        {"ts": 21, "latitude": 25.0010001, "longitude": 55.0010001},  # sub-5m drift -- excluded
        {"ts": 22, "latitude": None, "longitude": 55.002},  # invalid -- excluded
    ]
    distance = trip_distance_km(points)
    assert distance is not None
    assert 0.1 < distance < 0.2  # only the one real ~150m segment counted


def test_trip_distance_is_none_not_zero_with_fewer_than_two_fixes():
    assert trip_distance_km([]) is None
    assert trip_distance_km([{"ts": 0, "latitude": 25.0, "longitude": 55.0}]) is None


# ---------------------------------------------------------- alert verification


def test_alert_verification_round_trip(db, driver_and_admin):
    driver, admin = driver_and_admin
    alert = db.create_alert(driver["id"], "high_risk", risk_index=0.9)
    assert db.get_alert_verification(alert["id"]) is None

    db.verify_alert(alert["id"], "correct", verified_by=admin["id"], notes="matched footage")
    row = db.get_alert_verification(alert["id"])
    assert row["verification_result"] == "correct"

    db.verify_alert(alert["id"], "false", verified_by=admin["id"], notes="revised")
    row = db.get_alert_verification(alert["id"])
    assert row["verification_result"] == "false"


def test_alert_verification_rejects_unknown_result(db, driver_and_admin):
    driver, admin = driver_and_admin
    alert = db.create_alert(driver["id"], "high_risk", risk_index=0.9)
    with pytest.raises(ValueError):
        db.verify_alert(alert["id"], "definitely_wrong", verified_by=admin["id"])


# ------------------------------------------------------------- collision tests


def test_collision_test_pass_fail_is_derived_not_stored_redundantly(db, driver_and_admin):
    _, admin = driver_and_admin
    passed = db.record_collision_test(
        "staged_rear_impact", expected_collision_detection=True,
        actual_collision_detection=True, detection_latency_ms=350.0, recorded_by=admin["id"],
    )
    failed = db.record_collision_test(
        "staged_pothole", expected_collision_detection=False,
        actual_collision_detection=True, recorded_by=admin["id"],
    )
    assert passed["test_result"] == "pass"
    assert failed["test_result"] == "fail"
    assert [t["test_result"] for t in db.list_collision_tests()] == ["pass", "fail"]


# --------------------------------------------------------- orphan recovery


def test_recover_orphaned_sessions_closes_sessions_left_open_by_a_dead_process(db, driver_and_admin):
    driver, _ = driver_and_admin
    session = db.start_session(driver["id"], "s1")
    assert db.get_session(session["id"])["ended_at"] is None

    recovered = db.recover_orphaned_sessions()

    assert session["id"] in recovered
    after = db.get_session(session["id"])
    assert after["ended_at"] is not None
    assert after["end_reason"] == "orphaned_on_restart"
    assert after["complete"] == 0

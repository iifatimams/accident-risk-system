#!/usr/bin/env python3
"""Check synthetic RoadSense fixtures; only standard-library dependencies required."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
assert manifest['synthetic_test_only'] and manifest['not_for_real_alert_delivery']
assert len(manifest['scenario_list'])==10
n=0
for scenario in manifest['scenario_list']:
    frames=[json.loads(l) for l in (ROOT/scenario['frames']).read_text().splitlines()]
    labels=list(csv.DictReader((ROOT/scenario['labels']).open()))
    assert len(frames)==len(labels)==401
    assert all(f['stream_id']==scenario['stream_id'] and f['source']=='synthetic_replay' for f in frames)
    assert all(not any(key in f for key in ('heart_rate','rr_interval_ms','hrv_rmssd','spo2')) for f in frames)
    assert all(round(frames[k+1]['timestamp']-frames[k]['timestamp'],3)==0.1 for k in range(400))
    assert all(not any(key in f for key in ('lead_distance_m_truth','contact_now','event_in_next_5s')) for f in frames)
    for f,l in zip(frames,labels):
        assert f['timestamp']==float(l['timestamp'])
        assert all(name in f and t<=f['timestamp'] for name,t in f['sensor_timestamps'].items())
        assert f.get('gps_speed',0)>=0 and f.get('vehicle_speed',0)>=0
        assert all(math.isfinite(v) for k,v in f.items() if isinstance(v,(float,int)) and not isinstance(v,bool))
    impacts=[l for l in labels if l['contact_now']=='True']
    positives=[l for l in labels if l['event_in_next_5s']=='True']
    if scenario['outcome']=='collision':
        assert len(impacts)==1 and len(positives)==50
        assert float(impacts[0]['elapsed_s'])==28.0
        assert min(float(l['lead_distance_m_truth'] or l['pedestrian_distance_m_truth']) for l in labels if l['elapsed_s']=='28.0')==0
    else:assert not impacts and not positives
    n+=len(frames)
print(f'PASS: {len(manifest["scenario_list"])} scenarios, {n} canonical JSONL frames and matching labels.')
try:
    from accident_risk.schemas import SensorFrame
except ImportError:
    print('NOTE: RoadSense Python package not installed; Pydantic schema check skipped.')
else:
    for s in manifest['scenario_list']:
        for line in (ROOT/s['frames']).read_text().splitlines(): SensorFrame.model_validate_json(line)
    print('PASS: all frames validate against installed RoadSense SensorFrame Pydantic schema.')

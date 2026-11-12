#!/usr/bin/env python3
"""RoadSense synthetic software-test telemetry; NO real accident or physiological data."""
import csv, json, math, random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data'
OUT.mkdir(exist_ok=True)
SEED=20261113
DT=0.1
N=401  # 40 s at 10 Hz
BASE=1794528000  # arbitrary synthetic epoch; NOT a historical test
G=9.80665
# Approximate bounding-area anchors only. These are NOT map-matched routes.
ANCHORS=[(25.346,55.422),(25.279,55.364),(25.213,55.402),(25.121,55.203)]
SCENARIOS = [
 ('normal_cruise','normal','steady driving; no collision',0),
 ('gentle_brake','normal','benign slowdown without obstacle',0),
 ('normal_roundabout','normal','benign left turn; yaw not collision',0),
 ('stopped_pedestrian','normal','pedestrian visible while ego stopped',0),
 ('lead_vehicle_safe','normal','approach slower vehicle and brake safely',0),
 ('lead_vehicle_nearmiss','near_miss','closing on stopping lead vehicle; strong avoidance brake',0),
 ('lead_vehicle_collision','collision','simulated rear-end collision at 28.0s',28.0),
 ('pedestrian_crossing_nearmiss','near_miss','pedestrian enters corridor; driver brakes to stop',0),
 ('pedestrian_crossing_collision','collision','simulated pedestrian conflict at 28.0s',28.0),
 ('gps_obd_dropout','normal','sensor interruptions and reacquisition',0),
]

FRAME_FIELDS=['timestamp','stream_id','source','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','gps_latitude','gps_longitude','gps_speed','gps_heading','vehicle_speed','rpm','throttle_position','engine_load','steering_angle','heart_rate','spo2','hrv_rmssd','rr_interval_ms','physiology_signal_quality','is_roundabout','sensor_timestamps']
LABEL_FIELDS=['timestamp','stream_id','elapsed_s','scenario','outcome','collision_onset_s','event_in_next_5s','contact_now','lead_distance_m_truth','pedestrian_distance_m_truth','pedestrian_in_path_truth','hazard_present_truth','road_anchor_is_approximate','notes']


def acc_profile(name,t):
    if name == 'normal_cruise': return 13.,0.
    if name == 'gentle_brake': return (12.,0.) if t<13 else (max(0,12.-1.4*(t-13)), -1.4 if 13<=t<21.5714 else 0.)
    if name == 'normal_roundabout': return 8.,0.
    if name == 'stopped_pedestrian': return 0.,0.
    if name == 'gps_obd_dropout': return 11.,0.
    if name == 'lead_vehicle_safe':
        return (14.,0.) if t<18 else (max(6.,14.-1.4*(t-18)),-1.4 if t<23.7143 else 0.)
    if name == 'lead_vehicle_nearmiss':
        return (16.,0.) if t<21 else (max(0.,16.-4.2*(t-21)),-4.2 if t<24.8095 else 0.)
    if name == 'lead_vehicle_collision':
        if t<24:return 17.,0.
        if t<28:return 17.-1*(t-24),-1.
        return 0.,0.
    if name == 'pedestrian_crossing_nearmiss':
        if t<20:return 11.,0.
        return max(0.,11.-3.5*(t-20)),-3.5 if t<23.1429 else 0.
    if name == 'pedestrian_crossing_collision':
        if t<23:return 12.,0.
        if t<28:return 12.-0.45*(t-23),-0.45
        return 0.,0.
    raise ValueError(name)


def generate(name, outcome, description, onset, idx):
    rng=random.Random(SEED+idx)
    stream=f'sim_{name}_{SEED}'
    anchor=ANCHORS[idx%len(ANCHORS)]
    lat,lon=anchor
    # Displacement is a small local illustrative geometry, not actual road topology.
    east=0.;north=0.;heading=math.radians(58)
    prev_speed=acc_profile(name,0)[0]
    lead_distance=({'lead_vehicle_safe':220.0,'lead_vehicle_nearmiss':310.0,'lead_vehicle_collision':343.0}.get(name) if 'lead_vehicle' in name else None)
    ped_distance=({'pedestrian_crossing_nearmiss':260.0,'pedestrian_crossing_collision':318.0}.get(name) if 'pedestrian_crossing' in name else (13.0 if name=='stopped_pedestrian' else None))
    rows=[];labels=[]
    for k in range(N):
        t=round(k*DT,2)
        speed,acc=acc_profile(name,t)
        if name=='normal_roundabout':
            yaw=0.12 if 8<t<24 else 0.0
        else:yaw=0.0
        heading=(heading+yaw*DT)%(2*math.pi)
        north+=speed*math.cos(heading)*DT
        east+=speed*math.sin(heading)*DT
        # Lead trajectory assumptions explicitly synthetic. Vehicle ahead stationary in collision example.
        if lead_distance is not None:
            if name=='lead_vehicle_safe': lead_speed=7.0
            elif name=='lead_vehicle_nearmiss': lead_speed=4.0 if t<19 else 0.
            else: lead_speed=8.0 if t<15 else 0.
            lead_distance=max(0.,lead_distance+(lead_speed-speed)*DT)
        if ped_distance is not None and 'crossing' in name:
            ped_distance=max(0.,ped_distance-speed*DT)
        # Informational scenario ground-truth only, NOT fed into ML model.
        ped_in_path=('crossing' in name and 22<=t<=30)
        hazard=(('lead_vehicle' in name and lead_distance is not None and lead_distance<45) or ped_in_path)
        impact=onset!=0 and abs(t-onset)<DT/2
        # Deceleration encoded x; small accelerometer noise in phone-fixed synthetic axes.
        ax=acc+rng.gauss(0,0.055)
        ay=(speed*yaw)+rng.gauss(0,0.045)
        az=rng.gauss(0,0.05)  # user acceleration: excludes gravity
        if impact:
            # Schematic one-tick acceleration pulse; NOT crash physics or empirical amplitude.
            ax+=29.0
        gyro_z=yaw+rng.gauss(0,0.006)
        timestamp=BASE+idx*120+t
        # Different sensor cadences; missing fields use null, never fabricated 0.
        frame={'timestamp':timestamp,'stream_id':stream,'source':'synthetic_replay',
               'accel_x':round(ax,5),'accel_y':round(ay,5),'accel_z':round(az,5),
               'gyro_x':round(rng.gauss(0,0.003),5),'gyro_y':round(rng.gauss(0,0.003),5),'gyro_z':round(gyro_z,5)}
        times={q:timestamp for q in ('accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z')}
        gps_ok=not(name=='gps_obd_dropout' and 15<=t<25)
        obd_ok=not(name=='gps_obd_dropout' and 10<=t<21)
        if k%10==0 and gps_ok:
            frame.update(gps_latitude=round(lat+north/111200.,7),
                         gps_longitude=round(lon+east/(111200*math.cos(math.radians(lat))),7),
                         gps_speed=round(max(0,speed+rng.gauss(0,.12)),5),
                         gps_heading=round(heading,7))
            times.update({q:timestamp for q in ('gps_latitude','gps_longitude','gps_speed','gps_heading')})
        if k%5==0 and obd_ok:
            frame.update(vehicle_speed=round(speed,5),rpm=round(800+speed*85,1),
                         throttle_position=round(14. if acc<0 else 22.,1),engine_load=round(23. if acc<0 else 34.,1))
            times.update({q:timestamp for q in ('vehicle_speed','rpm','throttle_position','engine_load')})
        if name=='normal_roundabout' and k%10==0:
            frame['is_roundabout']=8<=t<=24
            times['is_roundabout']=timestamp
        frame['sensor_timestamps']=times
        rows.append(frame)
        labels.append({'timestamp':timestamp,'stream_id':stream,'elapsed_s':t,'scenario':name,'outcome':outcome,
                       'collision_onset_s':onset if onset else None,
                       'event_in_next_5s':bool(onset and t<onset and 0<onset-t<=5),
                       'contact_now':impact,
                       'lead_distance_m_truth':round(lead_distance,3) if lead_distance is not None else None,
                       'pedestrian_distance_m_truth':round(ped_distance,3) if ped_distance is not None else None,
                       'pedestrian_in_path_truth':ped_in_path,
                       'hazard_present_truth':bool(hazard),
                       'road_anchor_is_approximate':True,'notes':'SIMULATED_GROUND_TRUTH_NOT_SENSOR_MEASUREMENT'})
    # Python sensor schema accepts missing fields and separate sensor timestamps.
    with (OUT/f'{name}.jsonl').open('w') as fh:
        for row in rows: fh.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    with (OUT/f'{name}_labels.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=LABEL_FIELDS);w.writeheader();w.writerows(labels)
    return {'scenario':name,'outcome':outcome,'description':description,'stream_id':stream,'samples':len(rows),
            'duration_s':40.0,'frequency_hz':10,'onset_s':onset or None,'imu_unit':'m/s^2 excluding gravity',
            'gps_anchor_approx':[lat,lon], 'frames':f'data/{name}.jsonl','labels':f'data/{name}_labels.csv'}


def main():
    manifest={'title':'RoadSense synthetic integration fixtures v1','seed':SEED,
              'synthetic_test_only':True,'not_real_vehicle_data':True,'not_real_uAE_routes':True,
              'not_usable_to_claim_model_accuracy':True,'not_for_training_without_independent_validation':True,
              'not_for_real_alert_delivery':True,
              'generated_utc':datetime.now(timezone.utc).isoformat(),
              'format':'JSONL frames conform to src/accident_risk/schemas.py SensorFrame (optional fields omitted); separate CSV ground truth',
              'cadences_hz':{'imu':10,'gps':1,'obd_like':2,'whoop':None},
              'mounting':'idealized phone-fixed axes; no camera/photorealistic frames; no physical iPhone calibration',
              'scenario_list':[]}
    for i,s in enumerate(SCENARIOS): manifest['scenario_list'].append(generate(*s,i))
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"generated {len(SCENARIOS)} scenarios and {N*len(SCENARIOS)} frames")

if __name__=='__main__': main()

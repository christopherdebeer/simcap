"""Secondary diagnostics. Does not retune the primary exported models."""
import argparse,json
from pathlib import Path
import numpy as np
from experiment import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);args=ap.parse_args()
    out=Path(__file__).resolve().parent/'results'
    d=json.loads((args.data_dir/SESSION).read_text())
    mag,good,y,blocks,classes,ri,ref,splits,rawtest=prepare(d)
    a=np.array([[s[k]for k in ('ax','ay','az')]for s in d['samples']],dtype=float)
    # Gravity-only inclination proxy; acceleration can contaminate this value.
    # Never consume stored AHRS angles, calibration matrices, or learned baseline.
    pitch=np.degrees(np.arctan2(-a[:,0],np.sqrt(a[:,1]**2+a[:,2]**2)))
    labelled=(y!='?????')&good&~np.isin(np.arange(len(y)),ri)
    orientation=[]
    for direction in [1,-1]:
        tr=np.flatnonzero(labelled&(pitch*direction>=20))
        te=np.flatnonzero(labelled&(pitch*direction<=-20))
        cls=sorted(set(y[tr]));assert not set(tr)&set(te)
        row=dict(direction=direction,train_count=len(tr),test_count=len(te),
                 train_class_counts={c:int(np.sum(y[tr]==c))for c in classes},
                 test_class_counts={c:int(np.sum(y[te]==c))for c in classes},models={})
        for feature in ['raw','relative','soft_direction']:
            x=transform(mag,feature,ref);model=fit_model(x[tr],y[tr],'centroid',cls)
            pp,_,_=predict(model,x[te]);row['models'][feature]=metrics(y[te],pp)
        orientation.append(row)
    timestamp=np.array([s['t']for s in d['samples']])
    # This diagnostic cutoff is intentionally session-specific, not a deployable quality gate.
    # Normal recording times here span ~0..185 seconds; 1e6 and above are inconsistent.
    suspicious=(timestamp>=1e6)|(timestamp<0)
    quality={k:{c:{'n':int(np.sum(y[ix]==c)),
                       'anomalous_timestamp':int(np.sum((y[ix]==c)&suspicious[ix]))}
                 for c in classes}for k,ix in splits.items()}
    r=dict(note='Secondary same-session orientation diagnostic; no new untouched final test. Fixed centroid, no tuning.',
           pitch_definition='degrees(atan2(-ax,sqrt(ay^2+az^2))); not ground-truth orientation during motion',
           orientation=orientation,quality=quality,
           warning='Timestamp anomalies suggest integrity problems but do not prove corrupted magnetometer axes. No silent correction applied.')
    write_json(out/'diagnostics.json',r)
    for item in orientation:
        print('orientation',item['direction'],'n',item['train_count'],item['test_count'],
              {k:round(v['exact_chord_accuracy'],4)for k,v in item['models'].items()})

if __name__=='__main__':main()

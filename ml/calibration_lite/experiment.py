"""Leakage-controlled, compact magnetic chord classification experiment.

Run from the repository root:
 python ml/calibration_lite/experiment.py --data-dir ../captured-data/GAMBIT
No TensorFlow, stored calibration, derived orientation, or synthetic training data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
SESSION = '2025-12-31T14_06_18.270Z.json'
FEATURES = ['raw', 'relative', 'direction', 'soft_direction', 'relative_magnitude']
ALGORITHMS = ['centroid', 'gaussian', 'lda']
SCALE = 100 / 1024  # native MMC5603NJ sensor axes, NOT stored aligned mx_ut fields
TRIM = 5
REFERENCE_SAMPLES = 30

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def labels_of(document):
    result = []
    for i, label in enumerate(document.get('labels', [])):
        fingers = label.get('labels', label).get('fingers', {})
        if not all(fingers.get(f) in ('extended', 'flexed') for f in FINGERS):
            continue
        start = label.get('start_sample', label.get('startIndex'))
        end = label.get('end_sample', label.get('endIndex'))
        code = ''.join('e' if fingers[f] == 'extended' else 'f' for f in FINGERS)
        result.append(dict(id=i, start=start, end=end, code=code))
    return result

def audit(data_dir):
    rows = []
    for p in sorted(Path(data_dir).glob('*.json')):
        content = p.read_bytes()
        try:
            d = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            rows.append(dict(file=p.name, bytes=len(content), status='not_json_lfs_pointer',
                             sha256=hashlib.sha256(content).hexdigest()))
            continue
        if 'samples' not in d:
            continue
        samples = d['samples']
        mag = np.array([[s.get(k, np.nan) for k in ('mx','my','mz')] for s in samples])
        times = np.array([s.get('t', np.nan) for s in samples])
        blocks = labels_of(d)
        rows.append(dict(file=p.name, bytes=len(content), status='session',
                         sha256=hashlib.sha256(content).hexdigest(), samples=len(samples),
                         binary_labeled_blocks=len(blocks), classes=sorted({b['code'] for b in blocks}),
                         rail_samples=int(np.any(np.abs(mag) >= 32760, axis=1).sum()),
                         nonincreasing_t=int((np.diff(times) <= 0).sum()),
                         jumps_over_one_second=int((np.diff(times) > 1000).sum()),
                         metadata_sample_rate=d.get('metadata',{}).get('sample_rate')))
    return rows

def reference(mag):
    center = np.median(mag, axis=0)
    noise = float(np.median(np.linalg.norm(mag-center, axis=1)))
    if not np.isfinite(noise) or noise < 1e-6:
        raise ValueError('Reference has no measurable spread or invalid samples')
    return {'baseline': center.tolist(), 'noise': noise, 'tau': 3 * noise}

def transform(mag, name, ref):
    if name == 'raw':
        return mag.copy()
    delta = mag - np.asarray(ref['baseline'])
    norm = np.linalg.norm(delta, axis=1)
    if name == 'relative':
        return delta
    if name == 'direction':
        return delta / np.maximum(norm[:,None], 1e-9)
    if name == 'soft_direction':
        return delta / np.sqrt(norm[:,None]**2 + ref['tau']**2)
    if name == 'relative_magnitude':
        return norm[:,None] / ref['tau']
    raise ValueError(name)

def fit_model(x, y, algorithm, classes):
    means = np.array([np.mean(x[y==c],axis=0) for c in classes])
    global_var = np.maximum(np.var(x,axis=0), 1e-12)
    variances = np.array([np.var(x[y==c],axis=0) for c in classes])
    # Shared dimensional scale learned from training only.
    if algorithm == 'centroid':
        v = np.broadcast_to(global_var, means.shape)
    else:
        v = np.maximum(0.5*variances + 0.5*global_var, 1e-12)
    a = -0.5 / v
    b = means / v
    c = -0.5*np.sum(means**2/v,axis=1)
    if algorithm == 'gaussian':
        c -= 0.5*np.sum(np.log(v),axis=1)
    if algorithm == 'lda':
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto',
                                        priors=np.ones(len(classes))/len(classes))
        lda.fit(x, y)
        assert list(lda.classes_) == list(classes)
        a = np.zeros_like(means)
        b = lda.coef_
        c = lda.intercept_
    # OOD gate uses training-only class envelopes, never test points.
    gate_var = np.maximum(variances, global_var * 0.05)
    gate_distance = np.array([np.sum((row-means[list(classes).index(label)])**2 /
                                    gate_var[list(classes).index(label)]) for row,label in zip(x,y)])
    radius = float(np.quantile(gate_distance, .99))
    return dict(algorithm=algorithm, classes=list(classes), quadratic=a.tolist(),
                linear=b.tolist(), intercept=c.tolist(), means=means.tolist(),
                gate_variances=gate_var.tolist(), gate_radius=radius)

def predict(model, x):
    scores = x*x @ np.asarray(model['quadratic']).T + x @ np.asarray(model['linear']).T + model['intercept']
    order = np.argsort(scores,axis=1)
    best = order[:,-1]
    margin = scores[np.arange(len(x)),best] - scores[np.arange(len(x)),order[:,-2]]
    means = np.asarray(model['means'])[best]
    variances = np.asarray(model['gate_variances'])[best]
    distances = np.sum((x-means)**2/variances,axis=1)
    return np.asarray(model['classes'])[best], margin, distances

def metrics(y, pred, accepted=None):
    if accepted is None:
        accepted = np.ones(len(y), dtype=bool)
    correct = y == pred
    cls = sorted(set(y))
    per_class = {c: {'n':int(np.sum(y==c)), 'accuracy':float(np.mean(correct[y==c])),
                     'coverage':float(np.mean(accepted[y==c])),
                     'correct_accepted_fraction':float(np.mean((correct & accepted)[y==c]))} for c in cls}
    bits = lambda arr: np.array([[v=='f' for v in s] for s in arr])
    return dict(n=len(y), exact_chord_accuracy=float(np.mean(correct)),
                macro_recall=float(np.mean([p['accuracy'] for p in per_class.values()])),
                per_finger_accuracy=float(np.mean(bits(y)==bits(pred))),
                coverage=float(np.mean(accepted)), accepted_count=int(accepted.sum()),
                accepted_accuracy=float(np.mean(correct[accepted])) if accepted.any() else None,
                correct_accepted_fraction=float(np.mean(correct & accepted)),
                per_class=per_class)

def choose_rejection(y, pred, margins, distances, radius):
    # Development-only threshold, fixed before evaluating locked blocks.
    options = []
    for threshold in np.unique(np.r_[0, np.quantile(margins, np.linspace(0,.99,100))]):
        accepted = (margins >= threshold) & (distances <= radius)
        if accepted.sum() >= 20 and np.mean(pred[accepted] == y[accepted]) >= .95:
            options.append((int(accepted.sum()), float(threshold)))
    if not options:
        return 1e30  # No demonstrated operating point; reject everything.
    return max(options, key=lambda v:(v[0],-v[1]))[1]

def prepare(data):
    samples = data['samples']
    mag = np.array([[s[k] for k in ('mx','my','mz')]for s in samples], dtype=float)*SCALE
    good = np.all(np.isfinite(mag),axis=1) & np.all(np.abs(mag/SCALE)<32760,axis=1)
    blocks = labels_of(data)
    classes = sorted({b['code'] for b in blocks})
    first_open = next(b for b in blocks if b['code']=='eeeee')
    ref_idx = np.arange(first_open['start']+TRIM,first_open['start']+TRIM+REFERENCE_SAMPLES)
    if not good[ref_idx].all():
        raise ValueError('Invalid reference samples')
    ref = reference(mag[ref_idx])
    splits = {k:[] for k in ('train','development','test')}
    y = np.full(len(mag),'?????',dtype='<U5')
    block_id = np.full(len(mag),-1)
    seen = set()
    raw_test_idx=[]
    for block in blocks:
        # End is excluded conservatively: schemas mix inclusive/exclusive ends.
        idx = np.arange(block['start']+TRIM,block['end']-TRIM)
        y[idx]=block['code']; block_id[idx]=block['id']
        idx=idx[~np.isin(idx,ref_idx)]
        if block['code'] not in seen:
            seen.add(block['code'])
            cut=int(len(idx)*.6)
            splits['train'].extend(idx[:cut][good[idx[:cut]]].tolist())
            tail=idx[cut+5:]
            splits['development'].extend(tail[good[tail]].tolist())
        else:
            raw_test_idx.extend(idx.tolist())
            splits['test'].extend(idx[good[idx]].tolist())
    splits={k:np.array(v,dtype=int)for k,v in splits.items()}
    for a in splits:
        assert not set(splits[a]) & set(ref_idx)
        for b in splits:
            if a!=b: assert not set(splits[a]) & set(splits[b])
    assert set(y[splits['train']])==set(classes)
    return mag,good,y,block_id,classes,ref_idx,ref,splits,np.array(raw_test_idx)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('ml/calibration_lite/results'))
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    inventory=audit(args.data_dir);write_json(out/'inventory.json',inventory)
    data=json.loads((args.data_dir/SESSION).read_text())
    mag,good,y,block_id,classes,ref_idx,ref,splits,raw_test_idx=prepare(data)
    write_json(out/'split.json',dict(session=SESSION,classes=classes,reference_indices=ref_idx.tolist(),
        reference=ref, indices={k:v.tolist()for k,v in splits.items()},
        raw_test_indices=raw_test_idx.tolist(),trim_samples=TRIM,
        note='Same-session labeled-block holdout, NOT independent trials, sessions, or users.'))
    tr,dv,te=[splits[k]for k in ('train','development','test')]
    candidates=[]
    for feature in FEATURES:
        x=transform(mag,feature,ref)
        for algorithm in ALGORITHMS:
            model=fit_model(x[tr],y[tr],algorithm,classes)
            pred,margin,dist=predict(model,x[dv]);m=metrics(y[dv],pred)
            model['feature']=feature
            model['margin_threshold']=choose_rejection(y[dv],pred,margin,dist,model['gate_radius'])
            accepted=(margin>=model['margin_threshold']) & (dist<=model['gate_radius'])
            candidates.append(dict(model=model,development=metrics(y[dv],pred,accepted)))
    # Freeze each feature family's selected algorithm before test evaluation.
    winners={f:max([r for r in candidates if r['model']['feature']==f],
                   key=lambda r:r['development']['macro_recall']) for f in FEATURES}
    lite=max([winners[f]for f in ('relative','direction','soft_direction')],
             key=lambda r:r['development']['macro_recall'])
    selected_names={'no_calibration':winners['raw']['model']['feature'],
                    'calibration_lite':lite['model']['feature']}
    write_json(out/'development.json',dict(candidates=candidates,selected=selected_names))
    evaluation={}
    stress_shifts=[np.array([0.,0.,0.]),np.array([50.,-30.,20.]),np.array([200.,-150.,100.])]
    for feature,record in winners.items():
        model=record['model'];x=transform(mag,feature,ref)
        pred,margin,dist=predict(model,x[te]);accepted=(margin>=model['margin_threshold'])&(dist<=model['gate_radius'])
        result={'algorithm':model['algorithm'],'development':record['development'],
                'test':metrics(y[te],pred,accepted),'test_total_with_rails':len(raw_test_idx),
                'test_rail_rejected':len(raw_test_idx)-len(te),
                'coverage_including_rails':float(accepted.sum()/len(raw_test_idx)),
                'correct_accepted_including_rails':float(np.sum(accepted&(pred==y[te]))/len(raw_test_idx)),
                'confusion':{a:{b:int(np.sum((y[te]==a)&(pred==b)))for b in classes}for a in classes},
                'blocks':{str(k):metrics(y[te][block_id[te]==k],pred[block_id[te]==k],accepted[block_id[te]==k])for k in sorted(set(block_id[te]))},
                'stress':[]}
        for shift in stress_shifts:
            for gain in [0.5,1.,2.]:
                # Hypothetical common gain/offset, including reference acquisition.
                # This is not a model of moving individual rings or changing wrist pose.
                transformed=mag*gain+shift
                r=reference(transformed[ref_idx]);xt=transform(transformed[te],feature,r)
                pp,mm,dd=predict(model,xt)
                result['stress'].append(dict(offset_ut=shift.tolist(),gain=gain,
                    label='ideal_affine_feature_stress_ignores_clipping_not_real_generalization',
                    prediction_agreement=float(np.mean(pp==pred)),
                    metrics=metrics(y[te],pp,(mm>=model['margin_threshold'])&(dd<=model['gate_radius']))))
        # Recalibration sensitivity: another disjoint open-hand block, same session.
        open_blocks=[b for b in labels_of(data)if b['code']=='eeeee']
        result['alternate_reference_stress']=[]
        for block in open_blocks[1:]:
            ri=np.arange(block['start']+TRIM,block['start']+TRIM+REFERENCE_SAMPLES)
            r=reference(mag[ri]); keep=~np.isin(te,ri)
            pp,mm,dd=predict(model,transform(mag[te[keep]],feature,r))
            result['alternate_reference_stress'].append(dict(block=block['id'],reference=r,
                note='Diagnostic re-reference; same-session, not a fresh wear. Reference samples excluded.',
                metrics=metrics(y[te[keep]],pp,(mm>=model['margin_threshold'])&(dd<=model['gate_radius']))))
        evaluation[feature]=result
        write_json(out/f'model_{feature}.json',dict(schema_version=1,model=model,reference_example=ref,
            input=dict(fields=['mx','my','mz'],axes='native_sensor',unit='raw_lsb',scale=SCALE,
                       rail_limit_lsb=32760,reference_samples=REFERENCE_SAMPLES),
            source_session=SESSION, status='experimental_not_HID_validated'))
    write_json(out/'evaluation.json',dict(selected=selected_names,features=evaluation,
        environment=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__)))
    print('SELECTED',selected_names)
    print('FEATURE | ALGORITHM | DEV MACRO | TEST EXACT | TEST MACRO | ACCEPT COVERAGE | ACCEPT ACC')
    for f,r in evaluation.items():
        t=r['test'];print(f,r['algorithm'],round(r['development']['macro_recall'],4),
            round(t['exact_chord_accuracy'],4),round(t['macro_recall'],4),round(t['coverage'],4),t['accepted_accuracy'])
    print('split counts', {k:len(v)for k,v in splits.items()},'rail test rejection',len(raw_test_idx)-len(te))

if __name__=='__main__':main()

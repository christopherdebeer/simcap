"""Export float32 C models and verify C/JavaScript against held-out Python outputs."""
import ctypes
import json
import subprocess
from pathlib import Path
import numpy as np
from experiment import SCALE, SESSION, reference, prepare, transform, predict, write_json
import argparse

ROOT=Path(__file__).resolve().parent

def float_literal(x):
    s=format(float(np.float32(x)),'.9g')
    return s+('f' if '.'in s or 'e'in s else '.0f')

def export_header(config, path):
    m=config['model'];n=len(m['classes']);dim=len(m['means'][0])
    feature={'raw':0,'relative':1,'direction':2,'soft_direction':3,'relative_magnitude':4}[m['feature']]
    text='''/* Generated experimental SIMCAP model. No HID output. C99, no heap. */
#include <math.h>
#include <stddef.h>
'''
    for name,key in [('Q','quadratic'),('W','linear'),('C','intercept'),('MU','means'),('VAR','gate_variances')]:
        vals=np.array(m[key]).ravel()
        text+=f'static const float {name}[{len(vals)}] = {{'+','.join(map(float_literal,vals))+'};\n'
    text+='static const char *SIMCAP_CLASSES[] = {'+','.join('"'+s+'"'for s in m['classes'])+'};\n'
    text+=f'''#define SIMCAP_CLASS_COUNT {n}
#define SIMCAP_NUM_FEATURES {dim}
#define SIMCAP_FEATURE {feature}
#define SIMCAP_SCALE {float_literal(config['input']['scale'])}
#define SIMCAP_RADIUS {float_literal(m['gate_radius'])}
#define SIMCAP_MARGIN {float_literal(m['margin_threshold'])}
/* out = class index, accepted (0/1), score margin, training-envelope distance.
 * Baseline is native-axis microtesla; tau is 3*reference median radius.
 * A host/device collector must supply 30 explicitly requested open-hand samples.
 */
void simcap_predict(const float raw[3], const float baseline[3], float tau, float out[4]) {{
  out[0]=-1;out[1]=0;out[2]=0;out[3]=0;
  if(SIMCAP_FEATURE!=0 && (!baseline || !isfinite(tau) || tau<=0))return;
  float x[3],norm2=0;
  for(int j=0;j<3;j++){{
    if(!isfinite(raw[j])||fabsf(raw[j])>=32760)return;
    if(SIMCAP_FEATURE!=0 && !isfinite(baseline[j]))return;
    x[j]=raw[j]*SIMCAP_SCALE-(SIMCAP_FEATURE==0?0:baseline[j]);norm2+=x[j]*x[j];
  }}
  if(SIMCAP_FEATURE==2||SIMCAP_FEATURE==3){{
    float divisor=SIMCAP_FEATURE==2?fmaxf(sqrtf(norm2),1e-9f):sqrtf(norm2+tau*tau);
    for(int j=0;j<3;j++)x[j]/=divisor;
  }}else if(SIMCAP_FEATURE==4)x[0]=sqrtf(norm2)/tau;
  float best=-INFINITY,second=-INFINITY;int chosen=0;
  for(int i=0;i<SIMCAP_CLASS_COUNT;i++){{
    float score=C[i];
    for(int j=0;j<SIMCAP_NUM_FEATURES;j++){{int k=i*SIMCAP_NUM_FEATURES+j;score+=Q[k]*x[j]*x[j]+W[k]*x[j];}}
    if(score>best){{second=best;best=score;chosen=i;}}else if(score>second)second=score;
  }}
  float distance=0;
  for(int j=0;j<SIMCAP_NUM_FEATURES;j++){{int k=chosen*SIMCAP_NUM_FEATURES+j;float d=x[j]-MU[k];distance+=d*d/VAR[k];}}
  out[0]=(float)chosen;out[2]=best-second;out[3]=distance;
  out[1]=(out[2]>=SIMCAP_MARGIN && distance<=SIMCAP_RADIUS)?1:0;
}}
'''
    path.write_text(text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);args=ap.parse_args()
    out=ROOT/'results';generated=ROOT/'generated';generated.mkdir(exist_ok=True)
    data=json.loads((args.data_dir/SESSION).read_text());mag,good,y,blocks,classes,ri,ref,splits,rawtest=prepare(data)
    indices=splits['test'];raw=mag[indices]/SCALE
    checks={}
    for name in ['raw','relative','soft_direction']:
        config=json.loads((out/f'model_{name}.json').read_text());model=config['model']
        path=generated/f'{name}.h';export_header(config,path)
        lib=generated/f'{name}.so'
        subprocess.run(['cc','-O2','-std=c99','-shared','-fPIC','-x','c',str(path),'-lm','-o',str(lib)],check=True)
        library=ctypes.CDLL(str(lib));f=library.simcap_predict
        ptr=ctypes.POINTER(ctypes.c_float);f.argtypes=[ptr,ptr,ctypes.c_float,ptr];f.restype=None
        xyz=(ctypes.c_float*3)(*ref['baseline']);pred=[];accepted=[];margins=[]
        for row in raw:
            result=(ctypes.c_float*4)();f((ctypes.c_float*3)(*row),xyz,ref['tau'],result)
            pred.append(classes[int(result[0])]);accepted.append(bool(result[1]));margins.append(result[2])
        pp,mm,dd=predict(model,transform(mag[indices],name,ref));aa=(mm>=model['margin_threshold'])&(dd<=model['gate_radius'])
        c_correct=int(np.sum(pp==pred));c_accept=int(np.sum(aa==accepted));assert c_correct==len(pp);assert c_accept==len(pp)
        for invalid in [[float('nan'),0,0],[32767,0,0],[-32768,0,0]]:
            result=(ctypes.c_float*4)();f((ctypes.c_float*3)(*invalid),xyz,ref['tau'],result);assert result[0]==-1 and result[1]==0
        payload=dict(config=config,reference=ref,raw=raw.tolist(),referenceRaw=(mag[ri]/SCALE).tolist())
        code='''const fs=require('fs');const Lite=require(process.argv[1]);const d=JSON.parse(fs.readFileSync(0,'utf8'));
const m=new Lite(d.config);if(d.config.model.feature!=='raw'&&m.predict([0,0,0]).reason!=='needs_reference')throw Error('missing reference');
let calibration;for(const x of d.referenceRaw)calibration=m.addReference(x);if(!calibration.ready)throw Error('reference');
for(const x of [[NaN,0,0],[32767,0,0],[-32768,0,0]])if(m.predict(x).accepted)throw Error('invalid accepted');
process.stdout.write(JSON.stringify({reference:calibration.calibration,pred:d.raw.map(x=>m.predict(x))}));'''
        result=subprocess.run(['node','-e',code,str(ROOT/'runtime.cjs')],input=json.dumps(payload),text=True,capture_output=True,check=True)
        js=json.loads(result.stdout);np.testing.assert_allclose(js['reference']['baseline'],ref['baseline']);np.testing.assert_allclose(js['reference']['tau'],ref['tau'])
        jp=[p['code']for p in js['pred']];ja=[p['accepted']for p in js['pred']]
        assert np.all(pp==jp);assert np.all(aa==ja)
        # Mathematical invariance: offset for relative; positive common gain+offset for soft-direction.
        shift=np.array([200.,-150.,100.]);gain=2 if name=='soft_direction' else 1
        if name!='raw':
            changed=mag*gain+shift;r=reference(changed[ri])
            np.testing.assert_allclose(transform(changed[indices],name,r),transform(mag[indices],name,ref),rtol=1e-10,atol=1e-10)
        numeric_count=sum(np.asarray(model[k]).size for k in ['quadratic','linear','intercept','means','gate_variances'])
        checks[name]=dict(test_vectors=len(pp),c_label_matches=c_correct,c_rejection_matches=c_accept,
            js_label_matches=int(np.sum(pp==jp)),js_rejection_matches=int(np.sum(aa==ja)),
            float32_parameters=numeric_count,float32_parameter_bytes=numeric_count*4,
            max_c_margin_absolute_error=float(np.max(np.abs(mm-margins))),
            invalid_input_tests='passed',reference_capture_js='passed',
            c_header_bytes=path.stat().st_size)
        lib.unlink()
    write_json(out/'runtime_validation.json',checks);print(json.dumps(checks,indent=2))

if __name__=='__main__':main()

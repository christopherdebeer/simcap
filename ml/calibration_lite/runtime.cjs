/* Dependency-free ES5-compatible classifier core for Espruino/Node.
 * No Bluetooth, timers, HID output or automatic calibration adaptation.
 * Candidate only: held-out errors remain too high for unattended HID.
 */
function SimcapLite(config) {
  var m = config.model;
  var n = m.classes.length, dim = m.means[0].length;
  function flatten(rows) { var a=[]; for(var i=0;i<rows.length;i++) a=a.concat(rows[i]); return new Float32Array(a); }
  var q=flatten(m.quadratic), w=flatten(m.linear), c=new Float32Array(m.intercept);
  var mu=flatten(m.means), variance=flatten(m.gate_variances);
  var baseline=null, tau=0, referenceBuffer=[];
  function median(x) { x.sort(function(a,b){return a-b;}); var k=x.length>>1; return x.length%2?x[k]:(x[k-1]+x[k])/2; }
  function valid(raw) {
    if (!raw || raw.length!==3) return false;
    for(var j=0;j<3;j++) if(typeof raw[j]!=='number'||!isFinite(raw[j])||Math.abs(raw[j])>=config.input.rail_limit_lsb) return false;
    return true;
  }
  this.resetCalibration=function(){baseline=null;tau=0;referenceBuffer=[];};
  this.setCalibration=function(ref) {
    if(!ref || !ref.baseline || ref.baseline.length!==3 || !(ref.tau>0) || !isFinite(ref.tau)) throw new Error('Invalid calibration');
    for(var i=0;i<3;i++) if(!isFinite(ref.baseline[i])) throw new Error('Invalid baseline');
    baseline=ref.baseline.slice();tau=ref.tau;
  };
  this.addReference=function(raw) {
    if(!valid(raw)) return {ready:false,reason:'invalid_or_saturated'};
    if(referenceBuffer.length>=config.input.reference_samples) throw new Error('Reset calibration before collecting another reference');
    var v=[];for(var j=0;j<3;j++)v.push(raw[j]*config.input.scale);
    referenceBuffer.push(v);
    if(referenceBuffer.length<config.input.reference_samples)return{ready:false,count:referenceBuffer.length};
    var b=[];
    for(j=0;j<3;j++){var axis=[];for(var i=0;i<referenceBuffer.length;i++)axis.push(referenceBuffer[i][j]);b.push(median(axis));}
    var norms=[];
    for(i=0;i<referenceBuffer.length;i++){var sum=0;for(j=0;j<3;j++)sum+=Math.pow(referenceBuffer[i][j]-b[j],2);norms.push(Math.sqrt(sum));}
    var noise=median(norms);
    if(!(noise>=1e-6)){referenceBuffer=[];return{ready:false,reason:'degenerate_reference'};}
    baseline=b;tau=noise*3;referenceBuffer=[];
    return{ready:true,calibration:{baseline:b.slice(),noise:noise,tau:tau}};
  };
  this.predict=function(raw) {
    if(!valid(raw))return{accepted:false,code:null,reason:'invalid_or_saturated'};
    if(m.feature!=='raw'&&!baseline)return{accepted:false,code:null,reason:'needs_reference'};
    var x=[],sum=0;
    for(var j=0;j<3;j++){var v=raw[j]*config.input.scale;if(m.feature!=='raw')v-=baseline[j];x.push(v);sum+=v*v;}
    if(m.feature==='direction'||m.feature==='soft_direction'){
      var divisor=m.feature==='direction'?Math.max(Math.sqrt(sum),1e-9):Math.sqrt(sum+tau*tau);
      for(j=0;j<3;j++)x[j]/=divisor;
    } else if(m.feature==='relative_magnitude')x=[Math.sqrt(sum)/tau];
    var best=-Infinity,second=-Infinity,k=-1;
    for(var i=0;i<n;i++){
      var score=c[i];for(j=0;j<dim;j++){var p=i*dim+j;score+=q[p]*x[j]*x[j]+w[p]*x[j];}
      if(score>best){second=best;best=score;k=i;}else if(score>second)second=score;
    }
    var distance=0;for(j=0;j<dim;j++){var d=x[j]-mu[k*dim+j];distance+=d*d/variance[k*dim+j];}
    var margin=best-second,accepted=margin>=m.margin_threshold&&distance<=m.gate_radius;
    return{code:m.classes[k],accepted:accepted,margin:margin,distance:distance,
           reason:accepted?'candidate':(distance>m.gate_radius?'outside_training_envelope':'ambiguous')};
  };
}
if(typeof module!=='undefined' && module.exports)module.exports=SimcapLite;

/* Generated experimental SIMCAP model. No HID output. C99, no heap. */
#include <math.h>
#include <stddef.h>
static const float Q[30] = {-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f};
static const float W[30] = {-1.27844004e-07f,-2.67848645e-06f,-1.44591581e-06f,0.00677160313f,0.00127711496f,0.00327392621f,-0.00400832435f,-0.00147998927f,0.00066199014f,-0.000338505371f,0.00145506102f,0.0007433258f,0.0062690475f,0.00206179125f,0.000896852871f,-0.00240458967f,-0.00278434157f,0.00161866855f,0.00139536906f,0.00195451546f,0.00165920774f,0.000892180717f,0.00101673068f,-0.00205774279f,0.00236507063f,0.00112297991f,0.00086692546f,0.00316129299f,0.00189703703f,0.00081405876f};
static const float C[10] = {-1.93393066e-06f,-6.28017282f,-1.19598746f,-0.550876498f,-2.64814138f,-2.47165823f,-1.73549259f,-1.80777395f,-0.72322607f,-1.24707818f};
static const float MU[30] = {-0.010850694f,-0.849971056f,-1.09953701f,574.736328f,405.270172f,2489.6355f,-340.204468f,-469.648834f,503.406006f,-28.7304688f,461.738281f,565.257141f,532.082214f,654.273499f,682.005798f,-204.088318f,-883.562317f,1230.90576f,118.431236f,620.231445f,1261.73352f,75.723381f,322.641785f,-1564.79675f,200.734161f,356.358124f,659.247681f,268.31311f,601.991699f,619.045532f};
static const float VAR[30] = {4243.72412f,15866.6289f,38022.168f,11071.1475f,121816.414f,865793.188f,4779.41455f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f,27633.3145f,15866.6289f,38022.168f,10059.9707f,89390.5f,57858.7539f,4243.72412f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f};
static const char *SIMCAP_CLASSES[] = {"eeeee","eeeef","eeefe","eeeff","eefee","eefff","efeee","feeee","ffeee","fffff"};
#define SIMCAP_CLASS_COUNT 10
#define SIMCAP_NUM_FEATURES 3
#define SIMCAP_FEATURE 1
#define SIMCAP_SCALE 0.09765625f
#define SIMCAP_RADIUS 7.69019794f
#define SIMCAP_MARGIN 0.0f
/* out = class index, accepted (0/1), score margin, training-envelope distance.
 * Baseline is native-axis microtesla; tau is 3*reference median radius.
 * A host/device collector must supply 30 explicitly requested open-hand samples.
 */
void simcap_predict(const float raw[3], const float baseline[3], float tau, float out[4]) {
  out[0]=-1;out[1]=0;out[2]=0;out[3]=0;
  if(SIMCAP_FEATURE!=0 && (!baseline || !isfinite(tau) || tau<=0))return;
  float x[3],norm2=0;
  for(int j=0;j<3;j++){
    if(!isfinite(raw[j])||fabsf(raw[j])>=32760)return;
    if(SIMCAP_FEATURE!=0 && !isfinite(baseline[j]))return;
    x[j]=raw[j]*SIMCAP_SCALE-(SIMCAP_FEATURE==0?0:baseline[j]);norm2+=x[j]*x[j];
  }
  if(SIMCAP_FEATURE==2||SIMCAP_FEATURE==3){
    float divisor=SIMCAP_FEATURE==2?fmaxf(sqrtf(norm2),1e-9f):sqrtf(norm2+tau*tau);
    for(int j=0;j<3;j++)x[j]/=divisor;
  }else if(SIMCAP_FEATURE==4)x[0]=sqrtf(norm2)/tau;
  float best=-INFINITY,second=-INFINITY;int chosen=0;
  for(int i=0;i<SIMCAP_CLASS_COUNT;i++){
    float score=C[i];
    for(int j=0;j<SIMCAP_NUM_FEATURES;j++){int k=i*SIMCAP_NUM_FEATURES+j;score+=Q[k]*x[j]*x[j]+W[k]*x[j];}
    if(score>best){second=best;best=score;chosen=i;}else if(score>second)second=score;
  }
  float distance=0;
  for(int j=0;j<SIMCAP_NUM_FEATURES;j++){int k=chosen*SIMCAP_NUM_FEATURES+j;float d=x[j]-MU[k];distance+=d*d/VAR[k];}
  out[0]=(float)chosen;out[2]=best-second;out[3]=distance;
  out[1]=(out[2]>=SIMCAP_MARGIN && distance<=SIMCAP_RADIUS)?1:0;
}

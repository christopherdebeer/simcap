/* Generated experimental SIMCAP model. No HID output. C99, no heap. */
#include <math.h>
#include <stddef.h>
static const float Q[30] = {-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f,-7.36004877f,-2.07690811f,-1.92628407f};
static const float W[30] = {-0.0577945113f,-0.880868375f,-1.07306695f,3.06933928f,0.597699881f,3.51226234f,-6.35660887f,-2.46356773f,2.4679594f,-0.580064476f,2.62602735f,2.98077679f,6.9953208f,2.55223536f,2.39608955f,-1.84681547f,-2.32652044f,3.11183977f,1.23893178f,1.82972181f,3.44107652f,0.69529295f,0.8406142f,-3.76789212f,3.80579948f,1.90859628f,3.27261424f,4.3654747f,2.76490593f,2.63752866f};
static const float C[10] = {-0.242955208f,-1.96400964f,-2.89353538f,-1.9946425f,-3.19137287f,-2.02415109f,-1.99189472f,-1.94401765f,-2.32044768f,-2.47037411f};
static const float MU[30] = {-0.00392623153f,-0.212062433f,-0.278532892f,0.208513528f,0.143891752f,0.911667764f,-0.431832016f,-0.593085408f,0.640601099f,-0.0394062921f,0.632196367f,0.773711622f,0.475222468f,0.614431441f,0.621946037f,-0.125462174f,-0.56009233f,0.807731271f,0.0841660053f,0.440491766f,0.893190265f,0.047234267f,0.202371553f,-0.978020906f,0.258544445f,0.459480196f,0.849462986f,0.296565622f,0.665630341f,0.684615672f};
static const float VAR[30] = {0.00850875303f,0.0286650434f,0.0129783554f,0.00339671667f,0.0120371245f,0.0927232057f,0.00760369329f,0.0144577911f,0.0293194987f,0.00339671667f,0.0120371245f,0.0129783554f,0.00421955576f,0.0120371245f,0.0129783554f,0.00339671667f,0.0120371245f,0.0129783554f,0.00339671667f,0.0120371245f,0.0129783554f,0.00339671667f,0.0120371245f,0.0129783554f,0.00339671667f,0.0120371245f,0.0129783554f,0.00339671667f,0.0120371245f,0.0129783554f};
static const char *SIMCAP_CLASSES[] = {"eeeee","eeeef","eeefe","eeeff","eefee","eefff","efeee","feeee","ffeee","fffff"};
#define SIMCAP_CLASS_COUNT 10
#define SIMCAP_NUM_FEATURES 3
#define SIMCAP_FEATURE 3
#define SIMCAP_SCALE 0.09765625f
#define SIMCAP_RADIUS 6.36209106f
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

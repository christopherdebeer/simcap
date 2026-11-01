/* Generated experimental SIMCAP model. No HID output. C99, no heap. */
#include <math.h>
#include <stddef.h>
static const float Q[30] = {-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f,-5.89105184e-06f,-1.57563397e-06f,-6.5751118e-07f};
static const float W[30] = {0.000557911233f,0.000162578435f,3.74653901e-05f,0.00732964184f,0.00144237187f,0.00331283757f,-0.00345028541f,-0.00131473236f,0.000700901495f,0.000219533729f,0.00162031793f,0.000782237097f,0.00682708668f,0.00222704816f,0.000935764168f,-0.00184655061f,-0.00261908467f,0.00165757991f,0.00195340812f,0.00211977237f,0.0016981191f,0.00145021977f,0.00118198758f,-0.00201883144f,0.0029231098f,0.00128823682f,0.000905836758f,0.00371933193f,0.00206229393f,0.000852970057f};
static const float C[10] = {-0.017936755f,-6.78287077f,-0.966239691f,-0.651268244f,-3.09784913f,-2.27777433f,-1.97129917f,-1.86058509f,-0.937910557f,-1.53850269f};
static const float MU[30] = {47.3524323f,51.5914345f,28.4903069f,622.099609f,457.711578f,2519.22534f,-292.841187f,-417.207428f,532.99585f,18.6328125f,514.179688f,594.846985f,579.445496f,706.714905f,711.595642f,-156.725037f,-831.120911f,1260.49561f,165.79451f,672.672852f,1291.32336f,123.086662f,375.083191f,-1535.20691f,248.097443f,408.79953f,688.837524f,315.676392f,654.433105f,648.635376f};
static const float VAR[30] = {4243.72412f,15866.6289f,38022.168f,11071.1475f,121816.414f,865793.188f,4779.41455f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f,27633.3145f,15866.6289f,38022.168f,10059.9707f,89390.5f,57858.7539f,4243.72412f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f,4243.72412f,15866.6289f,38022.168f};
static const char *SIMCAP_CLASSES[] = {"eeeee","eeeef","eeefe","eeeff","eefee","eefff","efeee","feeee","ffeee","fffff"};
#define SIMCAP_CLASS_COUNT 10
#define SIMCAP_NUM_FEATURES 3
#define SIMCAP_FEATURE 0
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

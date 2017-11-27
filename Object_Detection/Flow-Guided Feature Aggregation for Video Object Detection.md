# Flow-Guided Feature Aggregation for Video Object Detection

### Abstract

이미지에서의 최신의 object detector를 비디오로 확장하는 것은 매우 어려운 일입니다. detection의 정확도는 비디오에서의 object의 모양이 달라지는 것(motion blure, video, defocus, rare pose) 때문에 많은 영향을 받습니다. 기존의 작업들은 box-level에서 temporal 정보를 이용하려고 했지만 이러한 모델은 end-to-end로 훈련되지 않습니다. 우리는 video-object-detection을 위해 flow-guided featue aggregation 방법을 제시합니다. 이는 feature-level 에서 temporal 한 정보를 이용합니다. motion 을 이용하여 근처 프레임의 feature를 통합함으로써 video recognition accuracy를 향상 시켰습니다. ImageNet VID 벤치마크에서 single-frame baseline의 성능을 크게 뛰어넘었습니다. VID는 특히 빠르게 움직이는 object 등이 포함되어 매우 어려운 데이터셋입니다. 우리의 모델은 2016 ImageNet VID 에서 승리를 거뒀고 어떠한 bells-and-whistle을 사용하지 않았습니다. Deep Feature Flow[49] 와 결합된 모델은 2017 ImageNet VID 챌린지에서 우승을 거뒀습니다. 코드 또한 공개되어 있습니다.

최근 몇년동안 object detection에서 많은 발전이 있었습니다. 최신의 방법들은 비슷한 2가지 구조를 가지고 있습니다. CNN 이 먼저 전체 이미지에 대한 feature를 추출하기 위해 적용되며 shallow detection-specific network가 feature map 을 부터 detection 결과를 생성합니다.

이 방법은 still image에서 우수한 결과를 얻었습니다. 그러나 이를 video object detectin에 바로 적용하는 것은 매우 어렵습니다. 비디오에서는 still 이미지에서는 거의 볼 수 없는 motion blur, video defocus, rare pose 와 같은 것들 때문에 still 이미지 모델을 바로 적용하면 recognition accuracy는 떨어지게 됩니다. 실험결과 최신의 still 이미지 object detector는 빠르게 움직이는 object에서 안좋은 성능을 보입니다.

짧은 시간 동안의 한 객체에 대한 여러개의 snapshot이 있기 때문에 비디오에서는 하나의 object에 대한 풍부한 정보를 가지고 있습니다. [18, 19, 12, 23] 에서 간단한 방법을 통해 이러한 temporal infomation을 사용합니다. 이들은 single frame에 object detector를 적용하고 post-processing을 통해 temporal 마다의 bounding box를 결합합니다. 이 단계에서 optical flow와 같은 motion estimation을 사용하고 object tracking과 같은 hand-crafted bounding box association rule을 사용합니다. 일반적으로 이러한 방법들은 single frame detecion box는 괜찮은 성능으로 뽑아내지만 전체의 detection 의 성능을 올리진 못합니다. 성능의 향상은 학습을 통한 것이 아닌 휴리스틱한 post-processing을 통해 이뤄집니다. 이는 end-to-end 학습을 할 수 없으며 이러한 방법들을 box-level 방법이라고 불릅니다.

우리는 video object detection에 대해 더 자세하게 살펴봅니다. 우리는 temporal infromation을 사용함으로써 detection의 성능을 향상시키고자 합니다. [11] 의 image recognition의 결과가 보여주듯이 feature가 가장 중요하며 우리는 temporal aggregation을 통해 per-frame feature를 사용하는 것을 제안합니다. 동일한 object의 feature는 video motion 때문에 프레임마다 spatially aligned 되어 있지 않습니다. 나이브한 feature aggregation은 성능을 떨어트릴수 있으며 이는 테이블 1에 나와있습니다. 이것은 학습과정 동안 motion을 모델링 하는 것이 중요하다는 것을 보여줍니다.

이 논문에서는 flow-guided feature aggregation(FGFA) 라는 것을 제안합니다. 피규어 1에서 볼 수 있듯이 feature extract 네트워크는 각각 프레임마다 적용되어져 per-frame feature를 생성합니다. 해당하는 프레임의 feature를 향상시키기 위해 [8]의 flow network는 해당 프레임과 근처 프레임 사이의 motion을 추정합니다. 근처의 프레임의 feature map은 motion에 따라 해당(중앙) 프레임의 feature가 warp 된것으로 볼 수 있습니다. warped(nearby) feature과 reference 프레임의 feature는 adaptive weighting 네트워크에 의해 합쳐지게 됩니다. 이렇게 합쳐진 feature는 detection 네트워크의 인풋으로 들어가고 reference 프레임에 대한 detection 결과를 출력하게 됩니다. feature extraction, flow estimation, feature aggregation , detection은 모두 end-to-end로 학습됩니다.

box-level 방법과 비교하면 이 논문은 feature-level을 다루고 end-to-end로 학습가능하며 seq-NMS와 같은 방법으로 보완이 가능합니다. 이는 per-frame feature의 성능을 향상시키며 더 좋은 bounding-box를 생성합니다. boxes의 품질은 box refine 을 통해 향상 됩니다. 우리의 방법론은 대규모 ImageNet VID 데이터셋에서 실험되었고 모듈을 제거하면서 하는 실험의 결과는 single frmae 베이스라인의 성능을 어떻게 향상시키는지를 검증합니다. box-level 방법과의 결합을 통해 성능을 향상키실 수 있습니다. 우리는 VID 벤치마크의 우승자와 비교하여 우리의 성능을 보고 합니다.

또한 motion의 강도에 따라 더 심층적인 평가를 수행합니다. 이 결과는 빠르게 움직이는 객체가 느리게 움직이는 것보다 어렵다는 것을 보여주고 우리의 접근방법이 잘 다룰수 있다는 것을 보여줍니다. 우리의 방법은 빠르게 움직이는 객체의 다양한 snapshot에서 좋은 feature를 만들 수 있습니다.

[49] 의 Deep Feature Flow 방법과 결합하여 VID 2017에서 우승을 할 수 있었습니다.

### 2. Related work

<b>Object detection from image.</b> : object detection을 위한 일반적인 방법은 CNN을 사용합니다. [11] 에서는 R-CNN 이라는 multi-stage 방법을 제안하였고 object detection을 위해 CNN이 proposal region을 판별하도록 하였습니다. Fast R-CNN, SPP-Net 등에서 속도 향상을 위해 ROI-pooling이 소개되어졌고 모든 region classfy는 이미지 전체에서 같은 feature를 공유합니다. Faster R-CNN에서는 RPN에 의해 proposal region을 생성하고 RPN, FAST R-CNN은 같은 feature를 공유합니다. 더 최근에는 RFCN이 ROI-pooling을 position-sensitivity ROI Pooling 으로 교체하여 feature sharing을 최대한 할 수 있게 하였습니다.


still-image object detection 방법과 다르게 우리는 비디오에서의 object detection에 초점을 맞춥니다. 이는 temporal 정보를 통합하여 conv feature map의 성능을 향상시키고 still-image obejct detection의 발전을 쉽게 이용할 수 있습니다.

<b> Object detection in video</b> 최근 ImageNet은 object detection을 비디오 도메인에서 수행하는 새로운 VID 챌린지를 소개합니다.이 챌린지에서 존재하는 대부분의 방법들은 temporal information을 "bounding box post-processing" 을 통해 다룹니다. [18, 19] T-CNN 에서는 예측된 bounding box를 인접한 프레임에 전달한 다음 optical-flow를 사용하여 high-confidence bounding box에 traking 알고리즘을 적용 해 tubelet을 생성합니다. box를 포함하는 tubelet은 tubelet classfication에 의해 re-scoring 되어집니다. seq-NMS 는 연속적인 프레임으로 부터 high-confidence bounding box의 근처의 box들을 제거합니다. boxes 들의 sequence는 average confidence(temporal 하게 average한다는 뜻인듯)로 re-scoring되며 근처의 다른 box들은 suppress 되게 됩니다. [23] MCMOT 는 post-processing을 일종의 multi-object tracking problem으로 간주합니다. hand-craft rule(detector confidence, color/motion clue, changing point detectin , forawrd-backward validation) 등을 사용해 bounding box가 추적중인 object에 속하는지를 판별하고 tracking 결과를 refine 합니다. 불행히도 이러한 방법들은 multi-stage pipeline이며 이전 단계의 결과가 다음 단계의 결과에 영향을 주게 됩니다. 그래서 이는 이전 단계에서 만들어진 error를 다시 수정하는 것이 어렵습니다.

이와 반대로 우리의 방법은 final box-level 에서가 아닌 feature-level에서의 temporal information을 고려합니다. 전체 시스템은 end-to-end로 학습 가능합니다. 또한 우리의 방법은 성능을 올리기 위해 다른 bounding box post-processing 기술을 적용 가능합니다.

<b>Motion estimation by flow</b> 비디오에서의 temporal information은 연속적인 프레임 사이의 관계를 구축하는 것이 중요합니다. optical flow는 다양한 비디오 분석, processing에서 많이 사용됩니다. 전통적인 방법[2, 15]는 variational 방법을 통해 optical flow를 계산하고 주로 small displacements를 주로 다룹니다. 최근의 접근방법은 [3] large displacemnets에 초점을 맞추고 [DeepFlow(44), EpicFLow(31)] 은 combinatorial matching을 variational 방법과 통합하엿습니다. 이러한 방법은 모두 hand-crafted 를 기반으로 하며 [FlowNet(8), 28, 17] 과 같은 딥러닝을 기반으로 하는 방법이 최근 나왔습니다. 우리의 작업과 가장 비슷한 논문은 Deep feature flow[49] 로써 비디오에서의 information redundancy를 사용해 성능을 약간 낮추지만 매우 빠른 연산속도가 가능 하다는 것을 보여줬습니다. 이는 flow sub-network와 recognition sub-network의 joint training의 가능성을 보여줍니다.

이 논문에서는 feature representation과 recognition 성능을 개선하기 위해서 연속적인 프레임에서의 appearance information을 연결하고 결합하는 다른 방법에 대해 집중합니다. 우리는 프레임간의 feature warping을 하기 위해 deep feature flow의 설계를 따릅니다.

<b>Feature aggregation</b> feature aggregtion은 action recognition, video description에서 광법위하게 사용됩니다. 대부분의 연구는 RNN을 활용하여 feature aggregation을 수행합니다. 다른 접근방법은 spatial-temporal convolution을 통해 바로 spatio-temporal feature를 추출합니다. 그러나 이러한 spatio-temporal convoltuion을 통한 방법의 커널 사이즈는 빠르게 움직이는 객체의 모델링을 제한 할 수 있습니다. 이를 해결하기 위해 큰 커널사이즈가 고려되어야 하지만 큰 커널 사이즈를 사용하면 파라미터 숫자가 늘어나 overfitting의 문제가 생기고 메모리 제한과, 연산량이 매우 증가합니다. 이와 반대로 우리는 flow-guided aggregation을 사용하여 다양한 유형의 object motion에 적용 가능합니다.

<b>Visual tracking</b> 최근에는 CNN을 object tracking에 사용하여 좋은 tracking accuracy를 달성했습니다. 새로운 target을 tarcking 할때 pre-train CNN과 레이어를 공유하지만 마지막의 binary classfication layer만 다른 네트워크를 생성하며 이는 online update 됩니다. tracking 과 video object detection과 다른 task 인데 첫번째 프레임에서 이미 object에 대해 localization이 되어 있다고 가정하며 class label을 예측하는 것이 필요하지 않습니다.

### 3. Flow Guided Feature Aggregation
### 3.1. A Baseline and Motivation

input 비디오 프레임 {I_i} i=1,,,,,n 이 주어졌을때 우리의 목표는 모든 프레임에 대해 object bounding box {y_i} i=1,,,,,,n 를 생성하는 것입니다. 베이스라인이 되는 접근 방법은 각각 프레임에 개별적으로 object detector를 적용하는 것입니다.

최근의 CNN을 기반으로 하는 object detector은 비슷한 구조를 보입니다. convolutional sub-network N_feat 는 인풋이지미 I에 적용되어져 feature map f = N_feat(I) 를 생성합니다. shallow detection-specific sub-network N_det 는 feature map f에 적용되어져 object detection 결과 y=N_det(f) 를 출력합니다.

비디오 프레임은 피규어 2와 같이 같은 object 에 대해 급격한 appearance change를 포함합니다. 단일 프레임에 대한 detection 결과는 안정적이지 않고 appearance가 명확하지 않은 경우 실패합니다. 피규어 1를 예로 보면 t time의 reference 프레임에 대한 "cat"에 대한 response는 매우 낮습니다. 이러한 것은 single frame detection이 실패하게 합니다. 근처의 t-10, t+10 프레임이 높은 response를 가지면 이들의 feature는 reference 프레임으로 전달되게 됩니다. reference frame에 대해 feature가 결합되면 detection이 성공하게 됩니다.

이러한 feature propagation, enhancement를 위해선 두가지 모듈이 필요합니다. 1) motion guided spatial warping : 이는 프레임 간의 motion을 추정하고 feature map을 warp 합니다. 2) feature aggregation : 이는 여러개의 프레임에서 어떻게 feature를 결합할지를 알아냅니다. 이 두개의 모듈과 feature extraction, detection 네트워크가 우리의 기본 요소입니다. 이들은 아래에서 자세히 기술합니다.

### 3.2. Model Design

<b>Flow-guided warping.</b> [48] 에서 동기를 얻어 reference 프레임 I_i와 근처의 프레임 I_j가 주어졌다면 flow network F에 의해 flow field M_ij = F(I_i, I_j) 가 추정됩니다. 

근처의 프레임의 feature map은 추정된 flow에 따라 warp 하게 됩니다. wapring function은 다음과 같이 정의됩니다.
<img src="https://latex.codecogs.com/gif.latex?f_%7Bj%5Crightarrow%20i%7D%20%3D%20W%28f_j%2C%20M_%7Bj%5Crightarrow%20i%7D%29%20%3D%20W%28f_j%2C%20F%28I_i%2C%20I_j%29%29" /> (1)

w는 bilinear warping function이며 feature map의 각각의 channel, location에 대해 적용됩니다. f_ji 는 프레임 j에서 프레임 i로의 warped feature map을 나타냅니다.

<b>Feature aggregation</b> feature warping 후에 reference 프레임은 근처의 프레임으로 부터의 여러개의 feature map을 accumulate 하게 됩니다. 이 feature map은 object에 대한 다양한 정보를 포함하게 됩니다. aggragation을 위해 우리는 각각 spatial location에 대해 다른 weight, 채널별로는 공유된 spatial weight를 사용합니다. warped feature f_ji 에 대한 2D weight는 w_ji 로 나타내며 reference 프레임 f_i에 대한 aggregated feature는 다음과 같이 얻어집니다.

<img src="https://latex.codecogs.com/gif.latex?%5Cbar%7Bf_i%7D%20%3D%20%5Csum%5E%7Bi&plus;K%7D_%7Bj%3Di-K%7Dw_%7Bj%5Crightarrow%20i%7Df_%7Bj%5Crightarrow%20i%7D" /> (2)

K는 aggergation 할 근처의 프레임의 범위를 나타내며 디폴트로는 K=10을 사용합니다. (2) 수식은 일종의 memory buffer의 feature에 다양한 weight를 사용하는 일종의 attention model과 비슷합니다. 

aggregated feature는 그 후 detection sub-network의 인풋으로 들어가고 y_i = N_det(f_i) 의 출력을 하게 됩닌다. 베이스라인과 기존의 box-level 방과 비교하면 final detection 전에 여러개의 프레임의 정보를 aggregate 하게 됩니다.

<b>Adaptive weight</b> adaptive weight는 모든 buffer 프레임 (I_i-K,,,I_i_K) 의 referece 프레임 I_i에 대한  spatial point의 중요성을 나타냅니다. location p에 대해 살펴보면 warped feature f_ji(p) 가 f_i(p) 와 가깝다면(warped feature가 referecne 프레임의 feature와 비슷한 값을 가진다면) 큰 weight를 할당하게 됩니다. 우리는 warped fature와 referecne feature와의 유사성을 특정하기 위해 cosine metric을 사용했습니다. 또한 우리는 N_feat(I) 에 의해 얻어진 feature를 바로 사용하지 않고 tiny fully convolutional network 을 feature f_i, f_ji 에 적용합니다. 이는 feature를 유사성 measure를 위한 새로운 embedding으로 사영하고 이는 embedding sub-network로 불립니다. 
(embedding sub-network가 왜 필요한거지..??) weight는 다음과 같이 계산합니다.

<img src="https://latex.codecogs.com/gif.latex?w_%7Bj%5Crightarrow%20i%7D%20%3D%20exp%28%5Cfrac%7Bf_%7Bj%5Crightarrow%20i%7D%5Ee%28p%29%20%5Ccdot%20f_%7Bi%7D%5Ee%28p%29%7D%7B%7Cf_%7Bj%5Crightarrow%20i%7D%5Ee%28p%29%7C%20%7Cf_%7Bi%7D%5Ee%28p%29%7C%7D%29" /> (3)

여기서 f^e 는 유사성 측정을 위한 embedding feature를 나타내며 w_ji 는 모든 spatail point p에 대해 nomlaization 하게 됩니다. <img src="https://latex.codecogs.com/gif.latex?%5Csum%5E%7Bi&plus; K%7D_%7Bj%3Di-K%7D%20w_%7Bj%5Crightarrow%20i%7D%28p%29%20%3D1" /> (각 spatial point p에 대한 모든 프레임의 weight의 합이 1이 되도록!!) weight의 추정은 embedding feature 사이의 코사인 유사도에 일종의 softmax 연산을 취하는 것으로 볼 수 있습니다. (normalization 하는 과정이 일종의 softmax 연산이네.)

### 3.3. Training and Inference

<b>Inference</b> 알고리즘 1은 inference 알고리즘을 나태냅니다. 연속적인 프레임과 aggregation range K가 주어지면 제안된 방법은 순차적으로 feature를 추출합니다. 이는 (시작과 끝의 K번째 프레임을 제외한 2K +1)의 길이의 silding feature buffer를 통해 각 i번째 프레임의 feature를 저장합니다. feature network는 시작부터 K+1 프레임 까지의 프렝미을 통해 feature buffer를 초기화 합니다. 그 후 모든 비디오 프레임에 대해 video object detection을 수행하며 feature buffer를 업데이트 합니다. i 번째 reference 프레임에 대해서 근처의 프레임의 feature map은 reference 프레임에 대해 warped 되며 이들의 aggregation weight 들이 계산됩니다. warped feature은 aggregation되고 detection 네트워크의 인풋으로 들어갑니다. (i+1) 번째 프레임을 reference 프레임으로 사용하기 전에 (i+K+1) 의 프레임에 해당하는 feature map을 추출하고 feature buffer에 추가됩니다. 

시간 복잡도를 계산하기 위해 sigle-frame 베이스라인과의 상대적인 시간복잡도는 다음과 같습니다.

<img src="https://latex.codecogs.com/gif.latex?r%20%3D%201&plus;%20%5Cfrac%7B%282K&plus;1%29%28O%28F%29%20&plus;%20O%28%5Cvarepsilon%20%29%20&plus;%20O%28W%29%29%7D%7BO%28N_%7Bfeat%7D%29%20&plus;%20O%28N_%7Bdet%7D%29%7D" />

O()는 time complexity를 나타내는 함수이며 일반적으로 N_det, e, W의 복잡도는 N_feat와 비교하면 무시되게 됩니다. 이는 근사적으로 <img src="https://latex.codecogs.com/gif.latex?r%20%3D%201&plus;%20%5Cfrac%7B%282K&plus;1%29%28O%28F%29%7D%7BO%28N_%7Bfeat%7D%29%7D" />이 됩니다. 증가한 연산량은 대부분 F(flow network)로 부터 이며 이는 N_feat보다는 훨씬 낮으므로 감당할 만 합니다.

<b>Training</b> 전체 FGFA 구조는 완전히 미분가능하며 end-to-end로 학습 가능합니다. 주목 해야 할 점은 feature warping module이 bilinear interpolation으로 구현되고 flow field, feature map 에 미분가능 하다는 점입니다. 

<b>Temproal dropout</b> SGD 학습과정에서 aggregation range K는 메모리에 의해 제한 됩니다. 우리는 트레이닝 때는 K=2 테스팅 때는 큰 K를 사용합니다. 이는 트레이닝, 테스팅 중에 adaptive weight가 normalization 되기 때문에 문제가 되지 않습니다. 트레이닝 동안 근처의 프레임은 inference와 동일한 K range의 프레임 중 랜덤하게 샘플링 됩니다. [37] dropout 기법과 비슷하게 이는 일종의 temporal dropout으로 설명될 수 있으며 랜덤하게 temporal 프레임을 제거함으로서 dropout과 비슷한 효과를 가져옵니다. 테이블3 에서 볼 수 있듯이 이러한 트레이닝이 잘 작동합니다.

### 3.4. Network Architectur

몇몇의 FGFA의 sub-네트워크의 소개합니다.

<b>Flow Network</b> 우리는 FlowNet[8] 의 간단한 버젼을 사용했으며 이는 Flying Chairs 데이터셋에서 pre-train되었습니다. 이는 이미지의 half resolution을 인풋으로 받아 strid 4의 출력을 냅니다. feature network의 아웃풋 stride 가 16이기 때문에 flow field와 feature map의 크기를 맞추기 위해 flow field는 downscale 됩니다.

<b>Feature network.</b> 우리는 Resnet-50,Resnet-101, Inception-Resnet을 feature network로 적용합니다. 원래의 Inception-Resnet은 이미지 recognition을 위해 디자인 되었습니다. object detection에 사용하가ㅣ 위해 feature misalignment 문제를 해결하려고 우리는 "Aligned Incpetion-Resnet" 이라고 불리는 수정된 버젼을 사용합니다. ResNet-50, 101, Aligned Inception-Resnet 모델은 모두 ImageNet 에서 pre-train 되었습니다.

pre-train 된 모델이 우리의 FGFA 모델에서의 feature network로 사용됩니다. 우리는 이 세개의 모델들을 object detection을 위해 약간 변경합니다. 마지막의 average pooling, fc-layer를 제거하고 convolution layer는 유지합니다. feature resolution을 증가시키기 위해 [4,5] 에 따라 마지막 블록(네트워크 전체의 stride를 말하는 것 같다.)의 stride를 32에서 16으로 바꿧습니다. 특히, 마지막 블록의 시작에서(conv5) stirde를 2에서 1로 바꿔 전체 stride를 32에서 16으로 바꿨습니다. receptive field size를 유지하기 위해 마지막 block의 conv layer의 dialation을 2로 설정했습니다. 마지막에 새로운 3x3 conv를 추가하여 feature 차원을 1024로 맞춥니다.

<b>Embedding network</b> 이는 1x1x512, 3x3x512, 1x1x2048 의 세개의 convolution layer로 구성되어져 있으며 랜덤하게 초기화 됩니다.

<b>Detection network</b> [49] 에 의해 디자인된 R-FCN 을 사용합니다. 마지막 1024 feature map에 RPN, R-FCN sub-network가 적용되며 각각 512, 512의 차원의 feature map을 인풋으로 받습니다. 9 anchor가 RPN에서 사용되었으며 각 이미지 마다 300개의 proposal을 생성합니다. R-FCN에서의 position-sensitive score는 7x7로 수행됩니다.

### 4. Experiments
### 4.1. Experiment Setup


<b>ImageNet VID dataset</b> 이는 video object detection을 위한 대형의 밴치마크입니다. [18, 23] 에 따라 학습과 evaluation은 3862개의 트레이닝 셋과 55개의 validation에서 각각 수행됩니다. 비디오 snippet은 완전히 annotation이 조재하며 이들은 25fps, 30fps를 가집니다. 30개의 카테고리가 존재하며 이들은 DET에서의 카테고리의 subset 입니다.

<b>Slow, medium, and fast motion.</b> 더 나은 분석을 위해 ground truth object는 motion speed에 따라 분류됩니다. object의 속도는 근처의 프레임(+=10 프레임) 의 평균 IOU score로 측정하게 됩니다. 이 speed indicator는 'motion IOU' 라고 불립니다. motion IOU가 낮을 수록 더 빠르게 움직이는 물체입니다. 피규어 3은 motion IOU의 score의 히스토그램을 보여줍니다. object는 slow(score > 0.9), medium(score 0.7 ~ 0.9), fast(score < 0.7) 로 나눠집니다. 피규어 4에 이들 그룹에 대한 예가 있습니다. 

evalutation시에 MAP외에도 slow, medium, fast 그룹에 대한 map를 보고 하며 각각 map(slow), map(medium), map(fast) 로 표기됩니다. 이를 통해 더 깊은 이해와 디테일한 분석을 제공합니다.

<b>Implementation details.</b> 

학습 과정은 [18, 23] 에 따라 DET, VID의 트레이닝 셋을 모두 활용합니다. 2 단계로 학습이 이뤄지면 첫번째 단계에서는 feature, detection 네트워크가 DET 의 학습데이터를 통해 학습되며 VID에 속하는 카테고리의 라벨만을 사용합니다. SGD를 통해 최적화를 수행하며 각 미니 배치는 하나의 이미지를 사용합니다. 4개의 GPU를 통한 120k iteration 이 수행되며 각 gpu는 하나의 mini-batch에 대해 수행합니다. learning rate는 80k 까지는 10^-3 그 이후의 40k 가지는 10^-4 로 설정합니다. 
 
 두번째 단계에서는 FGFA 모델이 VID 트레이닝 셋을 통해 학습됩니다. 이때의 weight는 첫번째 단계에서 학습한 weight로 초기화 하며 4개의 GPU를 통한 60K iteration이 수행됩니다. 처음 40k는 10^-3, 그 이후의 20k는 10^-4 의 learning rate를 적용합니다. 학습과 테스팅 모두 이미지의 제일 면이 600 px이 되도록 하며 flow network를 위해서는 제일 작은 면을 300 px이 되도록 resize합니다.
 
 ### 4.2. Ablation Study
 
<b>FGFA Architecture Design</b> 테이블 1은 우리의 FGFA와 single frame baseline, 그 변형들을 비교합니다. (a) 방법은 single-frame baseline이며 Resnet-101을 사용해 73.4% map을 얻었고 이는 [49]의 R-FCN, ResNet-101 을 사용한 73.9% map 와 비슷한 결과를 얻었습니다. 이는 우리의 baseline이 경쟁력이 있으며 evaluation을 위한 유용한 자료로 사용됨을 나타냅니다. 우리는 비교를 정확히 하기 위해 multi-scale training/testing, exploting context information, model ensemble 같은 bells and whistel을 사용하지 않았습니다.
 
다른 motion group에 대한 실험은 빠르게 움직이는 물체를 detection 하는 것이 매우 어렵다는 것을 나타냅니다. slow motion의 경우 82.5% map을 얻었으며 fast motion의 경우 51.4% 로 떨어졌습니다. 다른 크기의 객체가 다른 속도로 움직일 수 있으므로 우리는 object size에 대한 실험을 진행했습니다. 테이블 2는 small, middle, large object의 다른 motion speed에 대한 결과를 보여줍니다. 이는 'fast motion' 이 object size에 관계없이 본질적으로 어려운 문제라는것을 보여줍니다.

(b) 방법은 FGFA의 부분을 제거한 naive feature aggregation으로써 flow motion이 사용되지 않았습니다. 여기서 flow map M_ij는 모두 0으로 설정이 되었고 adaptive weighting 또한 사용되지 않았습니다. weight w_ij는 constant 1/2K+1로 설정되었습니다. 이 모델의 학습 방법은 FGFA와 동일하게 하였고 map는 72%로 떨어졌으며 베이스라인보다 1.4% 노픈 결과입니다. fast motion group에 대한 map(fast) 는 51.4% -> 44.6% slow group에 대한 map(slow)는 82.4% -> 82.3% 으로 떨어져 flow motion, adaptive weighting이 motion 정보를 이용하는데 결정적이라는 것을 보여줍니다.

(d) 방법은 제안된 FGFA 방법으로서 (c) 에 flow-guided feature aggregation을 추가한 것입니다. 이는 map를 2% 높여 76.3% 의 성능을 보였고 fast motion group의 경우 52.3% -> 57.6% 로 다른 group보다 높은 성능 향상을 보였습니다. 피규어 5는 adaptive weight가 (c) 보다 (d) 에서 더 균등하게 분포하는 것을 보여줍니다. 이는 특히 fast motion의 경우 눈에 뜁니다.  이는 flow-guided feature aggregation이 효과적으로 근처의 프레임의 정보를 추출한다는 것을 보여줍니다. (feature aggregation을 통해 warped feature를 구해 여기에 temporal한 정보가 들어가 가까운 프레임에 가중치를 덜 줘도 된다?) 이 FGFA 방법은 map를 2.9% 정도 올렸고 single-frame baseline에 비해 6.2 map(fast) 를 향상시켰습니다.

(e) 방법은 (d) end-to-end 학습을 하지 않는 버젼입니다. 이는 single-frame basline (a) 에서의 feature, detection sub-network를 가져오고 pre-train된 flownet을 사용합니다. 학습 도안 ㅇ이들 모듈은 고정되어져 있으며 embedding sub-network만이 학습됩니다. 이는 분명히 (d) 보다 안좋은 성능으 보이며 FGFA의 end-to-end 학습의 중요성을 보여줍니다.

실행시간에 관해서는 제안된 FGFA 방법은 프레임당 773ms가 걸립니다. 이는 single-frame baseline (288ms) 보다 느린데 이는 flow network가 2K+1 (K=10) 번을 forward 연산을 계산해야 하기 대문입니다. 이러한 연산을 줄이기 위해 다른 버젼의 FGFA를 실험했습니다. 이는 인접한 2개의 프레임에만 적용되는 flow network를 사용했습니다. 인접하지 않은 flow field는 중간의 flow field를 합성하여 만들어집니다. 이러한 방식으로 인접한 프레임의 각각의 연산은 다른 reference 프레임을 사용할때도 재사용 될 수 있습니다. 이 경우 FGFA의 실행시간은 356ms이며 flow field 의 오차의 누적 때문에 1% 정도의 성능 하락을 가져옵니다.

<b>\# frames in training and inference</b> 메모리의 문제 때문에 우리는 경량화된 Resnet-50에 대해 실험합니다. SGD를 이용한 최적화에서 미니배치당 2,5 프레임을 사용하였고 테스트 시에는 1,5,9,13,18,21,25 프레임을 사용한 실험 결과를 보여줍니다. 테이블 3은 2,5 개의 프레임을 사용한 결과가 매우 비슷한 것을 보여줍니다. 이는 우리의 temporal dropout 의 효과를 보여줍니다. 기대했던 대로 더 많은 프레임을 사용하면 더 높은 성능을 보여줍니다. 21프레임 까지 성능향상을 보이고 그 이후는 보이지 않았으며 우리는 학습에서는 2프레임, 테스트에서는 21프레임을 사용합니다.

### 4.3. Combination with Box-level Technique

우리의 방법은 비디오 프레임에서의 feature의 품질과 recognition 성능의 향상에 중점을 둡니다. object box의 추력은 post-processing을 통한 box-level 기술을 통해 향상 될 수 있습니다. 우리는 motion guided propagation (MGP) [18], Tubelet rescoring [18], and Seq-NMS [12 라는 3가지 방법을 실험을 했습니다. MGP, Tubelet rescoring은 2015 VID의 승자가 사용한 것입니다. 우리는 MGP, Tubelet rescoring의 공식 코드를 활용했으며 seq-NMS를 재 구현했습니다.

테이블 4는 그 결과를 보여줍니다. 이 세가지 기술은 먼저 Resnet-101을 사용하는 single frame baseline에 적용돼었고 이들은 성능을 향상 시킵니다. 이는 이러한 post-processing의 기술이 효과적이라는 것을 나타냅니다. seq-nms는 가장 큰 성능 향상을 보입니다. Resnet-101을 사용한 FGFA 와 이들을 결합했을때 MGP, Tubelet rescoring의 효과는 없었지만 seq-NMS는 여전히 효과적이였습니다.(map가 78.4% 로 향상) Aligned-Inception-Resnet을 feature network로 사용함으로써 FGFA + Seq-NMS의 결과는 80.1% 을 달성합니다. Seq-NMS와 FGFA는 서로 보완적인 관계로 보입니다.

<b>Comparison with state-of-the-art systems</b> 이미지에서의 object detection과 다르게 비디오에서의 object detection은 평가를 위한 메트릭과 가이드라인이 부족합니다. VID 2015, 2016에서의 우승자는 매우 인상적인 결과를 보이지만 이는 매우 복잡하게 엔지니어링 된 시스템이며 많은 bells and whistles을 사용합니다. 이것은 서로 다른 모델간의 공정한 비교를 어렵게 합니다.

이 논문은 가장 좋은 성능을 내는 시스템 대신 video object detection을 위한 근본적인 학습 프레임웤을 목표로 합니다. FGFG의 베이스라인 보다의 성능의 개선은 우리의 접근 방식의 효율적을 말해줍니다. 참고로 VID 2016의 승자는 81.2% 의 map을 얻었습니다. 이들은 ensembling, cascaded detection, context information, multiscale inference를 사용했고 이와 반대로 우리는 오직 Seq-NMS 만을 사용해 80.1% 의 map를 얻었습니다. 그래서 우리는 우리의 접근 방식이 매우 경쟁력이 있다고 주장합니다.

### 5. Conclusion and Future Work

이 논문은 video object detection을 위한 end-to-end이며 근본적인 학슴 프레임웤을 제공합니다. 우리의 접근이 좋은 feature에 집중 했기 때문에 이는 다른 box-level 프레임웤과 보완적입니다. 더 나은 연구를 위한 몇가지 중요한 것들이 남아있스빈다. 우리의 방법은 좀 느린 편이고 더 가벼운 flow network를 사용함으로 빠랄질 수 있습니다. [29] Youtube-boundingboxes 와 같은 fast object에 대한 더 많은 라벨을 통해 성능이 개선될 여지가 있으며 정확한 flow estimation은 성능의 향상을 보일 것입니다. 우리의 방법에 적용된 attention model 대신 다른 adaptive memory scheme 를 사용해 성능의 개선이 될 수 있습니다. 우리는 이러한 open question들이 미래의 연구에 영감을 줄 것이라고 믿습니다.
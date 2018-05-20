# Detect to Track and Track to Detect

## Abstact

최근의 비디오에서의 detection, tracking을 위한 높은 성능을 위한 모델들은 복잡하고 여러 단계의 솔루션으로 구성되어져 있습니다. 이 논문에서는 우리는 detection, tracking을 jontly 수행하는 CNN 아키텍쳐를 제안하고 tracking, detection을 간단하고 효율적인 방법으로 해결합니다. 우리의 공헌은 3가지입니다. (i) 프레임 마다의 object detection, 프레임 간의 tracking regression multi-task loss를 사용함으로써 tracking, detection을 동시에 수행하는 CNN 구조를 제안합니다. (ii) tracking을 위한 CNN에 도움을 주기 위해 시간에 걸쳐 나타나는 객체의 co-occurrence를 represent하는 correlation feature를 소개합니다. (iii) across-frame tracklets을 기반으로 frame level detection을 연결함으로써 video level에서의 높은 성능의 detection을 가능하게 합니다. spatio-temporal object detection을 위한 우리의 CNN 구조는 매우 큰 데이터셋인 VID에서 평가되었으며 최신의 성능을 달성합니다. 우리의 접근은 기존의 ImageNet 챌린지의 우승자보다 단일 모델성능이 뛰어나며 개념적으로 간단합니다. 마지막으로 temporal stride를 늘림으로써 tracker의 속도를 매우 향상시킬 수 있음을 보입니다. 우리의 코드와 모델은 공개되어져 있습니다.

## 1. Introduction

단일이미지에서의 Object detection은 지난 수년동안 많은 관심을 받았는데 주로 CNN, region based 방법들에 의해 엄청난 발전을 보였습니다. 비디오에서의 detection, tracking의 경우, 최근 접근들은 첫번째 단계에 detection을 사용하고, tracking을 적용하여 detection score를 전달하는 post-processing 방법들을 사용합니다. 이러한 'tracking by detection' 패러다임들은 발전해 왔지만 frame-level detection 방법들보단 성능이 떨어졌습니다.

최근에는 특히 Imagenet video object detection (VID) challenge가 도입된 이래로 비디오에서의 Object detection의 관심이 증가하고 있습니다. VID가 ImageNet Objcet Detection(DET) 와 다른점은, VID는 이미지 시퀀스안에 Object가 존재하며 몇가지 추가적인 문제점들이 있습니다. (i) size : 동영상이 제공하는 프레임 수가 월등히 많습니다. (VID 는 1.3M의 이미지를 가지고 있으며 이는 DET(400K), COCO(100K) 보다 월등히 많은 양입니다.) (ii) motion blur : object motion, rapid camera 때문에 motion blur 가 발생합니다. (iii) quality : VID는 인터넷에서 모아진 비디오 클립이며 이는 static photo보다 질이 떨어집니다. (iv) partial occlusion : object/viewer의 위치의 변화 때문에 occlusion이 발생합니다. (v) pose : 보편적이지 않은 object-to-camera pose가 비디오에서는 많이 관찰됩니다. 피규어 1, 은 VID 데이터셋에서의 몇몇 샘플을 보여줍니다. 

VID에서의 이러한 문제점을 해결하기 위해 최근의 VID 챌린지의 우승자들은 frame-level detector에 post-processing을 많이 사용합니다. 예를들어 2015 VID의 우승자는 Faster R-CN detection 프레임워크를 사용했고 context suppression, multi-scale traning/testing, optical-flow를 기반으로하는 tarcking을 사용한 score propagation, 앙상블 등을 사용합니다. 

이 논문에서는 우리는 실제 비디오에서의 object detection 문제를 해결하기 위한 통합된 접근방법을 제안합니다. 우리의 목표는 CNN을 통해 detection, tracking을 동시에 수행함으로써 여러 프레임에 대한 tracklet을 바로 추론하는 것입니다. 이를 달성하기 위해 우리는 R-FCN 을 tracking 을 위해 확장하며 이는 correlation, regression 를 기반으로 한 tracker에 영향을 받았습니다. [1, 13, 25] detection, tracking을 기반으로한 loss를 통해 fully convolutional network를 end-to-end로 학습시키고 jointly detection, tracking을 위한 D&T 방법을 사용합니다. 네트워크의 인풋은 여러 프레임이며 이는 Convnet trunk(resnet과 같은 cnn구조를 말하는것 같은데) 를 통과하여 feature를 추출하고 이는 detection, tracking task를 위해 공유됩니다. 인접한 프레임간의 feature 사이의 cross-correlation을 계산하며 이는 다른 feature scale에서의 local displacement(변화?) 를 추정합니다. feature의 끝에는 classfy, regression box를 위한 ROI-Pooling layer가 존재하면 프레임간의 box transformation을 regression하는 ROI-tracking layer가 존재합니다. 마지막으로 비디오에서의 long-therm tube를 생성하기 위해 우리의 tracklet을 기반으로 detection 결과들을 연결합니다. 

매우 큰 ImageNEt VID 데이터셋에대한 평가를 통해 우리의 접근 방법이 2016 챌린지의 우승자보다 단일 모델에서 좋은 결과를 달성한 것을 보였고 더 빠르고 개념적으로 간단합니다. 또한 우리는 tracking loss를 포함하는 것이 static object detection의 성능을 향상 시킬 수 있다는 것을 보였고 temporally-strided 인풋을 통해 빠른 D&T 버젼을 소개합니다.

## 2. Related Work

<b>Object detection</b> : 두가지 종류의 detector가 현재 가장 유명합니다. 첫번째로, region proposal을 기반으로 하는 R-CNN, Fast R-CNN, Faster R-CNN, R-FCN 이며 두번째는 이미지로 부터 box를 바로 예측하는 one-step 모델인 YOLO, SSD 입니다.

우리는 R-FCN을 기반으로 합니다. 이는 fully convolutional을 사용한 region proposal 기반의 방법중 간단하고 효율적인 방법입니다. 이는 per-region에 대한 연산을 수행하기 위해 multi-layer 네트워크를 사용하는 Faster R-CNN과 경쟁력 있는 성능을 내고 있습니다. (FC-layer를 사용하기 때문에 proposal ROI의 갯수에 선형적으로 연산이 늘어납니다). R-FCN은 position-sensitive ROI pooling을 도입하여 region classfication에 필요한 연산량을 줄입니다. position-sensitive ROI pooling 은 인풋 ROI의 spataially subsampled class score를 인코딩하게 됩니다.
 
 <b>Tracking</b> : tracking은 컴퓨터 비전에서 광범위하게 연구된 문제입니다. 최근의 발전은 CNN의 feature에서 동작하는 tracker에 집중되었습니다. [26] 에서는 CNN이 테스트 타임때도 fine-tuning되어 비디오 내에서의 대상을 bounding box regression, detection을 통해 추적합니다. test sequence에서의 샘플을 학습하는것은 매우 느리며 object detection 을 적용하지 못합니다. pre-trained CNN feature를 사용하여 tracking을 하는 다른 방법은 correlation or regression traker, bounding box 을 통해 좋은 성능을 달섭합니다. 이중 regression tracker[13] 이 우리의 방법과 관련되어져 있습니다. 이는 Siamese CNN을 기반으로 하며 이전 이미지의 중앙에 위치하는 객체의 두번째 이미지에 대한 위치를 예측합니다. 이 tracker는 position이 아닌 bounding box를 예측하며 이 때문에 tracked templated의 scale, aspect의 변화를 모델링 할 수 있습니다. 이 접근방법의 단점은 오직 하나의 single template에만 수행할 수 있다는 것이며 tracked box에 대한 모든 가능한 transformation을 학습하기 위해 매우 많은 data augmentation이 필요로 하게 됩니다. [1] 은 correlation을 사용한 tracker이며 우리의 방법에 영감을 줬습니다. [1] 의 tracker또한 tracking template를 인풋으로 받고 이미지를 search하는 fully convolutional Siamese 네트워크를 사용합니다. CNN 의 마지막 레이어의 feature는 target position을 찾는 것과 상관관계가 있습니다. 이러한 correlation tracker의 단점은 하나의 target에만 동작하며 object의 scale, aspect ratio 의 변화를 고려하지 않습니다.
 
 <b> Video object detection</b> : Action detection 또한 관련된 task이며 최근들어 많은 관심을 받고 있습니다. 대부분은 two-stream CNN을 통한 방법을 많이 사용합니다. [11] 에서는 two-stream R-CNN을 사용하여 region을 판별하고 action predcition, spatial overlap 을 기반으로 프레임사이의 detection 결과들을 연결합니다. 이러한 방법은 [33,27] 에서도 적용되었으며 R-CNN을 Faster R-CNN 으로 바꿔 적용합니다. 
 
 또 하나의 재미있는 분야는 weak supervision을 통한 detection을 학습하는 것입니다. Youtube object dataset은 이러한 목적으로 구성되어졌습니다.
 
 VID task는 ImageNet 챌린지에 소개된 이후 많은 관심을 받아왔습니다. [18 (T-CNN)] 에서는 각 frame-based box proposal에 tracking을 적용하여 tubelet proposal을 수행합니다. 프레임 간의 detection scoore는 1D CNN에 의해 re-scoring됩니다. 같은 그룹에서 낸 [17] 은 [18] 에 프레임 간의 optical flow를 활용하여 인접한 프레임에 score를 전파시키고 비디오 클립 안에서 잘 나타나지 않는 class의 score는 억제시킵니다. 좀 더 최근의 연구인 [16] 은 tubelet proposal network를 제안하고 여러 프레임에서의 object proposal을 regression 합니다. Faster R-CNN을 통해 뽑혀진 feature를 encoder-decoder LSTM 구조를 통해 tubelet score를 구하게 됩니다. [40 (DFF)] 에서는 CNN을 key frame에 대해서만 feature를 뽑고 optical flow CNN을 통해 optical flow를 추정하고 이를 이용하여 나머지 프레임에 대한 feature를 warping을 통해 구하게 됩니다. 이 접근 방법은 약간의 성능 하락이지만 5배 빠른 detection speed를 가질 수 있게 합니다. 2가지 원인이 에러를 생기게 하는데 첫째로 key-frame의 feature를 현재 프레임으로 warping하는데서 에러가 생길 수 있으며 둘째로, key frame은 현재 frame에 대한 정보가 부족할 수 있습니다. 매우 최근에는 매우 큰 video object detection 데이터셋에 소개되었으며 비디오 sequece마다 하나의 object annotation이 존재하는 데이터셋입니다.
 
 ## 3. D&T Approach
 
 이 섹션에서는 D&T 접근방법에 대해 처음으로 소개합니다. D&T는 둘 혹은 이상의 프레임을 인풋으로 받아 tracklets를 생성합니다. 더 자세한 것을 소개하기 위해 베이스라인 R-FCN detector 부터 시작하며 프레임간의 bounding box regression tracking objective function을 수식화합니다. 마지막으로 네트워크의 tracking process를 도와주는 correlation feature를 소개합니다.
 
 섹션 4는 프레임간의 tracklets을 어떻게 연결할지에 대해 설명하며 섹션 5는 D&T를 어떻게 VID 챌린지에 적용했는지를 설명합니다.
 
 ## 3.1. D&T overview We
 
 우리는 비디오에서의 object detection, tracking을 동시에 하는 것을 목표로 합니다. 피규어2는 우리의 D&T 구조를 보여줍니다. region classfication, regreesion을 위해 fully convolutional 을 사용하는 R-FCN을 구축하고 이를 멀티 프레임 detection, tracking으로 확장합니다. 2프레임의 높은 해상도의 인풋이 주어지면 먼저 convolutional feature map을 계산하고 이는 tracking, detection에서 공유됩니다. RPN이 각 프레임에서의 candidate region을 생성하고, 이 region을 기반으로 ROI-pooling이 position-sensitive score, regression map을 추출합니다. 
 
 우리는 이 구조에 추가적인 regressor를 도입하여 확장합니다. 이 regressor는 중간의 position sensitive regression map(concat을 통해 두 프레임에서의 regression map을 합친다. correlation map도 함께) 을 두 프레임으로 부터 받아서 ROI tracking의 인풋으로 받습니다. ROI tracking 연산은 한 프레임에서 다른 프레임으로의 box transformation을 출력합니다. 두 프레임 간의 일치를 위해 두 프레임에서 같은 proposal region에 대해 pooling을 수행함으로써 사이즈를 맞춥니다. 우리는 ROI-tracking task를 R-FCN의 Multi-task loss를 확장하여 학습합니다. 이때 프레임간의 objcet coordinate를 regression 하는 loss를 사용합니다. tracking loss는 ground truth object사이의 좌표 차이를 이용하며 soft L1 norm 을 사용합니다. 
 
 이러한 tracking 수식은 일종의 single target tracker의 multi-object 로의 확장으로 볼 수 있습니다. [13] 에서의 single target tracker는 두 프레임에서의 feature를 통해 object 의 bounding box를 예측하도록 학습되어졌습니다. 이러한 접근 방법의 단점은 명시적으로 translational equivariance하지 않다는 것입니다. 이 의미는 tracker가 트레이닝 데이터로 부터 모든 가능한 translation을 학습해야 합니다. 이 때문에 [13] 에서의 tracker는 특히나 많은 data augmentation을 필요로 합니다. (cnn은 transition equivariance하지 않음.. 그래서 data augment등등을 하고 막 그렇게 한다. 이러한 단점이 있다는 듯 하다.)
 
 correlation filter를 기반으로 하는 tracking 은 correlation filter가 tranlation equivarinat하기 때문에 명시적으로 translational equivariance 합니다. 최근의 correlation traker는 high-level CNN feature를 사용하며 tracking template와 search image간의 cross correlation 을 계산합니다. (혹은 이전 프레임에서 target이 위치 했던 주변만을 계산, 연산량을 줄일기 위해서!!) <b>이러한 correlation map의 결과는 일종의 template와 search image사이의 similarity를 나타내며 수직, 수평축에 대한 모든 circular shift를 반영 할 수 있습니다.</b> (왜지???) search image에서의 target의 위치는 correlation map이 최대가 되는 지점으로 찾을 수 있습니다.
 
 single target template에만 사용가능한 전통적인 correlation tracker와 달리 우리는 여러개의 객체에 대한 동시 tracking을 목표로 합니다. feature map에서의 모든 가능한 position에 대해 correlation map을 계산하고 계산된 feature를 사용하여 ROI-tracking이 더 나은 track regression을 수행하도록 합니다. 우리의 구조는 비디오 인풋으로 부터 end-to-end 학습이 가능하며 object detection 결과와 그들의 track을 출력합니다. 다음 섹션은 object detection, tracklet을 위한 네트워크 구조를 설명합니다.
 
 ## 3.2. Object detection and tracking in R-FCN
 
 우리의 구조는 t 시간에서 의 프레임을 인풋으로 받고 이를 CNN을 통해 feature map을 얻습니다. R-FCN 에서 기술 된 것 처럼 우리는 마지막 layer의 스트라이드를 16으로 줄이고 이를 보정하기 위해 dilated convolution을 사용합니다.
 
 R-FCN을 사용한 전체의 시스텡은 2단계로 이루어집니다. 첫째로 RPN을 사용하여 cadidate ROI를 뽑습니다. 두번째로 position-sensitive ROI pooling layer를 통해 region classfication, regression을 수행합니다. ROI pooling의 인풋은 추가적인 convolutional layer를 사용하여 구해진 x_cls를 인풋으로 받습니다. 이 레이어는 일종의 bank position-sensitive score map을 출력하며 이는 각 클래스 C 마다의 kxk spatial grid에 해당하는 상대적인 position을 인코딩 합니다. softmax 를 적용함으로써 각 ROI에 대한 C+1 카테고리에 대한 확률분포를 얻을 수 잇습니다. R-FCN의 두번째 가지 또한 추가적인 convolutional layer를 거친 feature를 사용하여 이는 x_reg라고 정의합니다. 이 또한 마찬가지로 position-sensitive ROI pooling을 거치며 이는 class-agnostic bounding box regression을 수행하게 됩니다.
 
 이제 두 쌍의 프레임을 인풋으로 받는다고 생각해 봅시다. 우리는 일종의 inter-frame bounding box regression layer를 도입하였고 이는 두 프레임에서 얻어진 x_reg feature를 concate 한 것에 position-sensitive ROI pooling을 수행함으로써 얻어집니다. 이는 두 프레임간의 ROI 간의 transformation을 수행하게 됩니다. correlation feature가 tracking을 위한 bounding box regressor에 사용되었고 이는 섹션 3.4에 기술할 것입니다.
 
 ## 3.3. Multitask detection and tracking objective To
 
 tracking regressor를 학습하기 위해 우리는 Fast R-CNN의 multi-task loss를 확장합니다. 원래의 multi-task loss는 classfication loss, regression loss의 합으로 이루어져 있으며 여기에 두 프레임 사이의 score에 대한 loss를 추가했습니다. N개의 ROI에 대한 한번의 iteration마다 네트워크는 확률분포 p_i, regression offset b_i, cross-frame ROI-track 을 출력합니다. 전체의 objective function은 다음과 같습니다.
 
 수식 생략-- 별거없다
 
 ground truth ROI의 할당은 다음과 같습니다. ROI 와 gt box와의 IOU가 0.5 이상일 경우 class label, regression label을 할당합니다. tracking target의 경우 두 프레임 모두 같은 object가 나타날 경우만 할당합니다. 그래서 loss function의 classfication loss는 N개의 box에 대해 학습되며 regression function은 오직 foreground ROI의 갯수만큼 학습됩니다. tracking loss는 두 프레임사이의 track이 존재 할 경우만 학습됩니다.
 
 tracking regression을 위해서 우리는 R-CNN에서 사용된 parameterized box regression을 사용합니다. 하나의 객체에 대한 gt box를 알고 있을때 그 수평,수직 중앙좌표, width, height를 regression gt로 사용합니다. parameterized box coordinate는 다음과 같습니다.
 
 수식 생략-- R-CNN과 동일하다
 
 ## 3.4. Correlation features for object tracking Different
 
전통적인 correlation tracker가 single target template를 사용하는 것과 다르게 우리는 여러개의 object를 동시에 tracking해야 합니다. 우리는 feature map의 모든 location에서의 correlation map을 계산하고 ROI pooling 이 correlation feature를 tracking regression에 사용하도록 하였습니다. feature map에서의 모든 가능한 circular shift를 고려하는 것은 출력의 차원이 너무 커지게 produce response for too large displacement(? 뭔소리일까.) 이 때문에 우리는 local neighborhood 에만 correlation을 제한시켰습니다. 이 아이디어는 optical flow 추정에 사용되는 방법으로써 [5] 에서는 프레임간의 feature point를 매칭시키기 위해 도입된 correlation layer에서 사용되었습니다. correlation layer는 두 프레임의 feature map x_t, x_t+b 를 point-wise 하게 비교함으로써 수행됩니다.

  <img src="https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D_%7Bcorr%7D%5E%7Bt%2C%20t&plus;%7B%5Ctau%7D%7D%20%3D%20%3C%20x_l%5Et%28i%2Cj%29%2C%20x_l%5E%7Bt&plus;%5Ctau%7D%28i&plus;p%2C%20j&plus;q%29%3E%20%5C%2C%5C%2C%5C%2C%5C%2C%284%29"/>


여기서 -d <= p <= d , -d <= q <= d는 i,j를 중심으로 하는 square neighbourhood 만을 비교하기 위한 offset 입니다. offset의 범위 d는 maximum displacemnet를 정의하게 됩니다. (d를 크게 잡으면.. 매칭되는 크기가 커진다는 느낌인데?? 불확실하다) 그래서 correlation layer의 feature map 크기는 HxWx(2d+1)x(2d+1) 이 되게 됩니다. (한 i,j 쌍의 비교마다 (d--d+1) 개의 값이 나온다는 느낌일까?) 수식 (4) 는 d에 의해 정의된 local square window 에서의 두 feature map 간의 correlation으로 볼 수 있습니다. 우리는 이러한 local correlation을 conv3,4,5 에 수행하였습니다. (사이즈를 맞추기 위해 conv3에서의 i,j는 스트라이드를 2를 사용합니다.) 피규어 4에는 두 샘플간의 correlation feature map을 보여줍니다.
 
 이러한 feature를 track-regression에 이용하기 위해 두 프레임에서 뽑힌 reggression feature, correlation feature 총 3개를 stack 하여 ROI-Pooling에 사용되도록 했습니다.
 
## 4. Linking tracklets to object tubes

현재의 최신의 object detection의 단점중 하나는 high-resolution 인풋이 필요 하다는 것입니다. 이러한 제한은 GPU 하드웨어의 메모리 제약 때문에 한번의 iteration에 적은 프레임을 사용할 수 밖에 없게 합니다. 그러므로 detection accuray와 프레임 갯수사이의 트레이드 오프가 있을 수 밖에 없습니다. 비디오는 redundant inforamtion을 많이 가지고 있으며 object는 보편적으로 시간축에서 smooth하게 움직입니다. 우리가 제한한 inter-frame track을 통해 detection 결과를 연결할 수 있으며 long-term object tube를 만들 수 있습니다. 이를 위해서 action localization에서 사용되는 방법을 적용하였고 이는 시간축에 잇는 detection 결과를 tube로 연결해 줍니다.

t 시간의 프레임에 대한 class detection을 고려해 봅시다. 

  <img src="https://latex.codecogs.com/gif.latex?D_i%5E%7Bt%2Cc%7D%20%3D%20%5Cbegin%7BBmatrix%7D%20x_i%5Et%2C%20y_i%5Et%2C%20w_i%5Et%2C%20h_i%5Et%2C%20p_%7Bi%2Cc%7D%5Et%20%5Cend%7BBmatrix%7D"/>

D_i^{t,c} 는 i번째 (보통 300개의 ROI가 있겠지) ROI를 나타내며, 이 센터는 x,y이며 width, height가 w,h 클래스 c에 대한 확률이 p_{i,c}^t 인 ROI를 나타냅니다. 비슷하게 track에 대해서도 정의할 수 있으며 이는 프레임 t 에서 다른 프레임으로의 box transformation을 설명합니다. 이제 시간축으로 detection, tracking을 결합 할 수 있는 class-wise linking score를 정의합니다.

  <img src="https://latex.codecogs.com/gif.latex?s_c%28D_%7Bi%2Cc%7D%5Et%2C%20D_%7Bj%2Cc%7D%5E%7Bt&plus;%5Ctau%7D%2C%20T%5E%7Bt%2C%20t&plus;%5Ctau%7D%29%20%3D%20p_%7Bi%2Cc%7D%5Et%20&plus;%20p_%7Bj%2Cc%7D%5E%7Bt&plus;%5Ctau%7D%20&plus;%20%5Cpsi%20%28D_i%5Et%2C%20D_j%2C%20T%5E%7Bt%2C%20t&plus;%5Ctau%7D%29%20%5C%2C%5C%2C%5C%2C%5C%285%29"/>

즉.. tracking이랑 IOU가 0.5이상인 detection box들 중에서... 둘다 classfication score가 높은 애들끼리 연결하겠다는 뜻이네.

  <img src="https://latex.codecogs.com/gif.latex?%5Cpsi%20%28D_%7Bi%2Cc%7D%5Et%2C%20D_%7Bj%2Cc%7D%5E%7Bt&plus;%5Ctau%7D%2C%20T%5E%7Bt%2C%20t&plus;%5Ctau%7D%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%2C%20%5C%2C%5C%2C%20if%20D_i%5Et%2C%20D_j%5E%7Bt&plus;%5Ctau%7D%20%5Cin%20T%5E%7Bt%2C%20t&plus;%5Ctau%7D%20%5C%5C%200%2C%5C%2C%5C%2C%20otherwise%20%5Cend%7Bmatrix%7D%5Cright.%5C%2C%5C%2C%5C%2C%5C%2C%286%29"/>

 
여기서 pairwise term psy는 tracking 결과와 IOU가 0.5 이상일때 1이며 나머지는 0이 됩니다. <b> 이는 필수적인데 track regressor의 결과가 detecion에서의 box regressor와 정확하게 매치하는 경우가 별로 없기 때문입니다.</b> 

비디오전체 에서의 optimal path를 찾기 위해서는 T duration을 가지는 비디오에서 score를 maximizing하는 것이 필요합니다.

  <img src="https://latex.codecogs.com/gif.latex?%5Cbar%7BD%7D_c%5E*%20%3D%20argmax_%7B%5Cbar%7BD%7D%7D%5Cfrac%7B1%7D%7B%5CGamma%20%7D%5Csum_%7Bt%3D1%7D%5E%7B%5CGamma%20-%5Ctau%7Ds_c%28D%5Et%2C%20D%5E%7Bt&plus;%5Ctau%7D%2C%20T%5E%7Bt%2C%20t&plus;%5Ctau%7D%29%20%5C%2C%5C%2C%5C%2C%5C%2C%287%29"/>
  
  수식 (7) 은 비터비 알고리즘을 적용해서 효율적으로 풀 수 있습니다. optimal tube 를 찾게되면 tube를 만드는데 사용된 detection들은 제거되고 나머지 region에 수식(7)을 다시 적용하게 됩니다.
  
  이렇게 class-specific tube 를 하나의 비디오에서 찾고 나면 tube안의 모든 detection score들을 다시 scoring하게 되는데 tube안에서 가장 높은 confidence를 가지는 score의 mean의 50%(하이퍼파라미터 알파) 더해주게됩니다. (mean을 더해준다고?? score는 scalar로 나오는거 아닌가.. 뭔소리인지를 모르겠다..아 score 정렬했을때.. 50% 까지의 box들의 score를 mean 해준다는 소리인거 같음) 우리는 이 파라미터 알파가 전체의 성능에 별로 영향이 없다는걸 찾았고 알파를 10%~100% 까지 변화시켰을때 0.5% 정도의 map의 변화가 있었습니다. 우리의 간단한 tube-based re-weighting은 detection이 fail한 positive box의 score를 촉진하기 위해서입니다.(tf를 증가시키기 위해서.) tube에서의 highest score를 이용하여 re-scoring 하는 것은 일종의 nms 처럼 작동합니다. <b>이는 Canny edge detector에서의 hysteresis tracking에 영감을 받았습니다.( 이게 뭐지 한번 찾아보자)</b> 우리의 re-scoring은 detector가 tube에서 절반정도 실패한다고 가정하고 알파의 선택이 별로 성능에 영향을 끼치지 않는다고 가정하지만 tracker의 robustness를 올려줍니다. 주의 할것은 우리의 접근법은 tube를 비디오 전체로 확장하지만, 간단함을 위해 시간마다 detection 결과를 제거하지 않습니다. score가 낮은 detection들을 tube를 따라 계속 제거하면 더 좋은 결과를 얻을수 있지만 우리는 future work를 위해 이들을 남겨두었습니다.(무슨 futre work일까.?) 다음의 섹션에서는 우리의 접근법을 object detection task에 적용합니다.
   
## 5. Experiments 5.1.

## 5.1. Dataset sampling and evaluation We

우리의 방법을 ImageNet VID 데이터셋에 대해 평가했습니다. 이는 30클래스와 3862 비디오, 555개의 validation video로 구성되어져 있습니다. 각 객체는 gt box annotation과 track id가 주어져 있습니다. test set에 대한 gt가 주어져 있지 않음으로 우리는 다른사람들이 하는 것과 같이 30개의 클래스에 대해 validation set에서의 map를 측정합니다.

VID 에서의 30개의 카테고리는 DET 의 200개의 부분집합입니다. 이전의 접근방법에 따라 우리는 R-FCN detector를 VID+DET 에서 학습시킵니다. DET 데이터셋에서 클래스당 샘플의 갯수가 차이가 많이 나기 때문에 DET 에서 클래스당 2k 이미지를 샘플링합니다. 또한 VID 에서 비디오당 10프레임만을 샘플링 합니다. 이러한 sub sampling은 DET 에서 많이 나타나는 클래스의 영향을 줄일수 있습니다. (DET 에서 dog 클래스의 이미지만 56k 입니다. 또한 VID 에서 너무 긴 비디오에 대한 영향을 줄입니다)
 
## 5.2. Training and testing RPN.

우리의 RPN은 [31] 에서 제안된 방법으로 학습됩니다. Proposal classfication, regression을 수행하기 위해 2개의 conv layer를 붙였으며 5 scale, 3 aspect 의 15개의 anchor를 사용합니다. [31] 에서 처럼 5 스케일에 대한 proposal을 추출한뒤 nms를 적용하며 IOU는 0.7로 합니다. 총 300개의 proposal을 뽑아 학습이 진행됩니다. full ImageNet DET 에서 pre-training 되는 것이 recall을 증가시키는 것을 관찰했고 RPN은 DET 200클래스에 대해 먼저 학습되고 그 이후 30 클래스 VID+DET 에 대해 fine-tuning 됩니다. 이미지당 300 proposal의 mean recall은 96.5% 를 달성했습니다.

<b>R-FCN</b> R-FCN은 [3,40] 과 비슷하게 학습됩니다. 우리는 stried-reduce resnet 101을 사용하고 conv5에는 dilated conv과 OHEM을 통해 학습합니다. [40(DFF)] 에 따라 feature dimension을 512로 줄이기 위해 3x3 dilation 6, conv layer를 맨 끝에 붙입니다. detection, regression을 위해 두개의 1x1 conv layer를 붙이고 이는 psROI-pooling의 인풋으로 들어갑니다. spatial grid는 7x7을 사용합니다.
 
 테스트와 학습 모두 작은 면이 600 pixel인 single scale 이미지를 사용했습니다.(응?? 기존에 하던 짓이랑 다른데? 그냥 고정해버렸다는 건가?), SGD를 사용하며 배치사이즈는 4, lr은 60k까진 0.0001 그 뒤의 20k는 0.00001을 사용합니다. test 때의 NMS는 0.3을 사용합니다.
  
<b>B&T</b> D&T 구조를 학습하기 위해서 우리는 위에서 언급한 R-FCN 모델에서 시작합니다. 그 후 full VID training set에서 fine-tune 되며 각 iteration마다 비디오에서의 인접한 2개의 프레임을 샘플링 합니다. 또한 DET training set 에서도 샘플리을 하는데 이는 VID training set에 대해 바이어스를 피하기 위해서입니다. DET에서 샘플링을 할때는 <b>두 프레임을 같은 것을 보내서 sequecne가 없도록 합니다. 이는 DET training set에서 학습된걸 잊지 않게 하고 small motion에 더 민감하도록 하는 효과가 있습니다.</b>([13] 에서 tracker motion augmentation을 위해 라프라시안 분포를 사용하는데 zero mean 을 사용한다. 이는 regression tracker가 small displacement에 편향되도록 만든다고 함). 우리의 correlation feature는 conv3,4,5 에서 계산되며 maximum displacemnet 8, 이며 conv3에서는 i,j의 stride를 2로 합니다. lr은 0.0001 40k, 0.000001, 20k 이며 배치사이즈는 4입니다. 테스트 동안 우리의 구조는 temporal stride tau 만큼의 sequecne에 적용되어 detection과 tracklets을 예측합니다. object가 중앙에 있는 tracking을 하기 위해 우리는 ROI-tracking layer에 regressed frame box를 인풋으로 넣습니다. 우리는 nms와 bounding box voting을 tracklet linking 전에 수행하며 이미지당 같은 클래스의 ROI를 최대 25개로 합니다. 최종 detection은 수식(7) 에 이용되어 tube를 추출하고 tube안의 box들은 re-scoring 됩니다. 

## 5.3. Results

우리의 모델과 sota 모델에 대한 실험을 진행했습니다. 다양한 비디오에 대한 질적 결과는 피규어 5에 있습니다.

<b>Frame level methods.</b> 먼저 우리는 어떠한 temporal processing이 없는 단일 프레임 모델에 대해 비교합니다. 우리의 R-FCN 베이스라인은 74.2% MAP을 달성하였고 이는 [DFF] 에서의 73.9%와 비교할만 합니다. 좀더 높은 MAP을 얻은 이유를 9 anchor가 아닌 15 anchor를 사용했기 때문이라고 생각합니다. [18,16,17] 에서의 Faster R-CNN single frame 모델은 각각 45.3%, 63.0%, 63.9% 의 성능을 보였습니다. 이들이 낮은 성능을 보이는 것은 다른 학습 방법과 데이터 샘플링 때문이라고 생각하며 더 안좋은 CNN을 썻기 때문이 아니라고 생각합니다. 우리는 Resnet-50 을 백본으로한 베이스라인으로 72.1% 이라는 성능을 얻었습니다. 다음으로 우리는 tracking loss를 포함해 fine-tuning을 한 뒤 이를 단일 프레임에 적용했습니다. 테스트 결과 75.8% map를 얻엇고 이러한 1.6%의 성능 향상은 tracking loss를 추가함으로써 per-frame detection에 도움을 줬기 때문일 수 있습니다.(tracking loss를 통해 feture에 시간적 정보가 섞였기 때문이 아닐까?) 하나의 가능성은 correlation feature가 gradient를 backprop하는 과정에서 더 중요한 object에 민감해 졌을 수 잇습니다. 우리는 특히나 움직임이 많은 panda, monkey, rabbit, snake에서의 큰 향상을 얻었습니다.

<b> Video level method</b> 다음은 여러개의 프레임을 테스트 동안 받는 경우입니다. 테이블 1에서 tracklet을 기반으로하여 생성된 tube의 성능을 볼 수 있습니다. D&T 는 79.8% 라는 MAP를 보였고 몇몇의 클래스의 AP는 특히나 많이 향상됩니다. 이러한 향상의 이유는 다음과 같습니다. 만약 비디오안의 object가 motion blur, small scale 등등 이 있다면 detection이 실패할 수 있습니다. 하지만 tube가 연결됨으로써 같은 객체인 detection scoring이 올라감으로써 질이 나쁜 프레임에서의 detection의 실패를 복구 할 수 있습니다. (심지어 매우 간단한 re-scoring을 사용했음에도!!!) AP가 떨어진 class는 whale만이 -2.6 만큼 떨어졌으며 이는 설명가능합니다. validation set에서의 대부분의 whale 클래스는 물위를 계속 떠올랐다 가라않는것을 반복합니다. 우리의 tube에 기반한 detection rescoring은 프레임안에서 whale이 가라앉을때 false positive box를 예측하게 됩니다.

현재의 sota 모델과 우리의 79.8% map을 비교했을때 다음과 같은 관찰을 할 수 있습니다. [18] 은 still image detector의 맨 뒷단에 temporal convolution을 통해 47.5%의 map을 달성합니다. 이를 확장한 [16] 은 encoder-decoder LSTM과 Faster R-CNN을 사용하는 tubelet proposal network를 제안하고 이는 68.4% map를 달성합니다. 2015 VID task의 승자는 두개의 Faster R-CNN기반 모델과 multil-scale training/testing, context suppression, high confidence tracking, optical flow guided propagation을 통해 73.8% map을 달성합니다. 2016 VID의 우승자는 cascaded R-FCN,context inference, cascacde regresion, correlation tracker를 통해 76.19% 을 단일 모델로 달성합니다. (multi-scale testing, model 앙상블을 통해 81.1% 을 달성합니다.)

<b>Online capabilities and runtime. </b> 우리의 모델을 online으로 적용하기 위해서 제한되는 부분은 tube rescoring입니다. causal re-scoring을 사용한 online-version에 대해서 평가하였고 이 방법은 78.7% 을 달성합니다. correlation layer, track regression이 fully convolutional 로 연산을 수행함으로 ROI-tracking을 제외한 추가적인 per ROI 연산이 없게 됩니다. 1000x600 해상도의 이미지에 필요한 추가적인 연산은 14ms 정도입니다(하나의 Titan X GPU). tube linking 에 소요되는 시간은 프레임당 평균 46ms정도입니다.

<b>Temporally strided testing. </b> 마지막 실험으로 우리는 testing 동안 큰 temporal stride 를 적용합니다. 이는 [6,7] 과 같은 video action recognition task에서 유용하게 사용됩니다. D&T 구조는 일정한 temporal stride 프레임 만큼만 적용되며 tracklet을 통해 stride만큼 link를 수행합니다. temporal stride 10인 경우 1인 경웨 비해 1.2% 낮은 78.6% map를 달성했으며 이는 전체 소요시간은 10으로 줄이지만 약간의 성능하락을 가져와 주목할만하다고 생각합니다.

더 개선해야할 점은 2개의 프레임이 아닌 연속적인 프레임에 대해 연산을 수행하는 것입니다. 작은 tmeporal stride를 가지는 연속적인 프레임에 대한 모델의 확장은 성능 향상에 큰 이점을 주지 못하는 것을 관찰 했습니다. (t 시간의 detection output을 augmenting 하는 것?? 은 80% map을 달성했다.. 뭔소리지??) temporal window를 t+-1 로 늘리고 bidirectional detection을 수행하고, t frame에서 tracking을 수행하는 것은 효과가 없었습니다. temporal stride 10으로 testing하고, tracked proposal을 통해 t 프레임의 detection을 augment하는 것은 78.6% 에서 79.2%로의 성능 향상을 보였습니다. 

short temporal window 일때의 accracy가 별로 별하지 않는 것은 tracking 내에서의 중앙에 있는 프레임의 detection score의 high redundacy 때문이라고 생각합니다. 그러나 큰 temporal stride로 부터 얻을 수 있는 이득은 tracked object 로 부터 보완적인 정보가 통합됨을 보여줍니다. 따라서 개선을 위한 방향은 multiple temporal stride input을 통해 detection, tracking을 수행하는 것입니다.

## 6. Conclusion We

우리는 비디오에서의 detection, tracking을 동시에 수행하는 통합된 프레임워크를 제안합니다. 우리의 fully convolutional D&T 구조는 Joint loss를 통해 end-to-end로 학습됩니다. 평가과정에서 우리는 지난 VID 챌린지의 우승자들과 경쟁력 있는 성능을 얻었으며 더 간단하고 효율적입니다. detection, tracking을 jointly 수행함으로써 얻을 수 있는 이득을 보였고 이는 비디오 분석에 대한 미래의 연구를 촉진 할 수 있습니다.






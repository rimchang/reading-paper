# R-C3D: Region Convolutional 3D Network for Temporal Activity Detection

### Abstract

우리는 continous, untrimed 비디오 스트림에서의 activity detection 문제를 다룹니다. 이는 각 activity의 시작과 끝을 정확하게 localizing하고 의미 있는 spatio-temporal featrue를 포착해야해 매우 어려운 테스크입니다. 우리는 새로운 R-C3D 라는 모델을 제안 하고 이는 3차원 conv network를 이용하며 activity를 포함하는 temporal region을 제안하고 후보 region들을 특정 acitivity로 분류합니다. region proposal, classfication이 같은 feature를 공유하므로 계산량이 절감되며 모델은 classfication, localization 의 multi-task 로 jointly end-to-end 로 학습됩니다. R-C3D 는 기존의 다른 모델보다 빠르며 THUMOS'14 에서 최고의 성능을 달성했습니다. 우리의 모델이 특정 데이터셋에 잘 작동하는것이 아닌 일반적인 acitivity detection 모델인 것을 보여줍니다. 이를 위해 AcitivityNet, Charades 데이터에 대해 실험을 진행하였습니다.

### 1. Introduction

continous 비디오에서의 activity detection은 recognizing 뿐만 아니라 localizing이 필요해 매우 어려운 문제입니다. 기존의 최고의 성능을 보이는 접근들은 이를 classfication 에 의한 detection으로 해결하였습니다. i.e [13, 20, 24, 37] 은 sliding window 방법을 이용하여 temporal segments를 판별합니다. [10, 35] 는 외부의 proposal 알고리즘을 사용합니다. 이러한 접근방법은 하나 혹은 이상의 단점이 있습니다. 이들은 end-to-end 방법으로 deep representation을 배우지 않고 hand-crafted feature나 VGG, Resnet, C3D 의 pretrained feature를 사용합니다. 이러한 representation은 다양한 비디오 도메인에서의 localizing activity를 위해서 최적이 아닐 수 있으며 낮은 성능을 보입니다. 또한 현재의 방법은 외부의 proposal 알고리즘이나 sliding window 방법을 사용하기 때문에 계산 효율이 떨어집니다. 마지막으로 sliding-windw 모델은 유연한 activit boundary를 쉽게 예측하지 못합니다.

##### 과거의 연구의 단점

1. feature를 고정해 놓고 classfication 만을 학습한다 이는 비디오 도메인에서 optimal이 아닐 수 있다.
2. 외부의 proposal, sliding window 방법은 계산 효율이 매우 떨어진다.
3. sliding-window 모델은 유연한 activity boundary를 예측하기 어렵다.

이 논문에서는 위의 모든 이슈를 다루기 위한 activity detection 모델을 제안합니다. 우리의 R-C3D 모델은 end-to-end 로 학습이 가능하며 classfication, localizing을 jointly 학습하여 테스크에 의존적인 feature를 학습이 가능합니다. Faster-RCNN 의 접근방법에 영감을 받아 우리는 fully-convolutional 3D convnet feature를 계산하고 activity를 포함하는 temporal region을 proposal 합니다. 그 후 activity class를 예측하기 위해 3d region을 roi-pooling 합니다. region proposal 단계는 많은 background 를 필터링 하여 sliding window 방법과 비교하여 매우 효율적인 계산 효율을 제공합니다. 또한 사전 정의된 anchor segment를 사용하여 proposal을 예측하므로 flexible activity boundary를 예측 할 수 있습니다.

end-to-end로 학습된 CNN의 feature는 activity recognition을 위해 성공적으로 사용되었고 특히 C3D 같은 3D Conv가 성공적 이였습니다. CD conv는 spatio-temporal feature를 학습 할 수 있습니다. 그러나 전통적인 16 프레임의 chunk를 인풋으로 받은 3D conv 와 달리 우리의 방법은 fully convolutional 을 사용하여 gpu 메모리가 허용하는 한의 프레임을 인풋으로 받을 수 있습니다. 그래서 풍부한 spatio-temporal feature를 배울 수 있습니다. 이런 feature map은 proposal, classfication을 위한 하위 네트워크에 공유되어 계산시간을 절약하고 jointly 하게 optimzie 하게 됩니다.

[4, 17, 18, 29, 39] 가 frame의 연속 혹은 비디오 chunk feature(VGG나 C3D에 의해 추출된) 를 다루기 위해 RNN을 사용하며 각 time step에서의 acitivity label을 예측합니다. 그러나 이러한 RNN을 기반으로 하는 방법들은 특정 길이의 temporal feature만을 다룰 수 있습니다. (C3D에서는 16 프레임 VGG 에서는 1프레임 feature). variable length proposal을 하나의 classfication 네트워크로 다룰 수 있도록 하기 위해 우리는 2D ROI pooling을 3D로 확장하며 이는 proposal로 부터 고정된 길이의 feature를 추출합니다. 따라서 우리의 모델은 어떠한 temporal granularity에서의 비디오 feature를 사용가능 합니다. 또한 RNN를 기반으로하는 detector는 regression을 통해 바로 temporal boundary를 예측합니다. [7, 31]과 같은 object detetion, [2] 와 같은 semantic segmentation 에서 볼 수 있듯이 regression-only 를 통한 방법은 proposal-based detection보다 낮은 성능을 보입니다.

우리는 THUMOS14, ActivityNet, Charades의 공개된 밴치마크 데이터셋을 사용하여 R-C3D와 최신의 activity detection 방법들을 비교합니다. THUMOS14, Charades 데이터 셋에서 최고의 성능을 달성했으며 ActivityNet 에서는 C3D feature를 사용하느 경우에서 최고의 성능을 달성했습니다. 우리의 공헌을 요약하자면 다음과 같습니다.

- arbitary length activities를 탐지할 수 있는 end-to-end activity detection 모델입니다.
- proposal, classfication sub-network가 fully-convolutional C3D feature를 공유하여 높은 계산 효율을 가집니다 (현재 방법보다 5배 더 빠릅니다)
- 세가지 데이터셋에 실험을 통해 우리의 모델의 일반적 적용 가능성을 보여줍니다.

### Related Work

<b>Activity Detection</b> : activity recognition과 trimmed 비디오 클립에서 카테고리를 예측하는 것은 매우 긴 역사를 가지고 있습니다. 하지만 activity detection은 untrimmed, long 비디오에서의 activity의 시작과 끝 시간을 예측하는 것이 필요합니다. 기존의 activity detection의 접근방법은 sliding window를 통해 sgement를 생성하고 각 segment를 calssfy 합니다. 이러한 방법들은 end-to-end로 학습되지 않고 stagewise 파이프라인을 가집니다. 또한 sliding window 방법은 계산적으로 비효율적이고 activity의 boundary를 제한 합니다.

최근의 몇몇 방법들은 임의의 길이의 비디오에서 detection을 위해 sliding window 방법을 우회합니다. [4, 17, 28, 29, 39] 는 RNN, LSTM을 이용하여 activity의 시간적 변화를 모델링하고 각 time 마다의 acitivity label을 예측합니다. deep action proposal model[4] 는 LSTM을 사용하여 16프레임의 C3D feature를 encode 하며 추가적인 proposal stage가 없이 regression을 통해 activity segment를 예측합니다. deep action proposal model과 비교하자면 우리는 RNN 레이러르 사용하지 않았고 큰 비디오를 fully-convolutional 3D conv 를 사용하여 encoding 합니다. 그리고 3D Roi pooling 을 사용하여 임의의 proposal을 다룰 수 있고 높은 정확도와 속도를 달성했습니다. [41] 방법은 다양한 해상도에서의 motion feature를 사용하였고 Pyramid of score distribution feature를 제안합니다. 그러나 이들 모델은 end-to-end 로 학습이 불가능하며 hand-crafed feature에 의존합니다.

supervised activity detection 과는 별도로 최근의 [36] 은 비디오 레벨에서 라벨링 되어 있는 데이터에서 weakly supervsied activity localization을 수행했으며 이는 uniformly sampled proposal, on-shot learning을 기반으로하는 attention wieght를 학습함으로써 달성합니다. [22] 는 language model을 살펴보고 acitivt length model을 제안합니다. [38, 40] 은 spatio-temporal activity localization을 다루고 이는 많은 변형이 존재합니다. 우리는 supervised temporal acitivt localization에만 집중합니다.

<b>Object Detection</b> : untrimmed 비디오에서의 acitivity detection은 object detection과 매우 관련되어 있습니다. 우리는 Faster RCNN, RCNN의 변형, Fast RCNN에서 영감을 얻었습니다. 이들은 ROI polling, region proposal network 를 사용합니다. 최근의 SSD, R-FCN 과 같은 object detection과 비교하여 Faster RCNN은 일반적이고 로버스트한 객체 검출 알고리즘이고 이는 다양한 데이터셋 에서 실험되었습니다. Faster R-CNN과 비슷하게 우리의 R-C3D 모델은 다양한 activity detection 데이터셋에서 쉽게 적용 가능하게 디자인 되었습니다. 이는 특정 데이터셋의 특징에 기반한 가정을 하지 않습니다. [18] UPC 모델이 ActivityNet 에 적용한 것은 각 비디오에 하나의 activity class가 포함되어 있다고 가정합니다. 우리의 방법의 효율성을 위해 세개의 데이터셋에서 실험하였습니다.

### Approach



 <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/c6bdeb45c4df1f0f22ecb05931bef14a447c6ab4/3-Figure2-1.png" />

우리는 RC3D 라는 새로운 네트워크를 제안하고 이는 continous 비디오 스트림에서의 activity detection에 적용됩니다. 피규어 2에 나와있는 네트워크 구조는 세가지 요소로 구성됩니다. 

1. 3D Conv 를 사용한 feature extractor
2. temporal proposal stage
3. activity classfication, refinement stage

효율적인 계산및 end-to-end 학습을 가능하게 하기 위해 proposal, classfiation sub-network는 C3D feature를 공유합니다. proposal sub-network는 acitivity를 포함하고 있을 것 같은 variable length temporal segment를 예측하며 classfication sub-network는 이들 proposal을 특정 acitivity or backgorund로 예측합니다. 그리고 proposal segment boundary를 refinement 합니다. (object detection에서의... refinement 트릭을 얘기하나 보다). 핵심적인 발전은 Faster R-CNN 에서의 2D Roi-pooling을 3D 로 확장한 것이며 이는 variable length proposal(ROI-Pooling으로 달성함)을 various resolutions(fully convolutional network 를 썻기 때문에?)에서 feature를 추출하게 합니다. 섹션 3.1 에서는 계층적으로 공유된 비디오 feature를 설명하며 3.2 에서는 temporal proposal subnet, 3.3 에서는 classfication subent, 3.3, 3.4 에서는 최적화 전략과 학습, 테스팅 방법을 설명합니다.

### 3.1. 3D Convolutional Feature Hierarchies

우리는 3D-conv 를 사용하여 풍부한 spatio-temporal feature 계층을 추출합니다. (왜 계층적 구조라는 것이지? C3D 논문을 봐봐야 할듯 하다.). spatial, temporal feature는 비디오를 represent 하는데 모두 중요하며 3D convent은 풍부한 spatial, temporal feature를 계층적 구조로 학습합니다. 모델의 인풋은 RGB 비디오 프레임이며 3xLxHxW 의 차원을 가집니다. 3D Conv의 구조는 [32]의 C3D 구조를 사용합니다. 하지만 [32]와 다른 점은 우리의 모델은 variable length를 다룰 수 있습니다. 우리는 C3D의 conv layer (conv1a 부터 conv5b)을 사용합니다. 그러므로 이 feature map은 512x(L/8)x(H/16)x(W/16) 의 차원을 가집니다. (512 는 conv5b의 채널입니다.) 우리는 이 conv5b featureamp을 proposal, classfication subnet의 인풋으로 사용합니다. 영상의 프레임의 height, width는 [32] 에 따라 112로 사용합니다. 프레임 갯수 L은 임의의 숫자이며 메모리에 의해서만 제한됩니다.

### 3.2. Temporal Proposal Subnet

variable length proposal을 예측하도록 하기 위해 우리는 temporal proposal sub-network 에 anchor segment를 사용합니다. subnet은 해당하는 anchor segment에 해당하는 proposal segment를 예측하며 예측된 proposal이 activity를 포함할지 안할지를 예측합니다. anchor segment는 temporal location을 L/8 차원의 temporal dimension의 점들을 중심으로 하는 multi scale window 입니다. K anchor segment 는 고정된 몇몇의 scale을 가집니다. 그래서 anchor segment 의 총 갯수는 (L=8)\*K 가 됩니다. K anchor의 같은 집합은 다른 temporal location에 존재하며 이는 proposal 예측이 temporally invariant하게 합니다. 이 anchor는 activity segment를 위한 일종의 reference로 제공되며 k의 값은 데이터에 의존적입니다.

각각의 temporal location에 해당하는 feature를 얻기 위해 우리는 conv5b의 뒤에 3x3x3 의 커널을 가지는 conv filter를 추가합니다. 이는 temporal proposal subnet을 위해 temporal receptive field를 확장합니다. (하여튼 faster RCNN과 똑같은 일을 한것). 그후 우리는 spatial dimension을 다운샘플링 하여 C_tpn이라는 512x(L/8)x1x1 의 feature를 얻습니다. 각각의 temporal location에 대한 각각의 512 차원의 feature vector는 상대적인 offset {ci, li} i = 1,,,K 를예측하기 위해 사용되고 이는 center point, length of anchor 를 예측합니다. 이는 또한 각 proposal이 acitivity or background일지에 대한 binary classfication score를 예측합니다. poposal offset, score은 C_tpn 에 1x1x1 conv를 사용하여 계산합니다. 

<b>Training</b> : 학습을 위해 anchor segment에 positive/negative label을 할당해야 합니다. 표준적인 object detection을 따라서 우리는 positive label을 다음과 같이 할당합니다.

1. gt activity 와의 IOU가 0.7 보다 proposla
2. gt activity 와 가장 높은 IOU를 가지는 proposal
3. 모든 gt와 IOU가 0.3 보다 작으면 negative sample로 

다른 것들은 모두 훈련에서 제외됩니다. proposal regression을 위해 섹션 3.4에 설명된 coordinate transformation을 통해 수행합니다. 우리는 positive/negative sample을 1:1 비율로 샘플링 했습니다.

### 3.3. Activity Classification Subnet

activity classfication stage는 3가지의 함수로 구성디 되어져 있습니다.

1. 이전 단계(proposal stage) 로 부터 proposal segment를 고른다 (classfication score로 부터 topn, nms 등을 통해)
2. selected proposal segment로 부터 fixed-size feature를 추출한다
3. pooled feature를 기반으로 하는 activity classfication, boundary regression을 수행한다.

proposal subnet으로 부터 생성된 몇몇의 activity proposal은 서로 많이 중첩되어 있으며 몇몇은 낮은 proposal score를 가집니다. [5,21] 과 같은 표준적인 object dection, [24, 39]와 같은 activity detection에 따라 우리는 NMS 전략을 사용하여 중첩되고 낮은 confidence proposal을 가지는 proposal들을 제거합니다. NMS 임계값은 0.7 로 설정됩니다.

seleceted proposal은 임의의 길이가 될 수 있습니다. 우리는 classfication, regression을 위한 fc-layer를 사용하기 위해서 고정된 크기의 feature를 추출할 필요가 있습니다. 우리는 3D Roi-pooling layer를 디자인 하였고 각각의 proposal로 부터 고저된 사이즈의 volume을 추출합니다. 이는 512x(L/8)x7x7 conv5b feature 에서 수행됩니다.(height, width 를 112로 고정해서 7이 나오나 봄) 3D pooling에 대해 더 구체적으로는 lxhxw 의 feature volume을 l_s x h_s x w_s 의 사이즈로 나눠집니다. sub-volumes은 근사적으로 (l/l_s)x(h/h_s)x(w/w_s) 사이즈가 되게 되며 max-pooling은 각 sub-volume에 대해 수행됩니다. 우리의 경우에는 각 proposal이 l_px7x7 사이즈라 가정하면 이 volume은 1x4x4 gird로 나눠지며 각각 gird에서 max-pooling이 수행됩니다. 그래서 arbitrary length의 proposal을 512x1x4x4로 만들 수 있습니다.

3D-Roi pooling의 결과는 두개의 fc-layer의 인풋으로 들어갑니다. 각 proposal은 어떤 activity인지 분류되며 regression layer에 의해 refine 됩니다. classfication, regression layer는 분리된 2개의 fc-layer이며 몇몇의 fc-layer를 거친후에 classfication, regression을 수행하게 됩니다.

<b>Training</b> : classfier subnet을 학습하기 위해서 각각의 proposal에 대해 activity label을 할당할 필요가 있습니다. 

1. gt label과 highest IOU를 가지는 proposal로 해당 라벨로 할당
2. gr label과 IOU가 0.5 보다 높으면 해당 라벨로 할당
3. IOU가 0.5보다 낮은 proposal은 background로 할당
4. positive/negative sample의 비율은 1:3

### 3.4 Optimization

clasfication, regression task를 jointly하게 최적화합니다. classfication을 위해서는 softmax loss loss, regression은 smooth L1 loss가 사용됩니다.

 <img src="https://latex.codecogs.com/gif.latex?LOSS%20%3D%20%5Cfrac%7B1%7D%7BN_%7Bcls%7D%7D%5Csum_i%20L_%7Bcls%7D%28a_i%2C%20a_i%5E*%29%20&plus;%20%5Clambda%20%5Cfrac%7B1%7D%7BN_%7Breg%7D%7D%20%5Csum_i%20a_i%5E*L_%7Bcls%7D%28t_i%2C%20t_i%5E*%29"/>


여기서 N_cls는 batch size, N_reg 는 anchor/proposal segment를 나타냅니다. lambda는 loss trade-off 파라미터이며 1로 설정했습니다. i 는 anchor/proposal segment의 인덱스이며 a_i 는 probability, a_i\*는 gt, t_i 는 anchor segment/proposal에 대하 ㄴ상대적인 offset입니다. t_i\* 는 gt 와 해당하느 segment/proposal에 대한 coordinate transform 입니다. coordinate transform은 다음과 같이 계산됩니다.

우리의 R-3D 모델에서 위의 loss function은 temporal proposal subnet, activity classfication subnet에 적용이 되었고 proposal subnet 에서의 binary classfication loss L_cls 는 proposal이 acitivity를 포함하는지 안하는지를 나타내며 regression loss L_reg는 proposal과 gr와의 상대적 차이를 최적화 합니다. acitivity classfication subent의 경우 multiclass classfication loss L_cls는 proposal에 대해 어떤 activity를 예측할지입니다. regression loss L_reg는 gt와의 상대적 차이를 최적화 합니다. 이 4가지 loss는 jointly 최적화됩니다.

### 3.5 Prediction

R-C3D 에서의 acitivity 예측은 2단계로 이뤄집니다. 첫번째로 proposal subnet이 후보군 proposal을 생성하고 score 뿐만 아니라 start-end time offset을 예측합니다. 그후 nms에 의해 refine 되며 임계값은 0.7로 합니다. nms 이후 선택된 proposal은 clasfication, regression layer의 인풋으로 들어가게 됩니다. classfication layer는 어떤 activity class 일지를 예측하며 regression layer는 예측된 proposal을 refine 합니다. proposal subnet, classfiation subent 에서의 boundary 예측은 center point, length of segment의 상대적 차이이 형식입니다. 예측된 start-time, end-time을 얻기위해서 inverse coordinate transformation이 수행됩니다.
# Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset

### Abstract 

UCF-101, HMDB-51 같은 현재의 action classfication 데이터셋의 적은 양의 비디오에서는 좋은 비디오 모델을 찾는것이 어렵습니다. 현재의 small-scale 벤치마크에서는 대부분의 방법들이 비슷한 성능을 얻었습니다. 이 논문에서는 최신의 구조들을 Kinetics Human Action video 데이터셋을 사용해 다시 평가합니다. Kinetics는 400개의 human action class와 클래스당 400개의 클립으로 이루어져 있으며 2배 이상의 더 많은 데이터로 구성되어져 있습니다. 이는 현실적이고 어려운 유투브 비디오에서 얻어졌습니다. 우리는 현재의 구조들이 Kineics 에서의 action classfication에서 어떻게 작동하는지에 대한 분석과 Kinetics에서 pre-trained 된 구조가 작은 데이터셋에서의 성능을 얼마나 올리는지를 보여줍니다.

또한 우리는 새로운 Two-Streamm inflated 3D Convnet이라는 구조를 제안합니다. 이는 2D Convnet inflation 을 기반으로 합니다. 이는 필터와 pooling 커널을 3D로 확장함으로써 spatio-temporal feature 를 잘 학습할 수 있고 ImageNet 에서의 구조와 그들의 파라미터를 잘 활용할 수 있습니다. Kinetics에서 pre-train된 I3D 모델은 지금까지의 최고성능의 action classfication을 향상시켰으며 HMDB-51 에서는 80.7% UCF-101 에서는 98.0% 를 달성했습니다.

### 1. Introduction

ImageNet에서 예기치 않은 이점 중 하나는 ImageNet의 1000개의 카테고리, 1000개의 이미지에서 학습된 구조가 다른 task, domain 에서 사용 될 수 있다는 것입니다. 초기에 이러한 것들이 발견된 것 중 하나는 PASCAL VOC의 classfication, detection 챌린지에 ImageNet에서 학습된 모델의 fc7의 feature를 사용하는 것입니다. 또한 AlexNet에서 VGG-16 으로 구조를 바꾸는 것은 PASCAL VOC 의 성능에 큰 영향을 끼쳤습니다. 그 이후로도 ImageNet에서 학습된 구조를 사용해 warm-starting 혹은 다른 테스크에 그대로 사용되는 경우가 많습니다. segmentation, depth prediction, pose estimation, action classfication 등에 사용됩니다.

비디오 영역에서 충분히 큰 데이터셋에서 학습된 action classfication 네트워크가 이미지에서와 같이 다른 task, 데이터셋에 적용될때 비슷한 성능 향상을 보일지에 대한 것은 open question 입니다. 대량의 비디오 데이터셋을 구축하는 것의 어려움은 현재의 가장 유명한 action recognition 데이터셋이 매우 적은양 10k 정도의 비디오로 이루어져 있는 것이 보여줍니다.

이 논문에서는 HMDB-51, UCF-101 보다 2배정도 더 큰 새로운 Kinetics 데이터셋을 이용하여 이러한 질문에 답을 하고자 합니다. Kinetics는 400개의 human action class를 가지며 각 클래스랑 400개의 샘플을 가지고 있습니다 이는 개별적인 유투브 비디오에서 모아졌습니다.

우리의 실험은 연구에서 주요한 몇몇의 구조들을 다시 구현한다음 이를 Kinetics 에서 pre-train 하고 이를 다시 HMDB-51, UCF-101에 fine-tuning 하여 이들의 transfer behavior를 분석하는 것입니다. 결과는 pre-training 을 사용하는 것이 성능의 향상을 보이지만 성능 향상의 정도는 모델의 구조에 따라 달라집니다. 

이러한 결과를 바탕으로 우리는 새로운 모델을 제안하고 Kinetics에서의 pre-train 결과를 활용해 높은 성능을 달성 할 수 있습니다. 이 모델은 "Two Stream inflated 3D Convnet" (I3D) 라고 불리며 최신의 이미지 classfication 구조를 기반으로 만들어졌습니다. 하지만 그들의 필터와 pooling 커널을 3D로 부풀리고 (선택적으로 파라미터의 양도) 매우 깊은 spatio-temporal classfier를 만들 수 있게 합니다. Inception V1을 기반으로 하는 I3D 모델은 Kinetics에서 pre-train 후에 state-of-the-art를 능가하는 성능을 보였습니다.

우리의 모델 비교에서 우리는 bag-of-visual-word representation과 같은 고전적인 접근방법을 비교하지 않았습니다. 하지만 Kinetics 데이터셋은 공개되어서 다른 사람들이 이러한 것들을 비교할 수 있습니다.

다음 섹션에서는 구현한 action classfication 모델에 대해 간략히 설명합니다. 섹션 3은 Kinetics 데이터셋에 대한 간략한 개요를 제공합니다. 섹션 4는 Kinetics에서의 이전의 모델에 대한 성능을 보고 합니다. 섹션 5에서는 Kinetics 에서의 pre-train 모델이 다른 데이터셋에서 얼마나 잘 transfer 되는지를 보고합니다. 이 논문은 결과에 대한 discusssion을 통해 결론을 맺습니다.

### 2. Action Classification Architectures

최근 몇년 동안 이미지 representation 구조에 대한 발전이 빠르게 일어났지만 비디오에서의 구조는 명확하지 않습니다. 현재 비디오 아키텍쳐의 주요한 차이는 2D 혹은 3D 커널을 사용하는지 여부입니다. 또한 네트워크의 인풋이 단지 RGB 비디오 인지 pre-computed 된 optical-flow 를 포함하는지가 다릅니다. 2D Convnet의 경우 프레임간에 정보가 얼마나 propagate 되는지가 중요하며 이는 LSTM 같은 temporally-recurrent layer를 사용하던지 시간에 대한 정보를 feature aggregation 하게 됩니다.

이 논문에서는 이러한 것들에 걸쳐져 있는 모델들을 비교하고 연구합니다. 2D Convnet 방법들 중에서 우리는 [5, 34]와 같은 LSTM 을 같이 사용한 것을 고려합니다. 또한 [8, 25] 와 같은 RGB와 optical-flow를 결합하는 two-stream 네트워크를 고려하며 C3D와 같은 3D ConvNet을 고려합니다.

이 논문의 주요한 공헌으로 우리는 TwoStream Inflated 3D ConvNet(I3D) 를 소개합니다. 기존의 3D Convnet은 매우 많은 파라미터와 비디오 데이터의 부족으로 매우 얕았습니다.(8 레이어 정도). 우리는 Inception, vgg-16, resnet과 같은 구조가 spatio-temporal feature extractor로 확장 가능하며 그들의 weight를 좋은 초기화로 사용할 수 있음을 관찰했습니다. 우리는 또한 two-stream 구성이 여전히 유용하다는 것을 발견했습니다.

우리가 실험한 5개의 구조에 대한 그림이 피규어2에 주어져 있으며 temporal interface에 관한 명세가 테이블 1에 주어져 있습니다.

C3D를 제외한 많은 모델들이 Imagenet에서 pre-trained 된 모델을 사용합니다. 우리의 실험 전략은 ImageNet 에서 pre-train 된 모델을 back-bone으로 사용하며 우리는 Inception-V1 with BN 을 사용하며 다양한 방법으로 변형합니다. 이러한 pre-train 가능한 back-bone이 공통적으로 존재하므로 우리는 action classfication에 가장 큰 영향을 주는 변화를 구분 할 수 있을 것이라고 기대합니다.


<img src="http://img.blog.csdn.net/20170912144126470?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcGFyYW5vaWRfQ05O/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast"/>

### 2.1. The Old I: ConvNet+LSTM

이미지 classfication은 높은 성능을 보이므로 비디오에 적용할 때 가능한한 적은 변화와 이미지 classfication을 최대한 이용하는 것이 바람직 합니다. 이는 각 프레임에 대해 독립적으로 feature를 추출하고 전체의 비디오에 걸쳐 그들의 예측을 pooling 하는 것으로 얻을 수 있습니다. 이는 일종의 bag of words 이미지 모델링 이며 실무적으로는 잘 작동하지만 temporal structure를 완전히 무시합니다. (이러한 모델은 잠재적으로 닫힌 문과 열린 문을 구별하지 못한다)

이론적으로 [5, 3] 와 같이 LSTM과 같은 recurrent layer를 추가 할 수 있습니다. LSTM은 state를 encode하며 temporal order와 long range dependency를 포착할 수 있습니다. 우리는 LSTM과 BN을 같이 사용했고 Inception-V1의 마지막 average pooling 이후 LSTM을 추가했습니다. 그 후 classfication을 위한 fc-layer가 추가됩니다.

이 모델은 모든 time-step에 걸친 결과에 대한 cross-entropy loss에 의해 학습됩니다. testing 시에는 마지막 프레임의 결과만 사용합니다. input 비디오 프레임은 25 fps 비디오 스트림에서 샘플링된 5개의 프레임을 사용합니다. 모든 모델의 full temporal 정보는 테이블 1에 나와있습니다.

### 2.2. The Old II: 3D ConvNets

3D ConvNet은 비디오를 모델링 하는 자연스러운 접근방식처럼 보이며 표준적인 convolutional 네트워크가 비슷하지만 spatio-temporal 필터가 존재합니다. 이들은 [14, 28, 29] 등에서 여러번 연구도었으며 이들은 중요한 특징을 가집니다. 이들은 spatio-temporal data의 계층적 representation을 만듭니다. 이러한 모델의 문제는 2D Convnet보다 많은 파라미터를 가지며 학습하기 힘들게 됩니다. 이는 temporal 차원의 커널이 추가되기 때문입니다. 또한 이들은 ImageNet pre-train의 장점을 사용하지 못하며 이전의 결과들은 상대적으로 얕은 구조를 가지고 scratch 로 학습되었습니다. 이러한 모델의 벤치마크는 유용하지만 최신의 기술과는 떨어지는 성능을 보였고 이러한 모델을 Kinetics에서의 실험의 후보로 삼았습니다.

이 논문에서 우리는 C3D의 몇몇 변형들을 구현하였고 8개의 레이어, 5개의 Pooling 레이어, 2개의 fc-layer를 가집니다. 이 모델의 인풋은 16프레임의 112x112 클립이며 이는 원래 논문의 구현과 같습니다. [29] 와 다른 점은 우리는 모든 convtional, fc-layer 후에 BN을 추가하였으며 원래의 모델은 첫번째 pooling layer에서 1의 stride를 사용했지만 우리는 2 stride를 사용했습니다. 이는 메모리 사용을 줄여줘 더 큰 batch size를 가능하게 합니다. 큰 batch size는 BN 에서 매우 중요합니다.(특히 fc-layer 이후의 BN에서 중요하며 weight tying 이 존재하지 않습니다. weight tying이 뭐지??) stride 2를 사용하여 표준적인 K40 GPU에서 배치당 15개의 비디오를 학습할 수 있었습니다.

### 2.3. The Old III: Two-Stream Networks

ConvNet의 마지막 레이어의 feature를 LSTM 의 인풋으로 사용하는 것은 high-level variation을 모델링 할 수 있지만 많은 경우에서 종종 중요한 low-level motion을 포착할 수 없습니다. 또한 학습 과정에서 여러 프레임에 대해 네트워크를 unrolling 해야 함으로 back-prop 의 계산이 비싸집니다. [25] 에서 소개된 실용적인 접근 방법은 short temporal snapshot 을 모델링 하기 위해 RGB 프레임의 예측을 평균내고 optical-flow를 10개 쌓은 프레임을 사용합니다. 후에 두개의 스트림이 Imagenet에서 pre-train된 Convnet의 인풋으로 들어가게 됩니다. optical flow 스트림은 input conv layer를 인풋의 채널의 2배 만큼으로 바꿔 적용되며(optical flow 프레임은 2개의 채널 horizontal, vertical 채널로 구성된다) 테스트 타임에는 비디오로 부터 여러개의 스냅샷이 샘플링 되며 action 예측은 평균으로 수행됩니다. 이것은 기존의 벤치마크에서 매우 높은 성능을 보였고 훈련과 testing 이 매우 효율적입니다.

최근의 [8] 은 이를 확장하여 마지막 conv layer에서 spatial, flow 스트림을 결합합니다. HMDB 에서 성능 향상을 보였으며 less test time augmentation(??이게 뭘 의미하지) 입니다. 우리의 구현은 [8] 을 따랏으며 Iception-V1 구조를 사용합니다. 네트워크의 인풋은 10프레임에서 샘플링된 5개의 연속적인 RGB 프레임과 10프레임에 해당하는 optical flow를 인풋으로 받습니다. Inception-v1의 마지막 average-pooling 전의 spatial, motion feature 는 3x3x3 3D conv layer를 통과하며 512 차원의 아웃풋 채널을 갖습니다. 3x3x3 3D max-pooling layer 를 수행하며 마지막 fc-layer가 존재합니다. 새로운 레이어에 대한 초기화는 Gaussian noise를 통해 이루어졌습니다. 원래의 2D two-stream과 3D 버젼 모두 end-to-end로 학습됩니다. (원래 모델의 two-stream average process를 사용합니다.)

### 2.4. The New: Two-Stream Inflated 3D ConvNets

이 구조를 통해 우리는 3D Convnet이 어덯게 ImageNet 2D Convnet 디자인에서 이점을 얻는지와 ImageNet 2D Convnet의 학습된 파라미터를 사용하는지를 보여줍니다. 우리는 또한 two-strema 구성을 채택하고 이를 섹션 4에서 보여줄 것입니다. 3D ConvNet이 RGB 스트림에서 직접적으로 temporal pattern을 학습 할 수 있지만 optical-flow 스트림을 추가함으로써 성능을 크게 향상 시킬 수 있습니다.

<b>Inflating 2D ConvNets into 3D</b> 지난 년도 동안 성공적인 이미지 classfication이 개발되었습니다. spatio-temporal 모델의 프로세스를 반복하는 대신에 우리는 성공적인 2D classfication 모델을 3D Convnet으로 변형합니다. 이는 2D 구조로 부터 시작하여 모든 필터, Pooling 커널을 temporal dimension에 대해 확장함으로써 수행할 수 있습니다. 필터는 일반적으로 정사각형 임으로 우리는 NxN 필터를 NxNxN 필터로 확장합니다.

<b>Bootstrapping 3D filters from 2D Filters. </b> 구조를 변경하는 것 이외에도 pre-train ImageNet 모델의 파라미터를 bootstrap 하는 것을 원할 수 있습니다. 이를 위해 이미지를 (boring) 비디오로 변환하는데 이미지를 단순히 비디오 sequnce로 복사합니다. 이 3D 모델은 boring-video fixed point 라는 것을 만족함으로써 implicitly pre-trained 될 수 있습니다. boring 비디오의 pooled activation은 원래 인풋 이미지와 같아야 합니다. linearity 덕분에 이러한 것이 가능하며 N time dimension 만큼 2D filter의 웨이트를 반복하고 이들을 N으로 나누게 됩니다. 이를 통해 convolutional filter의 activation이 같아지게 됩니다. boring 비디오의 output이 time에 대해 constant이기 때문에 pointwise non-linearity layer의 아웃풋과 max-pooling layer의 average 가 2D의 경우와 같아지게 되고 모든 네트워크의 response는 boring-video fixed point를 반영하게 됩니다.

<b>Pacing receptive field growth in space, time and network depth.</b> boring video fixed-point는 얼마나 time dimension에 대해 pooling 연산을 확장할지와 conv/pooling 에서 temporal stride를 설정할지에 대해 자유롭습니다. 이들은 feature receptive field의 크기를 결정하는 아주 중요한 요소입니다. 사실상 모든 image 모델은 두개의 spatial dimension(horizontal, vertical) 동일하게 처리합니다. (pooling 커널과 스트라이드를 똑같이 적용한다면?) 이는 매우 자연스러운 것입니다. 이는 네트워크의 깊은 layer에서 horizontal, vertical 축에서 멀어 지는 것이 이미지의 location에서 같은 영향을 받는다는 것을 의미합니다. 그러나 time을 고려하면 symmetric receptive field를 사용할 필요가 없습니다. 이는 프레임 속도와 이미지에 따라 달라집니다. space에 비해 상대적으로 time 축이 빠르게 변하면 다른 object의 edge가 융합되고 낮은 레벨의 feature detection이 안좋아 질 수 있습니다. time 축이 너무 느리면 scene dynamics를 포착하지 못할 수 있습니다.

Inception-V1 에서는 첫번째 conv layer가 스트라이드 2를 가지며 스트라이드 2인 4개의 max-pooling, 7x7 average-pooling layer가 존재하며 그 후 fc-layer가 classfication을 위해 존재합니다. max-pooling layer는 inception branch를 통해 병렬적으로 계산됩니다. 우리의 실험에서는 인풋 비디오가 25fps 로 처리되었습니다. 우리는 처음 두개의 max pooling에서는 temporal pooling을 수행하지 않는 것이 도움이 된다는걸 발견했고(1x3x3 커널로 time에는 스트라이드 1을 적용)  다른 max-pooling layer에서는 symmetric 커널을 사용합니다. 마지막 에는 2x7x7 average pooling 을 사용합니다. 전체 아키텍쳐는 피규어 4에 나와있습니다. 우리는 64 프레임의 snippets을 사용하여 모델을 학습하고 전체 비디오에 대해 temporally 예측을 평균냄으로써 테스팅 합니다.

<b> Two 3D Streams</b> 3D Convnet이 RGB 인풋에서 직접적으로 motion을 배울때 pure feed forward 연산을 수행하는 반면 optical flow 알고리즘은 recurrent한 해석을 가집니다. (각 flow field에 대해 반복적인 최적화를 수행합니다) 아마 이러한 recurrenc의 결어로 인해 여전히 two-stream 구성을 갖는 것이 효율적인 것으로 나타났습니다. 하나의 I3D 네트워크는김사랑 RGB 인풋에 의해 학습되며 다른 하나는 optical flow에 의해 학습됩니다. 우리는 두가지 네트워크를 독립적으로 학습시키고 그들의 예측을 평균냄으로써 test를 수행합니다.

### 2.5. Implementation Details

C3D-like 3D Convnet 을 제외한 모든 모델들은 ImageNet pretrained Inception-V1을 베이스로 합니다. 모든 구조에서 conv layer후에 BN, ReLU를 적용하며 class score를 해당하는 마지막 conv layer는 제외합니다.

학습은 표준적인 SGD, momentum 0.9 를 모든 경우에 사용했으며 32 GPU에서의 synchronous parallelization을 통해 학습했습니다. 3D ConvNet은 여러개의 프렝미을 입력받기 때문에 큰 batchsize를 위해선 더 많은 GPU가 필요합니다. 우리는 3D ConvNet에 대해선 64 GPU를 사용했습니다. 35K step 까지는 miniKinetics 를 사용하여 학습하고 110k 까지 Kinetics에 대해 학습했습니다. validation loss가 saturate하면 learning rate를 10씩 감소했습니다. 우리는 mini-Kinetics의 valdiation set을 사용하여 하이퍼파라미터를 튜닝했습니다. UCF-101, HMDB-51에 대해서는 5K step 까지 학습했으며 Kinetics 에서와 비슷한 learning rate decay를 사용하였고 16개의 gpu만을 사용했습니다. 모든 모델들은 Tensorflow로 구현되어졌습니다.

Data augmentation은 깊은 구조에서의 성능에 매우 중요하다고 알려져 있습니다. 학습과정동안 우리는 spatially, temporally random crop을 수행했습니다. 가장 작은 비디오의 면은 256으로 만들고 여기에서 224x224 패치를 random-crop 합니다. temporally 하게는 원하는 프레임수 (64 프레임) 가 되도록 start 프레임을 선택합니다. 짧은 비디오에는 모델의 인풋 프레임 만큼을 충족시킬 때까지 비디오를 단순 반복합니다. 우리는 또한 훈련 도중 left-right random horizontal flip 을 수행합니다. 테스트 타임에는 전체 비디오에 모델이 fully convolutional 하게 적용되며 224x224로 center crop 하며 예측은 평균을 내서 구합니다. 256x256 비디오에 대해서도 실험을 해봤지만 성능의 향상을 관찰하지는 못했습니다. 테스트 타임에도 left-right flip을 사용한다면 좀더 나은 성능이 나을 수 있으며 트레이닝 타임에 photometric같은 추가적인 data augmentation도 나은 성능을 보일 수 있습니다. 

### 3. The Kinetics Human Action Video Dataset

Kinetics 데이터셋은 activites, event 이 아닌 human action에 초점을 맞춥니다. action class들은 drawing drinking, laughing, punching과 같은 Person action과 hugging, kissing, shaking hands와 같은 person-person action, opening presents, mowinf lawn, washing dish와 같은 person-object action을 포함합니다. 몇몇의 action은 fine-grain 하여 temporal reasoning을 구별하는 것이 필요합니다. 예를들어 다른 종류의 수영을 구별해야 합니다. 다른 action들은 object를 구별하는 것이 필요합니다. 예를들어 다른 종류의 관악기를 연주하는 action이 있습니다.

데이터셋에는 400개의 human action class를 포함하며 각 클래스랑 400개 이상의 클립이 존재합니다. 각 클립은 고유한 비디오에서 추출되어졌으며 총 240k의 학습 비디오가 존재합니다. 클립은 약 10초 동안이며 이들은 모두 trim된 비디오 입니다. 데이터셋에 대한 전체 설명과 어떻게 구축했는지에 대한 설명은 [16] 에 있습니다.

이 논문의 대부분의 실험에서는 전체 Kinetics보다 작은 mini-Kinetics를 사용합니다. 이것으 213개의 클래스와 총 120k 클립을 가지는 데이터셋의 초기버젼 이며 3개의 split-data를 제공합니다. 트레이닝 셋은 클래스당 150~1000개의 클립이 존재하며 validation 셋에는 클래스랑 25 클립, 테스트 셋에는 클래스당 75 클립이 존재합니다.

### 4. Experimental Comparison of Architectures

이 섹션에서는 섹션2에서 기술된 5가지의 구조에 대해 실험하며 다양한 데이터셋에서의 트레이닝, 테스트 결과를 비교합니다. 테이블 2는 UCF-101, HMDB-51, mini-Kinetics에서의 트레이닝, 테스트 accuracy를 보여줍니다. UCF-101, HMDB-51에서는 split 1 test 셋에서 테스트를 진행하였고 mini-Kinetics에서는 주어진 테스트 셋에서 진행하였습니다. 몇가지 주목할 만한 것이 있습니다. 

첫째로, 우리의 I3D 모델은 모든 데이터에서 최고의 성능을 보였습니다. 이는 RGB, flow, RGB+flow 에서 모두 달성하였습니다. I3D 가 매우 많은 수의 파라미터를 가지고 있고 UCF-101, HMDB-51은 너무 작기 때문에 ImageNet 에서의 pre-train 된 모델이 3D ConvNet으로 확장되는 것의 이점을 보여줍니다.

두번째로, 모든 모델은 UCF-101보다 mini-Kinetics에서 낮은 성능을 보입니다. 이는 두 데이터셋에서의 어려움의 차이를 보여줍니다. 그러나 mini-Kinetics 에서의 성능이 HMDB-51 보다 좀더 높은 것을 보여줍니다. 이는 HMDB-51의 학습 데이터의 부족 때문일 수도 있고 HMDB-51의 데이터셋이 좀더 어렵게 만들어 졌기 떄문일 수 있습니다. HMDB-51의 많은 클립들은 같은 장면에서의 다른 action을 포함하고 있습니다. (drawng sword 비디오는 같은 비디오에 "sword", "sword exercise" 라는 다른 라벨이 달려 있습니다.)

세번째로, 다른 구조들의 성능의 순위는 대부분 일관성을 유지합니다.

네번째는 LSTM, 3D ConvNet 모델이 Mini-Kinetics 데이터에서 더 경쟁력이 있습니다. 이 모델들은 많은 데이터가 필요한 것으로 보입니다.

또한 two-stream 구조는 모든 데이터셋에서 우수한 성능을 보여주지만 RGB, optical flow의 상대적 가치는 mini-Kinetics와 다른 데이터셋과 다른 양상을 보여줍니다. flow를 추가함으로써 얻는 이득은 UCF-101 에서는 약간 높고, HMDB-51 에서는 꽤나 높으며 mini-Kinetics에서는 낮은 이득을 얻었습니다. 데이터셋을 육안으로 검사한 결과 Kinetics는 camera motion이 더 많아 motion stream의 학습을 어렵게 만드는 것을 알 수 있었습니다. I3D 모델은 다른 모델에 비해 더 많은 flow stream을 가지며 더 통합된 temporal feature 를 가지는 것으로 보입니다. 하지만 이것은 다른 모델에 비해 더 긴 temporal receptive field(64프레임 vs 10 프레임) 을 가지기 때문으로 설명될 수 있습니다. RGB 스트림이 discriminative information을 가지는 것이 타당해 보입니다. 우리는 종종 육안을 통해 flow 만으로 action을 알아내려고 했지만 매우 힘들었고 이는 RGB에서는 일어나지 않았습니다. 이 구조에 일종의 Motion stabilization을 통합하는 것이 향후의 연구가 될 수 있습니다.

### 4.1. Results on the full Kinetics dataset

우리는 I3D 모델을 full Kinetics 데이터셋에서 학습시켰고 imageNet weight를 이용하는 경우와 안쓰는 경우로 나눴습니다. 테이블3을 통해 결과를 보여줬고 ImageNet weight를 사용하고 flow 스트림을 가지는 것이 성능의 향상이 있는 것을 보여줍니다.

### 5. Experimental Evaluation of Features

이 섹션에서는 Kinetics에서 학습된 네트워크의 일반화 가능성을 평가합니다. 이를 평가하기 위해 2가지 방법을 사용합니다. 첫째로, 네트워크의 가중치를 고정시킨채 UCF-101/HMDB-51 데이터셋에서의 feature를 추출합니다. 그다음 UCF-101/HMDB-51의 학습데이터를 사용하여 multi-way soft-max classfier를 학습시키고 테스트셋에서 evaluation 합니다. 두번째로 네트워크를 UCF-101/HMDB-51의 학습데이터를 이용해 fine-tuning 시키고 테스트셋에서 evalute합니다.

이 결과는 테이블 4에 나와 있으며 모든 모델이 mini-Kinetics에서 pre-training의 이득을 얻는 것을 볼 수 있습니다. 몇몇의 모델은 더 많은 이득을 보았는데 I3D-ConvNet, 3D-ConNet 이 pre-train의 효과를 많이 보았습니다. miniKinetics에서 pre-train후 마지막 layer만 다시 학습하는 것도 scratch 부터 학습하는 것보다 훨씬 나은 성능을 얻을 수 있었습니다.

I3D 모델이 더 나은 transferability 를 보이는 것에 대해 high temporal resolution 때문으로 설명 할 수 있습니다. 이들은 25 fps의 영상에서 64 프레임 비디오 snippet을 사용해 학습되었고 테스트시에는 모든 비디오 프레임을 사용합니다. 이는 I3D가 fine-grained temporal structure of action(다른 종류의 수영을 구별할 수 있도록 함)을 포착할 수 있도록 만듭니다. 다르게 말하면, 이러한 큰 데이터셋에서는 sparser video input(적은 프레임의 인풋인듯) 의 이점이 적을 수 있습니다. sparser video input의 경우 비디오는 ImageNet에서의 단일 이미지와 많이 다르지 않기 때문입니다. C3D-like 모델과 I3D 모델의 차이는 더 깊어질 수 있고 훨씬 적은 파라미터를 사용합니다. 또한 ImageNet의 웨이트로 초기화 할 수 있고 4배 긴 비디오, 2배 더 높은 spatial resolution의 비디오에서 학습됩니다.

### 5.1. Comparison with the State-of-the-Art

테이블 5에는 UCF-101, HMDB-51 에서의 I3D과 이전의 최신 방법들의 성능을 비교합니다. 우리는 mini-Kinetics, full-kinetics에서 학습된 결과를 기술합니다. 학습된 모델의 conv1 필터가 피규어 4에 나와있습니다.

많은 방법들이 비슷한 결과를 얻지만 가장 좋은 결과는 Feichtenhofer 의 [7] 의 결과입니다. 이는 Resnet-50 을 사용하고 RGB, flow 스트림을 사용합니다. UCF-101 에서는 94.6% HMDB-51 에서는 70.3% 의 결과를 보이며 이는 dense trajectories 모델과 결합된 결과입니다. 우리는 3가지 train/test split에 대해 평가하고 mean accuracy를 벤치마킹 했습니다. Kinetics 에서 pre-train된 RGB-I3D 혹은 RGB-flow 모델을 단독으로 사용해도 다른 모든 모델의 성능을 능가합니다. 우리의 결합된 two-stream 구조는 UCF-101 에선 98% HMDB-51 80.7% 
으로 이전의 모델에서의 성능을 크게올렸습니다. 이전의 최고의 모델인 [7] 과 비교해서 각각 63%, 35%의 오분류율을 줄였습니다. 

기존의 C3D 모델과 Kinetics pre-trained I3D 모델은 매우 큽니다. C3D가 더 Sports-1M의 1M 샘플에서 학습되어 더 많은 데이터를 이용했음에도 큰 차이를 보이며 앙상블, IDT와 결합한 C3D보다 좋은 성능을 보입니다. 이는 Kinetics 데이터의 좋은 품질로 설명될 수 있으며 I3D가 더 나은 구조이기 때문일 수 있습니다.

또 다른 주목할 만한 효과는 mini-Kinetics와 비교하여 full-Kinetics 에서의 pre-train된 모델을 HMDB-51에 적용했을때 I3D-RGB 스트림의 성능을 향상시키는 것입니다. 이는 3D ConvNet이 robust motion feature를 학습하기 위해서 많은 양의 데이터가 필요하다는 것을 보여줍니다. full-Kinetics에서 pre-train 된 후 two-stream은 비슷한 성능을 얻지만 여전히 도움이 됩니다. 두 스트림의 예측을 평균 냈을때 74.8%(I3D-RGB) 에서 80.7%(I3D-two stream)로 성능 향상을 보였습니다.

### 6. Discussion

Introduction 에서 제시된 질문으로 돌아갑니다. "동영상에서 transfer learning의 효과가 있습니까?" ImageNet pre-trained CNN이 다른 테스크에 유용했던 것처럼 대형 비디오 데이터에서 pre-train이 큰 효과가 있음이 분명합니다. 이는 비슷한 테스크를 가지는 한 데이터셋 (Kinetics) 에서 다른 데이터셋 (UCF-101/HMDB-51) 로의 transfer learning을 보여줍니다. 하지만 Kinetics pre-traning 모델이 semantic video segmentation, video object detection, opticla flow computation과 같은 다른 task 에도 유용한지에 대한 검증이 필요합니다. 우리는 Kinetics 데이터셋에서 학습된 I3D 모델을 공개해 이 분야의 연구가 촉진되도록 할 것입니다.

물론 우리는 구조에 대한 포괄적인 탐색을 하진 않았습니다. 예를 들어 [11, 17]과 같은 action tube, [20] 과 같은 human actor에 대한 attension 방법을 고려하진 않았습니다. 최근에 제안된 연구인 [22, 24] 에서는 object detection과 결합하여 two-stream 구조 내에서 actor의 spatial, temporal 범위를 찾는 방법을 제안합니다. space와 tiem과의 관계는 미스터리합니다. 몇몇의 창의적인 논문은 이러한 관계를 포착하려고 시도합니다. 예를들어 [9] 에서는 action class에 대한 frame ranking function을 학습하고 이를 representation으로 활용합니다. [33] 에서는 action과 transforamtion 간의 분석을 수행합니다. [2] 에서는 프레임 시퀀스에서의 2D visual snapshot을 생성합니다 이는 고전적인 motion 에 관한 [3]의 아이디어와 관련되어져 있습니다. 우리의 비교에서 이러한 모델을 포함하는 것은 큰 가치가 있지만 시간과 공간의 부족 때문에 불가능 합니다 

미래의 작업으로 우리는 mini-Kinetics가 아닌 full-Kinetics를 사용하여 모든 실험을 반복해 볼 계획이며 다른 최신의 2D ConvNet을 확장할 계획입니다.

#### 읽어볼 관심가는 논문

[22] X. Peng and C. Schmid. Multi-region two-stream R-CNN
for action detection. In European Conference on Computer
Vision, pages 744–759. Springer, 2016

[24] S. Saha, G. Singh, M. Sapienza, P. H. Torr, and F. Cuzzolin.
Deep learning for detecting multiple space-time action tubes
in videos. British Machine Vision Conference (BMVC) 2016,
2016
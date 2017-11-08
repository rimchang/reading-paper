# Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition

### Abstract

spatio-temporal 3D 커널을 사용하는 CNN 네트워크는 action recognition을 위해 비디오로 부터 spatio-temporal feature를 추출할 수 있습니다. 3D 커널이 파라미터가 매우 많아 overfit하는 경향이 있지만 3D CNN은 거대한 비디오 데이터셋을 이용함으로써 성능향상이 있었습니다. 그러나 3D CNN의 구조는 Resnet과 같은 깊은 2D CNN에 비해 매우 앝은 깊이를 가지고 있습니다. 이 논문에서는 우리는 Resnet을 기반으로한 3D CNN을 제안합니다. 우리는 3D Resnet의 학습의 세부사항을 기술하고 ActivityNet, Kinetics 데이터셋에서 3D Resnet을 평가합니다. kinetics 데이터에서 학습된 3D Resnet은 모델의 매우 많은 파라미터에도 overfit을 겪지 않았으며 C3D와 같은 얕은 네트워크에 비해 더 나은 성능을 얻었습니다. 우리의 코드와 pre-trained 모델은 공개되어져 있습니다.

### 1. Introduction

실세계에서의 중요한 정보중 하나가 인간의 행동입니다. 비디오에서 인간의 행동을 인식하고 탐지하는 것은 감시시스템, 비디오 인덱싱, HCI와 같은 분야에서 널리 사용됩니다.CNN은 action recognition에서 높은 성능을 달성했습니다. 대부분의 CNN이 2D kernel을 사용하며 이는 이미지 인식에서와 비슷합니다. two-stream 구조는 RGB, optical-flow를 모두 사용하여 비디오에서의 spatio-temporal 정보를 표현합니다. 두 스트림을 합치는 것은 action recognition의 성능을 향상시킵니다.

시공간 정보를 포착할 수 잇는 다른 접근 방법은 2D 커널대신 3D 커널을 사용하는 것입니다. 3D CNN의 매우 많은 파라미터 때문에 UCF-101, HMDB51과 같은 작은 데이터셋에서느 학습이 힘들고 2D CNN과 비교해 낮은 성능을 보입니다. Kinetics와 같은 최근의 데이터셋은 3D CNN의 성능을 향상시키는데 큰 기여를 합니다. 3D CNN의 성능은 2D CNN에 비해 얕더라도 경쟁할만한 성능을 보입니다.

action recognition을 위한 깊은 3D CN에 대한 연구는 매우 많은 파라미터로 인한 학습의 어려움 때문에 잘 연구되지 않았습니다. 이미지 인식에서의 Resnet과 같은 매우 깊은 네트워크는 이미지 인식에서의 성능을 향상시켰습니다. action recognition의 성능 향상을 위해서는 더 깊은 3D CNN 구조를 연구하는 것이 중요합니다. Resnet은 가장 강력한 구조중 하나입니다. Resnet의 구조를 활용해 3D CNN을 구성하는 것은 action recognition의 성능 향상을 할 것이라 기대합니다.
 
이 논문에서는 action recognition을 위한 모델을 얻기 위해 3D Resnet을 실험합니다. 즉 시공간 recognition에서의 표준적인 pretrained 모델을 만드는 것이 목표입니다. 우리는 2D 기반의 Resnet을 확장하여 3D로 만들었습니다. ActivityNet, Kinetics 데이터를 이용하여 네트워크를 교육하고 그들의 성능을 평가합니다. 우리의 주요 공헌은 3D 커널을 사용한 Resnet 구조의 효과를 연구하는 것입니다. 이 작업이 3D CNN을 사용한 action recognition을 발전시킬 것이라 기대합니다.

## Related work

### 2.1 Action Recognition database

HMDB51, UCF101은 action recognition에서 가장 성공적인 데이터베이스 입니다. 그러나 최근에는 이 두 데이터가 그다지 크지 않다고 말합니다. 이들 데이터를 사용하여 우리의 모델이 overfit하지 않게 하는 것은 어렵습니다. 최근데은 거대한 Sports-1m, Youtube-8m 과 같은 데이터베이스가 제안되었습니다. 이들 데이터는 매우 크지만 주석이 노이즈가 많고 비디오 수준으로만 달려있습니다. 이러한 노이즈와 action과 관련없는 프레임은 학습이 잘 안되게 합니다. 이미지넷에서 학습된 매우 성공적인 pre-train 모델과 같은 것을 만들기 위해 딥마인드는 Kinetics 데이터셋을 발표했습니다. Kinetics 데이터셋은 300,000 비디오를 포함하고 400개의 카테고리를 가집니다. Kinetics 데이터의 크기는 Sports-1m, Youtube-8m 보다는 작지만 라벨링의 품질이 매우 좋습니다. 우리는 3D Resnet을 학습하기 위해 Kinetics 를 사용합니다.

### 2.2 Action Recognition Approach

CNN 기반의 action recognition중 가장 유명한 접근은 2D 커널을 사용한 two-stream 방법입니다. Simonyan은 RGB와 stacked optical flow를 사용하는 방법을 제안합니다. 이들은 두개의 스트림을 합치는 것이 성능의 향상된다는 것을 보였습니다. two-stream CNN을 기반으로하는 많은 방법들이 더욱 성능을 높였습니다. Feichtenhofer는 two-stream CNN을 Resnet 과 결합한 방법을 제안합니다. 이들은 Resnet의 구조가 action recognition에 도움된다는 것을 보였습니다. 위의 연구와 다르게 3D CNN 에 주목합니다. 매우 큰 비디오 데이터에서 3D CNN은 2D CNN의 성능을 능가합니다.

Ji는 비디오에서 spatio-temporal feature를 추출하기 위해 3D conv를 적용합니다. Tran은 C3D라고 불리는 3D CNN을 학습시켰습니다. 이들의 실험에서 3x3x3 filter가 좋은 성능을 보였습니다. Varol은 3D CNN의 temporal length를 늘리는 것은 성능 향상을 보여줬습니다. 이들은 또한 3D CNN의 인풋으로 optical flow를 사용하여 RGB 만을 사용한 것의 성능을 뛰어 넘었습니다. Kay는 Kinetics 데이터에서는 3D CNN이 2D CNN과 경쟁할만 하지만 UCF101, HMDB51에서는 2D CNN이 월등한 성능을 보였습니다. Carreira는 인셉션 구조를 3D CNN으로 만들어 최고 성능을 달성했습니다. 이 논문에서는 이미지 인식에서 인셉션보다 좋은 성능을 보이는 Resnet을 3D CNN 구조로 바꿉니다.

## 3D Residual Network

### 3.1 Network Architecture

우리의 네트워크는 Resnet을 기반으로 합니다. Resnet은 한 레이어의 인풋이 다음으로 바로 연결되는 shortcut 이라는 개념을 도입합니다. 이러한 shorcut pass는 gradient flow가 직접 연결되게하여 깊은 네트워크를 훈련하기 쉽게 합니다. 피규어 1은 residual block을 보여줍니다. shortcut connection은 블록의 위에서 꼬리로 바로 연결됩니다. Resnet은 여러개의 residual block으로 구성되어져 있습니다.

테이블1 은 우리 네트워크의 구조를 보여줍니다. 원래의 Resnet과 다른 점은 conv kernel의 차원과 pooling만이 다릅니다. 우리의 3D Resnet은 3D Conv, pooling을 수행합니다. 커널의 크기는 3x3x3을 사용하고 temporal stride는 1을 사용합니다. 네트워크는 16프레임의 RGB 클립을 인풋으로 받습니다. 인풋 클립은 3x16x112x112 가 됩니다.feature의 차원이 커질때 인풋의 다운샘플링이 수행됩니다. 이는 conv3_1, conv4_1, conv5_1에서 수행되며 stride 2를 통해 다운샘플링합니다. 우리는 zero-padding과 함께 shortcut을 수행하며 이는 파라미터의 수를 줄이기 위해서입니다.

## 3.2 Implementation

### 3.2.1 Training

모멘텀을 사용한 SGD를 통해 네트워크를 훈련시킵니다. 데이터 증강을 위해 비디오로 부터 랜덤하게 트레이닝 샘플을 생성합니다. 먼저 각 샘플의 temporal position을 uniform 하게 샘플링합니다. 16 프레임의 클립은 이 선택된 temporal point에서 생성됩니다. 만약 비디오가 16프레임 보다 적다면 필요한 만큼 비디오를 반복합니다. 그다음 4개의 코너, 중앙 총 5개의 spatial point를 랜덤하게 샘플링하여 각 포인트에서 multi scale crop을 수행합니다. 스케일 1의 의미는 maximum scale을 의미합니다. (프레임의 width, height중 작은 길이). crop된 프레임의 ratio는 1입니다. 생성된 샘플은 random horizonta flip 합니다. 우리는 또한 각 샘플에서 mean subtraction을 수행합니다. 같은 비디오에서 생성된 샘플은 모두 같은 라벨을 가집니다. Kinetics 데이터셋에서 3D Resnet을 학습시키기 위해 우리는 4개의 GPU를 사용했으며 256 배치를 사용했습니다. weight decay는 0.001으로 하고 momentum은 0.9를 줬습니다. 우리는 learning rate를 0.1에서 시작하여 validation loss가 포화될때 10씩 나누며 총 3번의 decay를 합니다. ActivityNet에 대한 예비실험에서 큰 learning rate와 배치사이즈는 성능에 큰 영향을 끼쳤습니다.

- 정리
    - ratio를 유지한채 height가 360이 
    - temporal position을 unifrom 샘플링 각 temporal position에서 16프레임의 비디오를 생성
    - 4개의 코너, 중앙 point에서 multi scale crop을 수행함(정사각형 crop)
    - random horizontal flip
    - mean subtraction
    - learning rate 0.1 val loss가 더이상 안낮아 질때마다 10씩 감소
    - weight decay : 0.001, momentum : 0.9
    - batch size : 256
    
### 3.2.2 Recognition

우리는 학습된 모델을 가지고 비디오에서 action을 인식합니다. 각 비디오를 겹쳐지지 않는 16프레임의 클립으로 나누고 각 클립은 센터를 중심으로 maximum scale로 크롭합니다. 각 클립에 대한 score를 구하고 이들을 평균내어서 비디오 수준에서의 score를 구합니다.

## 4.Experiment

### 4.1 Dataset

우리는 ActivityNet, Kinetics 데이터셋을 사용했습니다. ActivityNet 데이터는 200개의 카테고리와 클래스당 평균 137개의 untrimmed 비디오를 제공하며 비디오당 1.41개의 action이 있습니다. 총 동영상의 길이는 849시간이며 모든 activity 인스턴스의 갯수는 28,108개입니다. 데이터셋은 램덤하게 train 50%, val 25%, test 25% 로 나눠졌습니다.

Kinetics 데이터셋에는 400개의 카테고리와 각 클래스당 400개 이상의 비디오가 존재합니다. 비디오는 temporally trimmed 되어져 액션이 없는 프레임을 포함하지 않습니다. 10초 정도의 영상으로 구성되어져 있습니다. 비디오는 총 300000개로 구성되어져 있으며 train 240000, val 20000, test 40000 개로 구성됩니다. Kinetics의 action 인스턴스의 갯수는 ActivityNet보다 10배정도 많지만 총 비디오의 길이는 비슷합니다. 두 데이터셋 모두 원래의 ratio를 유지한 채 360 height를 가지도록 했습니다. 

### 4.2 Results

먼저 ActivityNet 데이터셋에서의 사전 실험에 대해 기술합니다. 이 실험의 목적은 3D Resnet이 상대적으로 작은 데이터셋에서의 학습에 대해 탐색합니다. 이 실험에서 우리는 18 레이어의 3D Resnet과 Sports-1m 에서 pretrain된 C3D 를 사용합니다. 피규어2는 학습과정에서의 val accuracy를 보여줍니다. 이 accuract는 전체의 비디오가 아닌 16 프레임의 클립에서 계산되어졌습니다. 이 결과는 ActiviyNet 데이터셋은 너무 작아서 3D Resnet을 scratch 부터 교육하기 힘듬을 보여줍니다. pre-trained C3D가 overfit 하지 않고 더 좋은 성능을 보여주는 것을 알 수 있습니다. C3D의 얕은 구조와 pre-train된 모델은 overfitting을 방지합니다.
 
그다음 Kinetics 데이터셋에서의 성능을 보여줍니다. 우리는 34-레이어의 3D Resnet을 학습하였습니다. 피규어3은 학습 동안의 train, val accuracy를 보여줍니다. 이는 클립에서의 accuracy의 계산입니다. 3D Resnet-34는 overfit 하지 않았고 좋은 성능을 달성했습니다. pretrained C3D도 또한 좋은 val accuracy를 달성했지만 낮은 train accuracy를 보였습니다. 게다가 3D Resnet은 pre-train이 없이 C3D와 경쟁할만한 성능을 보엿습니다. 이는 C3D가 너무 얕고 3D Resnet이  효과적이라는 것을 나타냅니다.

테이블 2는 3D Resnet-34와 다른 최신기술과의 비교를 보여줍니다. C3D w/BN은 각 conv 후에 BN을 사용하고 fc layer를 사용합니다. RGB-I3D w/o Imagent은 3D 커널을 기반으로는 인셉션 구조입니다. 여기서는 이미지넷에서의 pretrain이 없는 RGB-I3D w/o 의 결과를 보여줍니다. Resnet-34는 pre-trained C3D와 BN을 사용해 scratch부터 학습한 C3D 보다 좋은 성능을 보입니다. 이 결과는 3D Resnet의 효과를 보여줍니다. 반면 RGB-I3D는 Resnet-34보다 더 얕은 구조지만 가장 좋은 성능을 보였습니다. 이는 사용된 GPU의 갯수 때문일 수 있습니다. 큰 배치사이즈는 BN을 사용하여 학습할대 매우 중요합니다. Carreira는 32GPU를 사용한 반면 우리는 4개의 GPU를 사용했습니다. 매우 큰 배치사이즈를 학습에 사용했고 최고 성능을 보였을 것입니다. 다른 이유는 Input clip의 크기 때문일 수 있습니다. 3D Resnet의 인풋은 메모리의 제약 때문에 3x16x112x112 이였지만 RGB-I3D는 3x64x224x224를 사용합니다. 높은 spatail 해상도와 긴 temporal 은 성능을 향상시킵니다 그러므로 우리의 3D Resnet 또한 더 큰 배치사이즈, 더 큰 clip을 사용하면 성능이 향상 될 것입니다.

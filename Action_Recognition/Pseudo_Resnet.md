# Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks

### Abstract

CNN은 이미지 인식 문제의 강한 모델로 간주되어 왔습니다. 그럼에도 불구하고 spatio-temporal 비디오 representation을 배우는 것은 2D이미지에 비해 명확하지 않습니다. 몇몇의 연구는 3D conv가 spaital, temporal 정보를 배우는것을 보여줬습니다. 그러나 매우 깊은 3D CNN을 scratch부터 학습시키는 것은 매우 비싼 연산과 메모리가 듭니다. 이에 유효한 질문은 3D CNN을 위해서 2D 네트워크를 재활용 하자는 것입니다. 이 논문에서는 우리는 residual learning에서의 몇몇 bottleneck 구조의 변형을 고안합니다. 3x3x3 필터를 spatial 도메인에서의 1x3x3 필터와 인접한 feature map의 3x1x1 필터로 구성합니다. 또한 우리는 P3D Resnet이라는 새로운 구조를 제안합니다. 이는 모든 블록의 변형(3가지)를 이용하지만 Resnet의 다른 부분에서 위치합니다. 이는 구조적 다양성을 향상시키며 깊어지는 것은 뉴럴네트워크의 파워를 향상시킨다는 resnet의 철학을 따릅니다. 우리의 P3D Resnet은 Sport-1m 비디오 classfication 데이터에서 3d cnn보다 5.3% 2d cnn보다 1.8% 향상된 결과를 보였습니다. 우리는 비디오 representation의 일반적 성능을 실험하기 위해 우리의 pre-trained P3D Resnet을 3가지 task의 5개 benchmark를 수행하였고 최첨단 기술보다 좋은 성능을 보였습니다.

### Introduction

오늘날의 디지털 컨텐츠는 텍스트, 오디오, 이미지, 비디오 등 본질적으로 멀티미디어입니다. 센서가 달린 모바일 디바이스의 확산으로 이미지와 비디오는 새로운 communication 수단이 되었습니다. 이로 인해 다양한 멀티미디어의 어플리케이션을 이해를 위한 진보된 기술이 필요해졌습니다. 이러한 기술적 진보의 성공의 근간을 이루는 근본적인 것은 representation learning 입니다. 최근에 CNN은 특히 이미지 영역에서의 visual representation을 배우는 능력을 보여줍니다. 예를들어 resnet의 앙상블은 imagenet에서 3.57% top-5 에러를 달성했으며 이는 인간의 수준보다 5.1% 정도 더 좋은 성능입니다. 그럼에도 비디오는 큰 variation, complexities로 인해 일반적이고 강력한 spatio-temporal representation을 배우는 것이 어렵습니다.

비디오에서 spatio-temporal 정보를 배우는 방법중 하나는 CNN의 커널을 2D에서 3D로 확장하는 것입니다. 네트워크는 현재 비디오 프레임에서의 visual apperance 과 연속적인 프레임에서의 temporal evolution을 배울 수 있습니다. 최근 [8, 31, 33] 과 같은 연구에서 성능의 향상이 보고되었지만 3D CNN을 학습하는 것은 매우 비싸고 2D CNN과 비교하여 quadratic 한 모델사이즈를 가집니다. 널리 채택된 11-레이어의 3D CNN(C3D) 네트워크는 모델사이즈는 321MB이며 이는 152-레이어 2D Resnet 보다도 큽니다. 즉 깊은 레이어를 학습시키는 것은 매우 어렵습니다. 더 중요한 것은 Sports-1M 데이터셋에서 ResNet-152의 fine-tuning한 것이 C3D를 사용한 것보다 성능이 좋았습니다. spatio-temporal 비디오 representation을 위한 다른 대안은 다른 pooling을 사용하거나 RNN을 사용하는 것입니다. 이는 종종 2D CNN의 마지막 pooling의 activation 혹은 fc layer에 연결되어집니다. 그러나 이러한 접근방법은 높은-레벨의 feature에서의 temporal connection만을 만들 수 있고 낮은 레벨의 feature의 상관관계는 사용하지 못합니다.

우리는 위와 같은 제한들을 spatial, temporal filter를 사용하는 몇몇의 bottleneck 구조를 통해 해결할 수 있다는 것을 입증합니다. 특히 각 block의 구성요소들은 하나는 1x3x3 conv, 하나는 3x1x1 conv 의 조합으로 구성되고 병렬적이거나 cascaded 하게 합쳐집니다. 이는 표준적인 3x3x3 conv를 대체합니다. 이를 통해 모델의 크기는 매우 줄어들게 되며 이미지 분야에서 pre-trained된 2D CNN의 장점을 가져올 수 있습니다. 이는 1x3x3 conv filter를 2d CNN의 3x3 filter로 초기화 하는 것으로 가능합니다. 또한 우리는 새로운 P3D-Resnet을 제안하고 이는 네트워크의 구조적 다양성을 증가시키기 위해 resnet과 같은 구조의 전반에 걸쳐 다른 곳에 배치됩니다. 이 결과 우리의 P3D Resnet의 temporal connection은 낮은 레벨부터 높은 레벨까지 모든 레벨에서 구성되며 학습된 video representation은 객체, 장면, 행동과 관련된 정보를 캡슐화 하며 다양한 비디오 분석에 일반적으로 만듭니다.

이 연구의 주된 기여는 3D conv를 효과적으로 수행하는 bottleneck 구조를 제안하는 것입니다. 이것은 깊은 네트워크를 학습시키기위해 다양한 블록들을 어떻게 배치해야 할지에 대한 견해를 제공하며 비디오 representation learning을 위한 새로운 P3D Resnet 을 제안합니다. 광범위한 실험을 통해 우리의 P3D Resnet이 3가지 task의 5가지 benchmark에서 최신 모델을 능가했습니다.

### 2. Related Work

video representation learning을 위한 방법을 크게 hand-crafted, deep learing 의 2가지로 묶습니다. 

Hand-crafted 방법은 spatio-temporal interest point를 찾는 것으로 시작하고 이들 point들을 local representation으로 기술합니다. STIP, HOG, Optical Flow, 3D HOG, SIFT-3D 등의 이러한 방법은 3D volume에서의 temporal 차원으로부터 representation을 확장함으로써 제안되었습니다. 최근데 Wang은 dense trajectory feature를 제안합니다. 이는 다른 스케일로 각 프레임의 local patch를 densely sample 하고 이들을 dense optical flow에서 추적합니다.

최근의 video representation을 위한 방법은 딥러닝 구조에서 고안되었습니다. Karparthy는 고정된 크기의 window에서 프레임 단위로 CNN의 representatoin을 쌓은 다음 video representation 학습을 위한 spatio-temporal conv를 활용합니다. [25] 에서는 visual frame와 stacked optical flow를 인풋으로 받는 개별적인 2개의 CNN구조를 활용한 two-stream 구조를 제안합니다. 이 구조는 muilti-granular 구조, convolution fusion, key-volume mining, temporal segment network 등을 활용하여 확장됩니다. [35]에서는 trajectory의 중심에 있는 tube에 대한 local convnet의 반응을 pooling하여 video discriptor로 활용합니다. [20] 에서의 fisher vector를 사용하여 local descriptor를 global video representation으로 활용합니다. 최근의 LSTM-RNN 네트워크는 비디오에서의 temporal dynamic을 모델링하기 위해 성공적으로 사용되었습니다. [9,37] 에서 temporal pooing, stacked LSTM 네트워크가 optical flow, image를 합치기 위해 사용되었고 robust video representation을 위한 long-term temporal information을 배웁니다. 또한 [28]에서는 LSTM 인코더, 디코더를 기반으로한 비디오 표현을 제안합니다.

앞서 언급한 딥러닝 기반의 방법들은 비디오를 프레임/optical flow 이미지의 sequence로 생각합니다. 하지만 연속적인 프레임 사이의 temporal evolution은 완전히 이용되지는 않았습니다. 이러한 문제를 해결하기 위해 Ji는 3D CNN을 제안하고 이는 짧은 비디오에서의 spatio-temproal representation을 바로 배우기 위한 초기 연구중 하나입니다. 그 이후 [31] 에서 C3D라는 11-레이어의 3D CNN을 제안하고 이를 16 프레임의 비디오 clip을 학습하기 위해 사용합니다. [33]에서는 100 프레임의 더 긴 clip을 위한 temporal conv가 고안됩니다. 그러나 3D CNN 구조는 매우 비싼 연산과 메모리의 문제 때문에 제한됩니다. 이는 3D CNN을 학습하기 어렵게 합니다. 우리의 방법은 3D conv를 2D spatial conv와 1D temporal connecton을 통해 연산하며 이는 더 경제적입니다. 또한 비디오 표현을 위한 깊은 residual 구조에 이러한 것을 통합했습니다.

### 3.P3D Blocks and P3D Resnet

이섹션에서는 spatial 정보를 포착하는 2D spatial conv와 temporal 정보를 위한 1D temporal conv 필터로 나눠질 수 있는 3D convolution을 정의합니다. 그 후 P3D 라고 이름 붙여진 몇몇의 bottleneck 구조를 소개합니다. 마지막으로 우리는 새로운 P3D Resnet이라는 Resnet 구조의 각 위치에 다른 P3D block을 사용한 네트워크를 소개합니다. 또한 성능 및 시간 효율에 대한 연구를 위한 몇몇의 실험을 비교합니다.

### 3.1 3D Convolution

CxLxHxW 의 크기를 가지는 비디오 clip이 주어지면 c: channel, l : clip length, h : height, w: width 을 나타냅니다. spaito-temporal 정보를 배우기 위한 가장 자연스러운 방법은 3D conv 를 이용하는 것입니다. 3D cov는 2D 필터와 같은 spatial 정보를 모델링하고 동시에 프레임에서의 temporal connection을 구성합니다. 이를 간단히 하기 위해 우리는 3D conv 필터의 크기를 DxKxK 라고 하며 D:temporal depth of kernerl, k : kernel spatial szie 를 나타냅니다. 3x3x3 의 크기를 가진 3D conv 필터가 있따고 가정해 봅시다. spatial 영역에서의 2D CNN과 같은 1x3x3 conv filter와 temporal 영역에서의 1D CNN와 같은 3x1x1 conv filter로 나눠질 수 있습니다. 이러한 나눠진 3D conv는 일종의 Pseudo 3D CNN으로 간주될 수 있고 이는 모델의 상당히 줄여주고 이미지에서의 2D CNN을 사용한 pretrain을 가능하게 합니다. 이는 Pseudo 3D CNN이 2D 이미지로 부터 객체, 장면에 대한 지식을 활용할 수 있도록 합니다.

### 3.2 Psudo-3D blocks

최근의 수많은 이미지 인식 task에서의 Resnet의 성공에 영감을 얻어 우리는 새로운 모듈을 개발했다. P3D 블록이 2D residual unit을 대체하여 비디오에서의 Resnet과 같은 구조를 위한 spaito-temporal 정보를 학습할 수 있습니다.다음으로 우리는 Resnet에서의 residual unit의 기본 디자인을 상기하고 어떻게 P3D block을 고안했는지를 보여줍니다. 각 P3D block에서의 bottleneck 구조를 최종적으로 기술합니다.

<b>Residual Units</b> : Resnet은 Residual Units들의 쌓인 형태로 구성되며 각 Residual Uits은 x_t+1 = h(x_t) + F(x_t) 로 일반화 될 수 있습니다. 여기서 x_t와 x_t+1 은 Residual Unit의 인풋과 아웃풋을 나타내며 h(x_t) = x_t 는 identity 매핑을 나타내고 F는 non-linear residual function 입니다. 따라서 위의 수식은 
(I + F)x_t = x_t + Fxt := x_t + F (x_t) = x_t+1 로 다시 쓰여질 수 있습니다. 여기서 Fx_t 는 x_t에 대해 residual function을 수행한 결과를 나타냅니다. Resnet의 주요 아이디어는 shortcut connection을 통한 x_t를 참조하여 추가적인 residual function만을 학습하는 것입니다. (x_t를 참조하지 않는 non-linear function을 학습하는 것 대신)

<b>P3D Blocks design</b> : Resnet에서의 2D residual unit를 spatio-temporal 정보를 활용할 수 있는 3D 구조를 만들기 위해 우리는 기본적인 Residual Unit을  섹션 3.1에서 제안된 방법을 통해 수정합니다. 이러한 변경은 2가지의 디자인 문제가 연관되어 있습니다. 첫번째는 spatial 차원의 2D 필터와 temporal 차원의 1D 필터는 간접적 혹은 직접적으로 서로에게 영향을 끼칩니다. 이러한 두 타입의 필터들 사이의 직접적인 영향은 2D 필터의 출력이 1D 필터의 인풋이라는 것을 의미합니다. (cascade manner) 두 필터사이의 간접적인 영향은 각 종류의 필터가 네트워크의 다른 경로에 있는 것을 의미합니다. (parallel fashion). 두번째 문제는 두 종류의 필터가 마지막의 출력에 직접적으로 영향을 주는지에 대한 것입니다. 즉 마지막 아웃풋에는 두 필터의 아웃풋이 직접적으로 영향을 미쳐야 한다는 것입니다.

위의 두가지 디자인 이슈를 기반으로 우리는 3가지의 P3D block을 고안하였고 각각 P3D-A 부터 P3D-C 라고 부릅니다. 구조에 대한 자세한 비교는 다음에 기술합니다.

<b>P3D-A</b> : 첫번재 디자인은 spatial 2D 필터 이후에 temporal 1D 필터가 따륻록 합니다. 따라서 두가지 종류의 필터는 네트워크의 가은 경로에서 서로가 영향을 받을 수 있고 temporal 1D 필터만이 마지막 아웃풋에 직접적으로 연결됩니다. 

<b>P3D-B</b> : 두번째 디자인은 첫번째 디자인과 매우 비슷합니다. parallel fashion으로 두가지 필터가 다른 경로에 있으며 서로 간접적 영향을 끼칩니다. S와 T 사이에 직접적인 영향은 없지만 마지막 출력에 직접적으로 누적됩니다. 

<b>P3D-C</b> : 마지막 디자인은 P3D-A와 P3D-B의 중간의 디자인으로써 S,T, 마지막 아웃풋 간의 직접적인 영향을 받습니다. 특히 계단식인 P3D-A 구조를 기반으로 S와 마지막 아웃풋간의 직접적 연결을 가능하게 합니다. 마지막 출력이 밑의 수식과 같도록 만들기 위해 S로 부터 출력 간의 shortcut connection을 추가합니다.

<b>Bottleneck architectures</b> : 2D Residual Unit의 구조를 만들때 계산 복잡도를 줄이기 위해 기본적인 2D block은 bottleneck 구조로 변경됩니다. 피규어3에서 보듯이 하나의 3x3 spatial 2D 필터를 사용하는 것 대신 1x1, 3x3, 1x1 conv가 쌓인 구조가 됩니다. 첫번째와 마지막의 1x1 conv 레이어는 인풋의 차원을 줄이거나 복원하는 용도로 적용이 됩니다. 이러한 bottleneck 디자인으로 가운데의 3x3 conv를 작은 인풋, 출력을 가능하게 합니다. 따라서 우리는 이러한 디자인을 따르고 bottleneck 구조를 활용하여 P3D block을 구현합니다. 위와 비슷하게 하나의 P3D block은 하나의 spatial 2D 필터(1x3x3), 하나의 temporal 1D 필터 (3x1x1)과 양쪽의 1x1x1 conv로 구성됩니다. 1x1x1 conv는 인풋, 출력의 차원을 줄이고 복원하는 것을 담당합니다. 따라서 spatial 2D, temporal 1D 필터의 입력과 출력의 차원은 bottleneck 구조를 통해 감소됩니다. 이러한 bottleneck 구조의 세부사항은 피규어 3에 나와있습니다.

### 3.3 Pseudo-3D Resnet

이러한 3개로 구성한 P3D block의 장점을 알아보기 위해 우리는 세가지 P3D Resnet의 변형을 개발했습니다. (Resnet-50의 각 유닛을 한가지 종류의 P3D block으로 교체하여 P3D-A Resnet, P3D-B Resnt, P3D-C Resnet을 구현했습니다.) 기본적인 Resnet-50의 구조와 세가지의 변형에 대한 성능과 시간 효율에 대한 비교를 제시했습니다. 그런다음 구조적 다양성의 관점에서 3가지 P3D block을 혼합한 완전한 버젼의 P3D Resnet을 제안합니다.

<b> Comparisons between P3D block Resnet variant</b> : 비교는 action recognition dataset인 UCF101에서 수행됩니다. 특히 Resnet-50의 구조는 UCF101 데이터에서 fine-tuning 하였습니다. 우리는 인풋을 224x224 이미지로 설정했으며 이는 resize한 240x320 비디오 프레임에서 랜덤하게 크롭되어집니다. 게다가 [36] 에 따라서 첫번째 BN 레이어를 제외한 파라미터를 고정시키고 오버피팅의 효과를 줄이기 위해 추가적인 0.9 rate의 dropout 레이어를 추가합니다. Resnet-50을 fine-tuning 후에 네트워크는 각 프레임마다 하나의 score를 예측하고 비디오 레벨에서의 예측은 모든 프레임 레벨의 score를 평균냅니다. P3D Resnet의 변형들은 temporal conv를 제외하고 Resnet-50의 웨이트로 초기화되었고 UCF101에서 fine-tuning 했습니다. 각각의 P3D Resnet의 변형들마다 인풋 비디오 클립은 16x160x160이며 이는 16x182x242로 리사이즈된 겹쳐지지 않은 16-frame 클립에서 랜덤하게 크롭된 것입니다. 각각의 프레임/clip은 데이터증강을 위해 random horizontal flip 합니다. 우리는 128 프레임을 미니배치로 설정하였고 여러 gpu에서 병렬적으로 구현했습니다. 기본적인 SGD를 사용하여 파라미터를 조정하였고 초기 learing rate는 0.001 입니다. 이는 3K iteration마다 10씩 줄어듭니다. 총 7.5K 이후 학습이 중단됩니다.

Method |Model size |Speed |Accuracy
--- | --- | --- | --- 
ResNet-50 |92MB |15.0 frame/s |80.8%
P3D-A ResNet| 98MB |9.0 clip/s |83.7%
P3D-B ResNet| 98MB |8.8 clip/s |82.8%
P3D-C ResNet| 98MB |8.6 clip/s |83.0%
P3D ResNet| 98MB |8.8 clip/s |84.2%

위의 테이블은 UCF101 에서의 Resnet-50과 P3D Resnet의 변형들의 성능및 시간효율을 보여줍니다. 전반적으로 세가지의 변형들이 Resnet-50보다 나은 성능을 보였고 모델 사이즈는 약간 큽니다. 이러한 결과들은 P3D block을 통한 spatio-temporal 정보의 추가의 이득을 나타냅니다. 또한 P3D의 변종들은 클립당 8.6~9.0 의 속도를 보입니다.
 
 <b> Mixing different P3D block</b> : 최근 깊은 네트워크의 디자인에서 구조적 다양성의 성공에 영감을 얻어 우리는 P3D Resnet의 완전한 버젼을 고안했습니다. 이는 다른 P3D block을 네트워크 구조안에서 섞어 구조적 다양성을 향상 시킵니다. 특히 우리는 Residual Units을 P3D-A -> P3D-B -> P3D-C 의 순서로 구성합니다. 테이블 1은 완전한 버젼의 P3D Resnet의 성능과 속도를 보여줍니다. 구조적 다양성을 추가함으로써 P3D Resnet은 다른 변종에 비해 각각 0.5%(A), 1.4%(B), 1.2%(C)의 성능 향상을 보였습니다. 깊은 네트워크 구조에서 구조적 다양성을 추구하는 것은 네트워크의 성능을 향상시킬 수 있음을 나타냅니다.
 
 ### 4. Spatio-Temporal Representation Learning
 
 우리는 P3D Resnet의 디자인을 152-Resnet에서 검증한 다음 일반적인 spatio-temporal video representation을 생성합니다. P3D Resnet의 학습은 Sprots-1M 데이터셋에서 수행되었으며 이는 가장 큰 비디오 classfication benchmark중 하나입니다. 대략 487개의 라벨이 붙은 약 1백13만개의 동영상이 있고 동영상의 약 5% 는 하나 이상의 라벨이 있습니다. 우리가 비디오를 다운로드 할대 9.2%의 비디오의 URL이 죽어 나머지 1020만개의 동영상에 대한 실험을 실시하였고 70%, 10%, 20%를 traning, validation, test 셋으로 사용했습니다.
 
 <b>Network Traning</b> : 매우 큰 Sport-1M 트레이닝 세트에 대해 효율적인 학습을 위해 우리는 각 비디오에서의 5초의 영상을 무작위로 선택합니다. 학습동안 데이터 증강과 미니배치는 섹션 3.3과 같습니다 단 dropout rate는 0.1로 하였습니다. learning rate는 0.001로 초기화되어 60k iteration 마다 10씩 줄어듭니다. 150k 배치후 학습이 완료됩니다.
 
 <b>Network Testing</b> : 우리는 학습된 P3D Resnet의 성능을 테스트 셋에서의 video/clip classfication accuract로 측정했습니다. 특히, 우리는 각 비디오에서 랜덤하게 20 clip을 샘플링하고 각 clip마다 center clip을 하였습니다. 이는 네트워크를 통해 forward되어 클립당 예측 score를 얻습니다. 비디오-레벨의 score는 비디오 안의 모든 clip의 score를 평균내어 계산합니다.
 
 - 하이퍼 파라미터 정리
    - 데이터 : 70%, 10%, 20% 를 train, val, test 셋으로 사용
    - 데이터 샘플 : 각 비디오에서 5초의 영상을 무작위로 선택
    - data argu : random crop, random flip
    - 인풋 :  비디오 클립은 16x160x160이며 이는 16x182x242로 리사이즈된 겹쳐지지 않은 16-frame 클립에서 랜덤하게 크롭된 것입니다.
    - optimizer : SGD with lr 0.001 decay each 60k iteration
    - BN은 첫번째 빼고 고정하고 추가적인 dropout 레이어 사용
    - 미니배치 : 128 clip
    
우리는 성능 비교를 위해 다음과 같은 접근방식을 비교합니다. (1) Deep video(Single frame) and Slow fusion. 클립레벨의 score를 예측하기 위해 [14] 에서와 비슷한 구조로 실행합니다. 이는 다른 temporal의 각 프레임을 fusion하고 네트워크를 통해 클립 수준의 예측을 합니다. (2) Convolutional Pooling 은 클립의 프레임간의 마지막 conv 레이어를 위해 맥스풀링을 사용합니다. (3) C3D 는 클립 volume에서 3D conv를 사용하여 시간적 정보를 모델링하고 전체 구조는 Sports-1M 데이터셋을 통해 처음부터 학습되거나 I380K 의 pre-train 모델에서 fine-tuning 할 수 있었습니다. (4) Resnet-152 Resent은 fine-tuning 했으며 클립 레벨의 score를 생성하기 위해 클립안의 각각 프레임에 대해 적용됩니다.

Method |Pre-train Data |Clip Length| Clip hit@1 |Video hit@1 |Video hit@5
--- | --- | --- | --- | --- | ---
Deep Video (Single Frame) [10]| ImageNet1K| 1 |41.1% |59.3% |77.7%
Deep Video (Slow Fusion) [10] |ImageNet1K |10 |41.9% |60.9% |80.2%
Convolutional Pooling [37] |ImageNet1K |120 |70.8% |72.3% |90.8%
C3D [31]| – |16 |44.9% |60.0% |84.4%
C3D [31]| I380K| 16| 46.1%| 61.1%| 85.2%
ResNet-152 [7] |ImageNet1K| 1| 46.5%| 64.6%| 86.4%
P3D ResNet (ours)| ImageNet1K| 16 |47.9% |66.4% |87.4%

성능과 비교는 테이블 2에 요약되어져 있습니다. 전반적으로 P3D Resnet은 Renet-142, C3D에 대한 top-1 video-level accuracy를 1.8% 5.3% 향상시킵니다. 결과는 기본적으로 spatio-temporal 정보를 활용하는 것에 대한 이점을 나타냅니다. 기대한대로 Deep video(Slow fusion) 은 하나의 단일 프레임만을 이용하는 Deep video(single frame) 보다 우수한 성능을 나타냅니다. deep video, covolutional pooling , P3D Resnet의 세가지 모두 temporal fusion을 사용하지만 이들은 기본적으로 어떻게 temporal connection을 구성하는지가 다릅니다. deep video의 성능은 activation을 계산하기 위해 temporal convolution을 수행한 결과이며 Convolutional pooling은 temporal 프레임간의 마지막 conv 레이어의 결과를 간단히 maxpooling한 것입니다. 결과에서 알 수 있듯이 우리의 P3D Resnet은 spatial, temporal conv의 다양한 조합을 사용하면서 deep video(딥비디오도 resnet 구조를 사용하나보다)의 성능을 향상시킵니다. 이는 우리의 P3D Resnet이 네트워크 설계의 구조적 다양성의 혜택을 받는다는 것을 의미합니다. P3D Resnet의 성능이 convolutional pooling 보다 낮다는 사실은 놀랍지 않습니다. convolutional pooling은 frame rate를 1fps로 하여 하나의 클립을 120 프레임에서 temporal pooling을 수행합니다. 이와 반대로 우리는 연속적인 16 프레임을 기본단위로 사용하여 0.5초 미만으로 처리하지만 강한 spatio-temporal connection을 가집니다. 즉 우리의 P3D Resnet이 더 일반화 성능이 좋습니다.

피규어5는 학습된 P3D Resnet 모델을 시각화 합니다 [36]에 의해 우리는 DeepDraw toolbox를 적용하였고 이는 white noise의 인풋 클립을 gradient ascent합니다. 학습 동안 모델의 클래스 라벨의 오류를 평가하고 back-prop을 통해 white noise 인풋 클립을 업데이트합니다. 그래서 마지막 생성된 input clip은 내부적 클래스에 대한 지식을 시각화 한것으로 여길 수 있습니다. 우리는 3가지의 카테고리를 선택하여 시각화 했습니다. 그림에서 알 수 있듯이 P3D Resnet은 spatial, visual 패턴과 temporal motion들을 포착합니다. tai chi 카테고리를 살펴보면 우리의 모델은 사람이 다른 포즈를 취하는 비디오 클립을 생성하여 action의 과정을 묘사합니다.

<b> P3D Resnet Representation</b> Sprots-1M 데이터셋에서의 P3D Resnet 구조의 학습을 마친 후 네트워크는 어떠한 비디오 분석에 적용가능한 일반적인 representation extractor로 사용될 수 있습니다. 우리는 16프레임의 20개의 비디오 클립을 선택하고 각 비디오 클립은 P3D Resnet 구조에 입력되어 2048 차원의 activation이 이 클립의 representation으로 출력됩니다. 마지막으로 클립 수준의 representation은 평균되어져 2048 차원의 video representation을 생성합니다. 우리는 별도의 명시가 없는 경우 비디오 representation을 다음의 평가에 사용합니다.


### 5. Video representation Evaluation

다음으로 우리는 P3D Resnet의 비디오 representation을 5가지 데이터셋의 3가지 task에 대해 평가합니다. 사용한 데이터셋은 UCF101, ActivityNet, ASLAN, YUPENN, Dynamic Scene 등이 있고 이중 UCF101, ActivityNet은 video action recognition 벤치마크에서 가장 유명한 데이터입니다. UCF101은 13320개의 비디오로 구성되며 101개의 action 카테고리로 구성되어져 있습니다. UCF101은 3가지의 train/test 데이터 세트를 제공하며 UCF101의 각 데이터는 9.5K traning, 3.7k test 비디오를 제공합니다. ActivityNet 데이터셋은 인간의 activity를 이해하기 위한 대규모 비디오 데이터셋입니다. 데이터셋의 가장최신(v.13) 버젼이 나왔으며 이는 19994 비디오와 200개의 activity 카테고리를 가지고 있습니다. 19994개의 비디오는 10024, 4926, 5044 개의 비디오로 각각 train/val/test set으로 나눠집니다. 테스트셋의 라벨이 공개되지 않았음으로 ActivityNet 데이터셋의 성능은 validation 세트에서 보고됩니다.

ASLAN은 action간의 비디오간의 유사성을 예측하는 action similarity 를 포함한 데이터셋입니다. 이 데이터에는 3697 비디오와 432 action 카테고리를 포함합니다. 우리는 저자들이 제공하는 공식 10-fold cv 전략을 따랐습니다. 또한 YUPENN과 Dynamic Scence은 장면인식을 위한 데이터입니다. YUPENN은 30개의 비디오로 이루어져있고 14개의 scene 카테고리로 구성됩니다. Dynamic Scene은 클래스당 10개의 비디오와 13개의 scene class로 구성됩니다. 이 데이터셋은 표준적인 leave-one-video-out 방법에 따라 train/test 셋을 나눕니다.

테이블3)  

End-to-end CNN architecture with fine-tuning

Method | Accuracy
---|---
Two-stream ConvNet [25] |73.0% (88.0%)
Factorized ST-ConvNet [29]| 71.3% (88.1%)
Two-stream + LSTM [37] |82.6% (88.6%)
Two-stream fusion [6] |82.6% (92.5%)
Long-term temporal ConvNet [33]| 82.4% (91.7%)
Key-volume mining CNN [39] |84.5% (93.1%)
ST-ResNet [4] |82.2% (93.4%)
TSN [36] |85.7% (94.0%)

CNN-based representation extractor + linear SVM

Method | Accuracy
---|---
C3D [31] |82.3%
ResNet-152 |83.5%
P3D ResNet |88.6%

Method fusion with IDT

Method | Accuracy
---|---
IDT [34] |85.9%
C3D + IDT [31] |90.4%
TDD + IDT [35] |91.5%
ResNet-152 + IDT |92.0%
P3D ResNet + IDT |93.7%

<b>Comparision with the state-of-the-art</b> 우리는 먼저 UCF101과 ActivityNet validation set에서 몇몇의 최신 기술과 비교합니다. 성능 비교는 테이블 3,4에 요약되어져 있습니다. 우리는 UCF101에 대한 접근방법을 3가지 범주로 그룹화 합니다. UCF101에서 fine-tune 된 end-to-end CNN 구조, CNN 기반의 feature extractore와 SVM classfier를 사용하는 방법은 IDT와 함께 하단 행에 기술했습니다. UCF101 에서 사용되는 최근의 end-to-end CNN 구조는 종종 optical flow, audio 와 같은 1개 이상의 인풋을 혼합합니다. 따라서 비디오 프레임만 활용하는 것과 optical flow와 비디오 프레임의 두종류의 인풋을 받는 것(IDT 을 말하는듯) 모두 보고합니다. 

테이블 3을 살펴보면 P3D Resnet의 성능이 88.6% 를 달성하여 TSN, Resnet-152보다 2.9%, 5.1%의 향사을 보였습니다. LSTM을 사용하는 [37] 과 비교할 때 P3D Resnet은 전체 구조를 통한 temporal connection(낮은수준부터 높은 수준까지 모두 temporal connection이 존재한다.)의 효과를 얻어 [37] 을 능가했습니다. optical flow와 하나의 프레임의 정보를 융합해 사용한 P3D Resnet은 여전히 [25, 29, 37] 를 능가했습니다. 괄호안의 성능으로 나타낸 두가지 종류의 인풋을 합친 결과는 비디오 프레임만 사용하는 것과 비교할때 명백한 개선효과가 있는 것으로 나타납니다. 이것은 앞으로의 연구에서 P3D Resnet 구조를 다른 입력과 함께 학습하도록 하는 동기를 부여합니다. 또한 2D spatial conv와 1D temporal conv를 사용하느 P3D Resnet은 3D conv를 사용하는 C3D보다 좋은 성능을 보입니다. 

handcrafted feature인 IDT를 결합하면 93.7% 까지 성능이 향상됩니다. 또한 최근의 최신 인코딩 방법인 [22] 를 사용해 P3D Resnet에서의 res5c 레이어의 activation을 인코딩하면 90.5% 의 성능을 달성할 수 있습니다. 이는 global representation을 통해 1.9% 정도의 성능 향상을 얻은 것입니다.

Method |Top-1| Top-3| MAP
---| ---| ---|---
IDT [34] |64.70% |77.98%| 68.69%
C3D [31] |65.80%| 81.16%| 67.68%
VGG 19 [26] |66.59%| 82.70%| 70.22%
ResNet-152 [7]| 71.43%| 86.45%| 76.56%
P3D ResNet |75.12% |87.71% |78.86%

여러 evaluation metric에 대한 결과는 P3D Resnet에서 얻어진 video representation이 기준선에 비해 성능이 향상되었음을 나타냅니다. 테이블 4에서 볼 수 있듯이 P3D Resnet은 IDT, C3D, VGG 19, Resnet-152 를 10.4%, 9.3%, 8.5%, 3.7% 만큼 능가했습니다. C3D와 Resnet-152 사이에는 큰 성능차이가 있는데 이것은 ActivityNet의 카테고리가 일상생활에서의 인간의 행동에 대한 데이터이기 때문입니다. ActivityNet의 데이터는 Sports-1M 데이터와 매우 다르고 SPorts-1M 데이터에서 학습된 C3D는 낮은 성능을 보였습니다. 대신 이미지넷에서 학습된 Resnet-152는 이러한 경우 도움이 됩니다. P3D Resnet은 이미지 데이터로부터의 pre-trained 2d spatial conv를 사용하고 비디오에서의 temporal 정보를 위한 1D temporal conv를 학습하여 시공간 영역에서의 영역을 완전히 활용해 성능을 향상시킵니다.

Method |Model |Accuracy |AUC
--- | --- | --- | ---
STIP [13]| linear| 60.9%| 65.3%
MIP [12] |metric |65.5% |71.9%
IDT+FV [19]| metric |68.7%| 75.4%
C3D [31]| linear |78.3% |86.5%
ResNet-152 [7]| linear| 70.4%| 77.4%
P3D ResNet |linear |80.8% |87.9%

두번째 task는 action similarity 라벨링 과제이며 이는한쌍의 비디오가 같은 action을 나타내는지를 예측합니다. [13, 31]의 세팅에 따라 우리는 P3D Resnet의 4개의 레이어에서 feature를 추출하고(16-프레임 클립에서의 prob, pool5, res5c, res4b) 비디오 수준에서의 representation은 모든 클립 수준의 representation을 평균하여 얻어집니다. 한쌍의 비디오가 주어졌을때 이 4가지 type이 representation에서 12개의 유사성을 계산합니다. 그래서 결국 한쌍의 비디오에서 38 차원의 벡터를 생성합니다. 이 48 vector에서 L2 정규화를 수행하고 Linear SVM을 이용해 classfier 합니다. ASLAN에서의 성능비교는 테이블 5에 나와있습니다. 전반적으로 P3D Resnet은 hand-crafted feature와 CNN 기반의 모델보다 AUC, accuracy가 뛰어납니다. 일반적으로, CNN 기반의 representation은 hand-crafted feature보다 정확합니다. action recognition task와 다르게 action similarity 에서는 C3D가 Resnet-152보다 뛰어난 성능을 보였습니다. 이것이 단일 이미지에서 학습된 Resnet-152 모델이 비디오간의 유사성을 알기에는 어려움이 있었을 거라고 추측합니다. 비디오 데이터로 부터 학습된 C3D는 비디오를 구별하는데 더 높은 능력을 갖습니다. P3D Resnet이 Resnet-152와 C3D의 장점을 모두 가집니다.

테이블6)

Method |Dynamic Scene| YUPENN
--- | --- | ---
[3] |43.1% |80.7%
[5] |77.7% |96.2%
C3D [31] |87.7% |98.1%
ResNet-152 [7] |93.6% |99.2%
P3D ResNet |94.6% |99.5%

세번째 실험은 scene recognition에 대해 수행되어졌습니다. 테이블6은 다양한 방법에 대한 성능을 보여줍니다. P3D Resnet은 YUPENN, Dynamic Scene benchmark에서 hand-crafted feature 방법보다 16.9% 3.3% 높은 성능을 보입니다. C3D와 Resnet-152과는 YUPENN에서 1.4% 0.3% 의 성능향상을 보였습니다.

<b> The effect of representation dimension </b>

피규어 6은 IDT, Resnet-152, C3D, P3D Resnet의 PCA의 결과를 나타냅니다. 전반적으로, P3D Resnet에 의해 학습된 video representation은 500차원부터 10차원(축소한 차원을 말하는듯?)까지 다른 방법들보다 우월합니다. 일반적으로 높은 차원의 representation은 더 좋은 accuract를 제공합니다. 차원의 수를 줄이면 Resnet-152의 성능이 C3D, P3D Resnet보다 급격하게 감소합니다. 이것은 비디오 표현을 생성하기 위한 Resnet-152의 약점을 나타냅니다. Resnet-152는 순수하게 이미지에서 학습되었기 때문에 비디오 도메인과이 차이가 있어 차원의 수가 낮을 때 representational capability를 낮출 수 있습니다. P3D Resnet은 video, image 도메인에서의 지식을 모두 학습하여 video representation의 차원의 변화에 robust하는 이점이 있습니다.

<b> Video representation embedding visualization </b> 피규어 7은 비디오 representation의 T-SNE 시각화를 보여줍니다. 우리는 UCF101에서 10K의 랜덤한 비디오를 선택하고 이를 T-SNE를 사용해 2차원으로 프로젝션합니다. P3D Resnet의 representation이 Resnet-152보다 의마적으로 잘 분리된것을 볼 수 있습니다.

### Conclusion

우리는 딥 네트워크에서 시공간적 비디오 표현을 배우는 P3D Resnet 구조를 발표했다. 특히 spatial 2D 필터와 1D temporal connection을 사용하여 3D conv를 단순화하는 문제를 연구했습니다. 우리의 주장을 검증하기 위해 우리는 3가지의 bottleneck 구조를 고안하였고 구조 다양성을 위해 다양한 위치에 이들을 배치했습니다. P3D Resnet 구조는 Sports-1M 데이터셋에서 학습되어졌고 우리의 제안과 분석의 타당성을 보여줍니다. action recognition, action similarity labeling, scene recognition의 3개의 task에 대한 5개의 데이터셋에서 수행된 실험을 우리의 P3D Resnet의 효율성과 일반화를 보여줍니다. 다른 feature learning 기술과 비교했을때 성능향상은 분명하게 보여집니다.

우리의 미래의 연구는 다음과 같습니다. 첫번째로 우리의 P3D Resnet의 representaion을 향상시키기 위해 어텐션 메카니즘을 통합할 것입니다. 둘째로, clip안의 프레임의 수를 증가 했을때 P3D Resnet의 성능이 어떻게 될지에 대해 연구 할 것이며 세번째로 optical flow, audio 와 같은 다른 인풋을 사용할 수 있도록 P3D Resnet을 확장할 것입니다.
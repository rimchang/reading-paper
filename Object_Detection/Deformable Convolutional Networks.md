# Deformable Convolutional Networks

### Abstract

CNN은 본질적으로 fixed geometric structure(rectangle한 구조 때문?) 때문에 geometric transformation을 모델링 하는 것이 제한됩니다. 이 논문에서는 CNN의 transformation modeling capacity를 향사이키기 위해 두가지 모듈을 소개하며 각각 deformable convolutiona, deformable ROI-pooling이라고 불립니다. 두 아이디어 모두 spatial smapling location을 augment 하는 것에 기반하고 있습니다. 이들은 추가적인 offset을 도입하고 target task로 부터 offset을 학습함으로써 달성하며 offset을 학습하는데는 추가적인 supervision이 필요하지 않습니다. 새로운 모듈은 기존의 CNN을 쉽게 대체 할 수 있으며 표준적인 backprop에 의해 end-to-end로 학습가능합니다. 이를 통해 deforamble convolutional network를 얻을 수 있습니다. object detection, semantic segmentation 와 같은 task에 대한 실험을 통해 우리의 접근 방법의 유효성을 검증합니다. 코드 또한 공개되어져 있습니다.

### 1. Introduction

visual recognition에서의 주요한 과제는 object의 scale, pose, viewpoint, part deformation에 따른 geometric variation, geometric transformation 을 모델링 하는 것입니다. 일반적으로 두가지 방법이 존재합니다. 1. 첫번째는 학습데이터가 충분한 variation을 가지도록 하는 것입니다. 이는 이미 가지고 있는 data sample을 affine transformation 과 같은 augment을 통해 달성됩니다. roubst representation은 이러한 데이터로 부터 학습될 수 있지만 이는 학습을 힘들게 하며 model parameter를 복잡하게 만듭니다. 두번째는 transformation-invariant feature와 알고리즘을 사용하는 것입니다. 이러한 카테고리는 SIFT(scale invariant feature transform), sliding window based object detection 등의 접근이 잘 알려져 있습니다.

이러한 기존의 방법에는 두가지의 단점이 있습니다. 첫째로 geometric transformation은 고정되어져 있고 알려진 것으로 가정합니다. 이러한 prior knowledge는 data augment에 사용되며 feature와 알고리즘을 디자인할때 사용됩니다. 이러한 가정은 unknwon geometric-transformation을 가진 새로운 task로의 일반화를 방해합니다. 두번째로 invariant feature에 대한 hand-crafted feature, algorithm은 설계하기가 매우 어렵거나 매우 복잡한 transformation을 포함하는 task에 대해서는 불가능합니다.

최근 CNN이 image calssfication, semantic segmentation, object detection과 같은 visual recognition task 에서 큰 성공을 이루었습니다. 그러나 이들은 여전히 위에서 언급한 단점들을 가지고 있습니다. CNN은 data augmentation, larget model capacity, hand-crafted module(max-pooling은 약간의 translation-invariance를 가져다 준다) 을 통해 geometric transformation을 모델링 할 수 있는 능력을 가지게 됩니다.

CNN은 본질적으로 large, unknwon transformation을 모델링 하는 것이 제한됩니다. 이러한 제한은 CNN 모듈의 고정된 geometric 구조 때문에 발생합니다. 이러한 고정된 geometric 구조는 다음과 같은 예들이 있습니다. convolution unit은 input feature map을 고정된 location에서 샘플링 합니다. pooling layer는 고정된 ratio를 통해 spatial resolution을 감소시킵니다. ROI-pooling layer는 ROI를 고정된 spatial bin으로 분리합니다. 이들은 geometric variation을 다룰 수 있는 내부적인 매커니즘이 부족하며 다양한 문제를 야기합니다. 예를 들어 같은 CNN layer안의 모든 activation unit들의 receptive field는 모두 같습니다. 이는 semantic 정보를 인코딩 하는 high level layer 에는 바람직 하지 않습니다. 다른 location의 object는 다른 scale을 가지는 경우가 많기 때문에 scale, receptive field의 크기를 adaptive 하게 결정하는 것은 visual recognition, fine localization에 필수적입니다. 다른 예로는 object detection이 [16, 15, 44, 7, 38] 과 같은 논문에서 빠르게 성장하고 있는 반면 모든 접근 방법은 bounding box 에 기반한 feature extraction을 사용하며 이는 매우 일차원 적인 방법이며 non-rigid object에 대한 sub-optimal 이 됩니다.

이 논문에서는 CNN의 capacity of modeling geometric transformation 을 향상시키는 새로운 모듈을 제안합니다. 첫번째는 deformable convolution 입니다. 이는 표준적인 convolution의 sampling grid에 2D offset을 추가합니다. 이는 피규어 1에 나와있는 것처럼 sampling grid의 자유로운 deformation 형태를 허용하게 됩니다. 이 offset은 추가적인 convolutional layer를 통해 이전의 feature map에서 학습되게 됩니다. 그래서 deformation은 input feature를 조건부로 하여 local, dense, adaptive 방법으로 학습됩니다.

두번째는 deformable ROI-pooling 입니다. 이전의 ROI-poling의 각각의 bin partition 안에서 각각의 bin postion에 대한 offset을 추가합니다. 위에서 언급한 것과 마찬가지로 offset은 이전의 feature map과 ROI를 사용하여 학습되며 differenct shape object 에 대한 adaptive part localization을 가능하게 합니다.

두 모듈 모두 매우 가볍고 offset을 학습하기 위한 약간의 파라미터와 연산량을 추가하게 됩니다. 이들은 다른 CNN 아키텍쳐에 쉽게 통합 될 수 있으며 backprop을 통해 end-to-end 학습이 가능합니다. 이 새로운 CNN은 deformable convnet 이라고 부릅니다.

우리의 방법은 spatial transform network, deformable part model의 아이디어를 공유합니다. 이들은 모두 내부적으로 transformation 파라미터를 가지고 있으며 데이터로 부터 이러한 파라미터를 학습하게 됩니다. 그러나 deformable convnet에서의 주요한 차이점은 간단하고, 효율적이고 end-to-end가 가능한 방법을 통해 dense spatial transformation을 다룬다는 것입니다. 섹션 3.1 에서는 이 논문과 이전의 연구와의 관련에 대해 논의하고 deformable convent의 효율성을 분석합니다.

deformable convnet은 sematic segmentation, object detection의 최신 아키텍쳐에 적용되었습니다. ablation study와 이전의 연구와의 비교는 우리의 접근 방법의 탈성한 성과를 보여줍니다. 처음으로 우리는 CNN에서 dense spatial transformation을 학습하는 것이 가능하고 object detection, segmetation 등에서 효과적이라는 것을 보여줍니다.

### 2. Deformable Convolutional Networks

CNN 에서의 feature map, convolution은 모두 3D 이지만 (C,H,W) deformable convolution, deformable ROI-pooling 연산은 channel domain에서는 같은 wieght를 가지는 2D domain 에서 수행됩니다. 그렇기 때문에 이 논문에서는 2D domain 에서 설명합니다. 이 섹션의 방정식을 3D로 확장하는 것은 매우 간단하며 표기의 간단함을 위해 생략되어졌습니다.

<b>2.1. Deformable Convolution</b>

2D convolutiondms 2단계로 구성됩니다. 1) regular grid R 을 사용하여 input feature map x에 대한 샘플링을 수행합니다 2) w를 통해 sampled value를 weighted summation을 수행합니다. grid R은 receptive field와 dilation을 정의합니다. 예를 들어

  <img src="https://latex.codecogs.com/gif.latex?R%20%3D%20%5Cbegin%7BBmatrix%7D%20%28-1%2C%20-1%29%2C%20%28-1%2C0%29%2C%5Ccdots%20%2C%280%2C1%29%2C%20%281%2C1%29%20%5Cend%7BBmatrix%7D"/>

이 grid R은 3x3 kernel with dilation 1을 정의합니다.

output feature map y의 각각의 location p_o 에 대해 우리는 다음과 같은 식을 얻습니다.

  <img src="https://latex.codecogs.com/gif.latex?y%28p_0%29%20%3D%20%5Csum_%7Bp_n%20%5Cin%20R%7Dw%28p_n%29%5Ccdot%20x%28p_0%20&plus;%20p_n%29"/>

이는 output feature map의 p_o location은 convolution(cross correlation) 에 의해 계산된다는 의미.

deformable convolution은 regular grid R을 offset 
  <img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7BBmatrix%7D%20%5CDelta%20p_n%7Cn%3D1%2C%5Ccdot%2CN%20%5Cend%7BBmatrix%7D%20%5C%2C%5C%2C%20where%20%5C%2C%5C%2C%20N%3D%7CR%7C"/>
을 통해 augment 합니다.

  <img src="https://latex.codecogs.com/gif.latex?y%28p_0%29%20%3D%20%5Csum_%7Bp_n%20%5Cin%20R%7Dw%28p_n%29%5Ccdot%20x%28p_0%20&plus;%20p_n%20&plus;%20%5CDelta%20p_n%29"/>

위의 수식은 iiregular grid에 대해 sampling을 수행하며 그 location은 p_n + delta p_n 으로 결정됩니다. offset delta p_n 이 일반적으로 정수가 아니므로 bilinear interpolation을 통해 다음과 같이 구현됩니다.

  <img src="https://latex.codecogs.com/gif.latex?x%28p%29%20%3D%20%5Csum_pG%28q%2Cp%29%5Ccdot%20x%28q%29"/>

여기서 p는 arbitrary (fractional) location (p = p_0 + p_n + delta p_n) 을 나타내며 q는 feature map x에서 적분되어야 할 모든 spatial location을 나타냅니다. G는 bilinear interpolation kenerl 이며 이는 2차원 입니다. 2차원 kernel은 다음과 같이 2개의 1차원 커널로 분리 할 수 있습니다.

  <img src="https://latex.codecogs.com/gif.latex?G%28q%2C%20p%29%20%3D%20g%28q_x%2C%20p_x%29%5Ccdot%20g%28q_y%2C%20p_y%29"/>

여기서 g(a,b) = max(0 , 1-|a-b|) 를 나타냅니다.








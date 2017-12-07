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

<b>Learning the offset</b> 피규어 2에서 볼 수 있듯이 offset은 input feature map에 conv layer를 적용 함으로써 얻어집니다. convolution 커널은 현재의 convolutional layer와 같은 spatial resolution을 가집니다. (피규어 2에서는 3x3) offset filed의 출력은 input feature map과 같은 spatial resolution을 가집니다. channel 은 2N 이며 이는 N개의 2D offset vector를 인코딩 합니다. 학습동안 이 두개의 convolutional kernel은 output feature를 생성하고 학습 가능한 offset을 생성합니다. deformable convolution module의 그라디언트는 bilinear operation을 통해 backporp 가능합니다.

### 2.2. Deformable RoI Pooling

Roi-pooling은 arbitrary size의 rectangle region을 고정된 크기의 feature로 변환합니다. 이는 모든 region proposal based object detection 방법에서 사용됩니다.

크기가 WxH 인 ROi가 주어지면 이를 kxk 의 크기의 bin으로 균등 하게 나눕니다. 표준적인 Roi-pooling 단계는 input feature map x로 부터 kxk pooled featue map y 를 출력합니다. (i,j)-th bin 에 대한 pooling operation은 다음과 같이 정의됩ㄴ디ㅏ.

  <img src="https://latex.codecogs.com/gif.latex?y%28i%2Cj%29%20%3D%20%5Csum_%7Bp%20%5Cin%20bin%28i%2Cj%29%7D%20x%28p_0&plus;p%29/n_%7Bij%7D"/>

여기서 p_o 은 ROI의 top-left corner 이며 n_ij 는 bin안의 픽셀의 갯수를 나타냅니다. deformable convolution과 비슷하게 offset 을 spatial binning postion에 추가합니다.

  <img src="https://latex.codecogs.com/gif.latex?y%28i%2Cj%29%20%3D%20%5Csum_%7Bp%20%5Cin%20bin%28i%2Cj%29%7D%20x%28p_0&plus;p%20&plus;%20%5CDelta%20p_%7Bij%7D%29/n_%7Bij%7D"/>

여기서도 delta p_ij가 분수일수 있음으로 이는 bilinear interpolation을 통해 구현됩ㄴ디ㅏ.

<b>Learning the offsets </b> 피규어3 에서 볼 수 있는바와같이 표준적인 ROI-pooling을 통해 먼저 pooled feature map을 생성합니다. pooled feature map 에서 fc-layer가 noamlized offset 을 출력합니다. 이 normalized offset은 위의 수식의 offset으로 변환되게 됩니다. 이 변환은 ROI의 width, height 와의 element-wise product를 통해 수행되며 pre-fixed scalar gamma를 가지고 있습니다. 이 논문에서는 gamma=0.1을 사용합니다. 

  <img src="https://latex.codecogs.com/gif.latex?%5CDelta%20p_%7Bij%7D%20%3D%20%5Cgamma%20%5CDelta%20%5Chat%7Bp_%7Bij%7D%7D%20%5Ccirc%20%28w%2Ch%29"/>

normalized offset을 사용하는 것이 ROI size에 대해 invariant 하게 만들어 준다는 것을 발견하였고 성능이 향상되었습니다. deformable ROI pooling 모듈에서도 offset을 생성하기 위한 fc-layer는 back-prop을 통해 학습 가능합니다.

<b>Deformable Position-Sensitive RoI Pooling</b> postion-sensitive ROI-pooling은 표준적인 ROI-pooling의 변형입니다. 이는 classfication, bounding box regression과 같은 특정한 task을 위한 postion-sensitive score map 위에서 연산이 수행되며 학습해야할 weight 가 필요하지 않습니다.

psRoi-pooling은 deformable 하게 확장될 수 있습니다. deformable ROI-pooling 연산의 수식 6을 postion-sensitive score map 에서 적용하면 됩니다. 표준적인 ROI-Pooling과 다른 점은 offset 을 생성하는 것을 sensitive score map이 아닌 다른 feature map 에서 생성되어야 합니다.(position-sensitive score map은 채널별로 location에 대한... 활성화 정도가 달라서 바로적용하기 힘든가 보다. 아닌가 class 별로 채널이 나눠져 있어서.. 적용하기 힘든가??? 하여튼 바로 적용하기 힘들다.) 이 논문에서는 position-sensitive score map 바로 전의 feature map을 활용하여 offset을 생성합니다.

### 2.3. Deformable ConvNets

deforamble convolution, Roi-pooling 모듈은 원래의 버젼과 같은 input, output을 갖습니다. 이를통해 기존의 CNN 모듈을 쉽게 대체 할 수 있습니다. 트레이닝 과정에서 offset을 학습하기 위한 추가적인 conv/fc layer들은 0으로 초기화 되며 learinig rate는 기존 레이어의 beta 배가 됩니다. (beta=1 이 디폴트) 이들은 bilinear interpolation을 통한 backprop을 통해 학습되게 됩니다. 이러한 CNN 구조를 defromable Convnet이라고 명칭합니다.

최신의 CNN 아키텍쳐와 deformable Convnet을 통합하기 위해 두단계로 구성합니다. 먼저 fully convolutional network 를 통해 feature map을 생성합니다. 두번째로 shallow task specific network가 이 feature map을 통해 결과를 출력합니다. 우리는 밑에서 이러한 두 스탭을 자세히 설명합니다.

<b>Deformable Convolution for Feature Extraction</b> 우리는 feature extraction을 위한 Resnet-101, InceptionResnet의 변형을 채택했습니다. 두 모델 모두 ImageNet classfication task 에서 pre-train 되어졌습니다. 원래의 Inception-Resnet 구조는 image recognition을 위해 설계되었으며 dense prediction 을 위한 valid conv/pooling layer 때문에 feature mis-alignment 문제가 발생합니다. 우리는 네트워크 구조를 수정해 이러한 mis-alignment 문제를 해결했습니다. 이러한 수정된 아키텍쳐는 "Aligned-Inception-Resnet" 이라고 부르며 자세한 내용은 부록에 참조되어 있습니다.

두 모델 모두 몇개의 convolutional block과 image classfication을 위한 average pooling과 1000 fc-layer로 구성되어져 있습니다. 우리는 average pooling, fc-layer를 제거하고 마지막 conv channel을 1024로 줄이기 위해 랜덤하게 초기화된 1x1 convolution을 마지막에 추가합니다. [4,7] 에서의 일반적인 관행에서와 같이 우리는 feature map resolution을 증가시키기 위해 마지막 conv block의 stride를 줄여 effective stride를 32 pixel에서 16 pixel로 줄입니다. 구체적으로 말하면 마지막 블록의 시작 에서 stride를 2에서 1로 줄입니다 (두 모델모두 conv5 의 stride를 조절합니다.) recpetive field를 동일하게 맞추기 위해 마지막 block의 커널사이즈가 1 이상인 convolution filter들의 dilation을 1에서 2로 변경합니다.

deformable convolution은 마지막 몇개의 커널사이즈가 1 이상인 convolution에 적용됩니다. 우리는 다양한 실험의 결과 마지막 3개의 convolutional layer에 적용하는 것이 다양한 task에 좋은 하이퍼 파라미터 라는 것을 발견했습니다. 이는 테이블 1에 나와 있습니다.

<b>Segmentation and Detection Networks</b> task specific 네트워크는 위에서 언급한 feature extraction 네트워크에 의해 계산되게 됩니다. C는 클래스에 갯수로 명시합니다.

<b>DeepLab</b> 은 semantic segmentation을 위한 최신 방법으로써 마지막 feature map에 1x1 conv layer를 추가하고 (C+1) feature map을 출력하고 이는 per-pixel classfication score를 representation 합니다. softmax 를 적용해 per-pixel prob을 얻을 수 있습니다.

<b>Category-Aware RPN</b> 은 [44] 에서 사용된 RPN과 동일하지만 2 class (object or not) 대신 (C+1) class classfier를 사용합니다. 이는 SSD의 간단한 버젼으로 생각 될 수 있습니다.

<b>Faster RCNN</b> 은 최신 object detector 입니다. [23] 에서 Resnet-101의 conv4, conv5 사이에 ROI-pooling이 삽입되어 per-ROI 당 10개의 layer에 대한 계산이 필요합니다. 이 아키텍쳐는 높은 성능을 보이지만 per-ROI 당 연산량이 높습니다. 이러한 연산량을 줄이기 위해 [FPN(36)] 에서와 같이 경량화된 버젼을 사용합니다. ROI-pooling layer는 feature extraction 네트워크의 끝에 추가됩니다. (마지막 layer의 모델의 경량화를 위해 256 차원의 feature를 출력하도록 수정됩니다) 두개의 새로운 1024 dim fc-layer가 추가되어 ROI pooled feature에 적용됩니다. 그 이후 bounding box regression, classfication branch가 적용됩니다. [Faster RCNN(44)] 에 따라 RPN 이 conv4 block의 끝에 추가됩니다.

선택적으로 Roi-pooling layer는 deformable ROi-pooling으로 대체 될 수 있습니다. 이 경우 offset을 위한 추가적인 fc-layer의 learning rate는 기존 layer의 0.01 배로 설정합니다.

R-FCN은 per-ROI 연산을 줄일 수 잇는 최신의 detector입니다. 우리는 R-FCN의 구현을 따랐습니다.

선택적으로 psRoi-pooling을 deformable 버젼으로 바꿀 수 있습니다. per-ROI 연산이 없기 때문에 (C+1) 카테고리에 대한 (C+1) 종류의 offset을 학습 가능하며 이에 추가적으로 class-agnoistic bounding box regression을 위한 offset의 그룹이 추가됩니다.

### 3. Understanding Deformable ConvNets

이 논문은 간단한 아이디어로 부터 시작되었습니다. convolution, ROI-pooling의 spatial sampling location 이 추가적인 offset에 의해 augment 됩니다. 이러한 offset은 데이터와 target task로 부터 학습됩니다. deformable 모듈이 여러개가 쌓기에 되면 deformation의 효과가 심해지게 됩니다.

이것은 피규어 4에서 살펴볼 수 있습니다. receptive field, sampling location 이 top-left feature map에 모여 있는 것을 볼 수 있습니다. 이들은 object의 scale, shape에 따라 adaptive 하게 조정됩니다. 더 많은 예제들은 피규어 5에 나와있으며 테이블 2는 정량적 분석을 제공합니다.

deformable Roi-pooling의 효과 또한 비슷하며 피규어 6에 나와있습니다. 표준적인 ROI-pooling에서의 regular grid 구조는 더이상 나타나지 않습니다. 대신 part 들이 초기의 ROI에서 벗어나 근처의 object region으로 이동하게 됩니다. localization capability가 향상되며 non-rigid object의 경우 특히 향상됩니다.

### 3.1. In Context of Related Works

<b>Spatial Transform Networks (STN)</b> 이 연구는 딥러닝 프레임워크에서 데이터로 부터 spatial transformation을 배우도록 한 첫번째 작업입니다. 이는 affine transformation과 같은 global parametric transformation을 통해 feature map을 warping 하게 됩니다. 이러한 warping은 연산량이 비싸고 transformation 파라미터를 배우는 것이 어렵습니다. STN은 작은 scale의 이미지에서의 classification에만 성공적 이였습니다. inverse STN 방법은 효율적인 transformation parameter propagation을 통해 연산량이 비싼 feature warping을 대체했지만 다른 문제점들은 여전히 남아 있습니다.

deformable convolution은 [25] 에서 offset을 학습하는 것에 대해 언급한 것철머 일종의 경량화된 spatial transformer 로 볼 수 있습니다. 그러나 이는 parametric transformation을 적용하지 않고 feature map을 warping을 하지 않습니다. 단지 feature map의 샘플링 하는 방법을 바꿧을 뿐이며 local, dense 한 방법입니다. 새로운 feature map을 출력하기 위해 deformable convolution은 단지 weighted summation 만을 수행하며 이는 STN에서 볼 수 없습니다. (STN에서는 구해진 transformation 파라미터에 따라 feature를 warping해야 해서 연산량이 많은데 deformable conv는 단지 weighted summation 일 뿐이라서 매우 가볍다)

deformable conv는 현존하는 CNN의 구조에 쉽게 통합이 가능하며 학습을 하기도 쉽습니다. 이는 STN에서 학습하지 못하는 경우에 대해도 적용가능합니다. 처음으로 우리는 spatial transformation과 통합된 CNN이 dense(semantic segmentation) or semi-dense(object detection) prediction이 필요한 대규모의 vision task에 효율적이라는 것을 보여줍니다.

<b>Effective Receptive Field</b> 이 논문은 receptive field의 모든 pixel이 output response에 동일하게 기여하지는 않는다는 것을 발견했습니다. 중심 근처에 있는 pixel이 더 많은 영향을 끼치기 때문에 effective receptive field는 이론적인 receptive field과 다르며 gaussian 분포를 따르게 됩니다. 이론적인 receptive field 의 크기는 conv layer가 증가함에 따라 선형적으로 증가하지만 놀라운 결과는 effective receptive field의 크기는 conv layer의 숫자의 제곱근에 선형적으로 증가한다는 것입니다. 그러므로 우리가 기대한 것 보다 receptive field는 훨씬 느리게 증가합니다.

이러한 발견은 현재 최근 CNN 구조의 가장 높은 layer도 충분한 receptive field를 가지지 못할 수 있ㅇ므을 나타냅니다. 이는 atrous convolution 같은 것들이 vision task에서 널리 사용되는지를 부분적으로 설명해 줍니다. 또한 adaptive receptive field learning 에 대한 필요성을 보여줍니다. deformable convolution은 피규어 4,5 에서 볼 수 있듯이 adaptive receptive field를 학습 할 수 있습니다.

<b>Atrous convolution</b> 일반적인 cnn 필터의 stride를 1보다 크게 늘리고 원래의 weight를 sparse location 로 놓습니다. 이는 receptive field size를 늘리지만 연산량과 파라미터는 같은 복잡도를 가지게 됩니다. 이는 [39, 5, 50] 과 같은 semantic segmentation, object detection, image calssfication 에서 널리 사용됩니다 [50] 에서는 dilated convolution 이라고 불립니다. deformable convolution은 atrous convolution의 일반화된 버젼입니다. atrous convolution에 대한 비교가 테이블 3에 나와있습니다.

<b>Deformable Part Models (DPM)</b> deformable ROI pooling은 DPM과 매우 비슷합니다. 두가지 방법 모두 classfication score를 최대화 하기 위해 object part에 대한 spatial configuration을 학습합니다. deforamble ROI-pooling은 object part간의 관계를 고려하지 않아 더 간단합니다.

DPM 은 sallow model이며 doforamtion을 모델링 하는 능력이 제한됩니다. DPM의 inference 알고리즘은 distance transform을 일종의 특별한 pooling 연산으로 취급한 CNN으로 변형 될 수 있지만 이는 end-to-end 학습이 불가능 하며 object part size, component를 결정해야 하는 휴리스틱이 포함되어져 있습니다. 반면 deformable convnet은 deep model 이며 end-to-end 학습이 가능합니다. deformabel convolution, Roi-pooling을 쌓으므로써 deformation의 모델링 능력이 강해집니다.

<b>DeepID-Net</b> DPM과 비슷하게 DeepID-NET은 object detection을 위해 object part의 deformation을 고려한 deformation constrained pooling layer를 도입합니다. 그러나 이 레이어는 deformable ROI-pooling에 비해 매우 복잡하고 어렵습니다. 이 논문은 RCNN을 기반으로 매우 고도로 설계되었으며 이는 최근의 object detection 방법과 통합되기 어렵습니다.

<b>Spatial manipulation in RoI pooling</b> Spatial Pyramid pooling은 미리 정해진 sclae에 따른 hand-crafted pooling region을 사용합니다. 이것은 컴퓨터 비전 커뮤니티에서의 주된 접근 방법이며 딥러닝 기반의 object detection에도 사용됩니다. pooling region의 spatial layout를 학습하는 것은 거의 연구되지 않았습니다. [26] 에서는 large over-complete set(? 이게 뭐지) 에서 sparse subset of pooling region을 학습합니다. 이러한 large set 은 hand-crafted 로 설계되어졌고 학습도 end-to-end가 아닙니다. deformable ROI-poling은 CNN안에서 pooling region을 end-to-end로 학습하는 첫 연구입니다. 현재는 region이 같은 크기를 가지지만 spatial pyramid pooling과 같이 다양한 size로 확장하는 것은 매우 간단합니다.

<b>Transformation invariant features and their learning</b> transformation invariant feature를 설계하기위한 엄청난 노력들이 있어왔습니다. 주목할 만한 예로는 SIFT(sclae invariant feature transform), ORB(O for orientation) 등이 있습니다. CNN을 사용한 연구에서도 다양한 연구가 많이 있습니다. [34] 에서는 image transforamtion 에 대한 CNN representation의 invariance, equivalence에 대한 연구가 진행되었습니다.

몇몇 연구는 다양한 transforamtion에 관한 CNN의 invariant representation을 학습합니다. [3] scattering network [30] convolutional jungle [31] TIpooling 등이 있습니다. 몇몇 연구는 symmetry, scale, rotation에 관한 특정 transformation 에 대해 다룹니다.

섹션 1에서 말했듯이 이러한 연구들은 transformation이 알려져 있다고 가정합니다. 이러한 지식은 feature extraction 알고리즘을 위한 hand-craft structure를 만드는데 사용됩니다. SIFT와 같이 고정된 feature를 사용하거나 CNN을 기반으로 학습 가능한 파라미터로 구현될 수 있습니다. 이러한 것들은 새로운 task에서 알려져 있지 않은 transformatio을 다룰 수 없습니다.

반면 deformable module은 다양한 transformation에 대해 적용 가능하며 이러한 transformation invariance는 데이터로 부터 자동적으로 학습됩니다.

<b>Dynamic Filter</b> deformable convolution과 비슷하게 dynamic filter는 input feature에 조건화 되어 convolution region sample을 변경합니다. 다른점은 sampling location이 아닌 filter weight만을 학습합니다. 이 연구는 video, stereo prediction에 적용됩니다.

<b>Combination of low level filters </b> 가우시안 필터와 가우시안의 smooth 미분 필터는 corner, edge, T-junction과 같은 low level image 구조를 추출하는데 널리 사용됩니다. 특정 조건 하에서 이러한 필터는 일종의 basis set을 형성하고 이들의 선형 조합을 통해 같은 그룹의 geometric transformation의 새로운 필터를 형성합니다. [12]의 steerable filter, [43] 의 multiple scale 등이 있습니다. 우리는 [43] 에서 deformable kernel 이라는 용어가 사용된걸 주목했지만 우리의 연구에서의 의미는 다릅니다.

대부분의 CNN 구조는 그들의 필터를 scratch로 배우게 됩니다. 최근의 연구 [24] 에서는 scratch 부터 학습하는게 불필요 할 수 있음을 보여줍니다. 이 논문은 필터의 weight를 low level filter(가우시안의 4차 tailor 근사)의 weigted combination 으로 대체하고 weight coefficient만을 학습합니다. filter function space에 대한 regularization은 학습 데이터가 작은 경우 일반화 능력을 향상 시키는 것으로 나타났습니다.

위의 작업은 multiple filter, 특히 다양한 scale filter가 결합된 경우 우리의 연구와 관련이 있습니다. 학습된 필터는 매우 복잡한 weight를 가질 수 잇으며 deformable convolution filter와 유사 할 수 있습니다. 그러나 deformable convolution은 filter weight 대신 sampling location을 학습합니다.

### 4.1. Experiment Setup and Implementation

<b>Semantic Segmentation </b> 우리는 PASCAL VOC, CityScape 데이터 셋을 사용합니다. PASCAL VOC에는 20개의 semantic category가 존재합니다. [19, 39, 4] 에 따라 우리는 VOC2012의 데이터셋과 추가적인 [18] 에서 사용된 mask annotation을 사용합니다. 학습 데이터는 10582개의 이미지와 val set 1449 이미지로 구성되어져 있습니다. CityScape 를 위해서는 [5]에 따라 학습데이터는 2975 이미지, val set은 500개의 이미지로 구성하였고 19개의 semantic category와 background category가 존재합니다.

evalutation을 위해 우리는 mIOU 를 사용하며 이미지 전체 픽셀에 대해 정의된 매트릭입니다. [10,6] 에 따라 우리는 PASCAL VOC에는 mIOU@V, Cityscape에는 mIOU@C 를 사용합니다.

traning, inference 에서는 이미지의 짧은 면이 360(PASCAL VOC) ,1024(Cityscape) pixel이 되도록 합니다. SGD 학습과정에서 각 미니배치마다 하나의 이미지가 랜덤하게 골라집니다. 총 30K(PASCAL VOC), 45K(Cityscape) 만큼을 업데이트 하였습니다. 8개의 GPU와 각 GPU마다 하나의 미니배치를 사용합니다. 처음 2/3 iteration 만큼은 1-e^3, 나머지 1/3은 1-e^4 learning rate를 적용합니다.

<b>Object Detection</b> 우리는 PASCAL VOC, COCO 데이터 셋을 사용합니다. PASCAL VOC는 [15] 에 따라 VOC 2007 trainval, VOC 2012 trainval 을 합쳐서 사용합니다. VOC 2007 test에 대해 evalutation이 수행됩니다. COCO 데이터 셋은 [37] 에 따라 trainval set 120k 이미지 test-dev set 20k 이미지를 사용합니다.

evaluation을 위해서는 map score를 사용합니다. PASCAL VOC는 임계값 0.5, 0.7 에 대한 IOU를 보고 하며 COCO 는 map@(0.5:0.95) 를 사용합니다

training, inference 과정에서 이미지는 짧은 면이 600 pixel로 리사이즈 됩니다. SGD training에서 한 미니배치당 랜덤하게 하나의 이미지를 샘플링 합니다. class-aware RPN에는 256 roi 가 샘플링 되며 Faster RCNN, R-FCN 을 위해서는 RPN 에는 256, object detection 네트워크에는 128 ROI를 사용합니다. roi-pooling은 7x7 bin을 사용합니다. VOC에서의 절제 실험을 위해 우리는 [36]을 따라 pre-trained fixed RPN, Faster R-CNN, R-FCN을 사용하며 RPN과 object detection network와의 feature는 공유되지 않습니다. RPN 네트워크는 [44] 의 첫번째 단계처럼 개별적으로 학습됩니다. COCO의 경우 [45] 에서와 같은 joint training을 수행하며 training 과정에서 feature가 공우됩니다. PASCAL VOC에는 30k, COCO 에는 240k iteration 만큼 학습며 8개의 GPU를 사용합니다. 2/3 iter까지는 10^-3, 나머지 1/3 iter 는 10^-4 learning rate를 적용합니다.

### 4.2. Ablation Study

다양한 절제 실험이 우리의 접근 방법을 검증하기 위해 실험되었습니다.
<b>Deformable Convolution</b> 테이블 1은 resnet-101을 feature extractor로 사용한 deformable convolution의 효과를 평가한 결과입니다. Deeplab, class-aware RPN의 경우 deformable convolution 을 사용할때 성능이 꾸준히 향상됩니다. Deeplab의 경우 3개의 deformable layer를 사용할때 성능개선이 멈췃으며 class-aware RPN, Faster RCNN, R-FCN의 경우 6개의 deforamable layer에서 성능 개선이 포화했습니다.

피규어 4, 5에서 볼 수 있듯이 deformable convolution layer로 부터 학습된 offset이 이미지의 내용에 따라 adaptive 하다는 것을 관찰했습니다. deformable convolution의 메커니즘에 대한 더 나은 이해를 위해 우리는 deformable filter를 위한 effetive dilation 이라는 매트릭을 정의합니다. 이는 필터 내의 인접한 sampling location 사이의 거리의 평균입니다. 이는 필터의 receptive field size에 대한 대략적인 측정이 됩니다.

우리는 R-FCN에 3개의 deformable layer를 적용하고 VOC test set에 대해 평가했습니다. deformable convolution filter를 small, medium, large, background의 4가지 범주로 나눴으며 이는 filter의 센터가 위치한 ground truth box의 라벨을 통해 나눴습니다. 테이블 2는 effective dilation 값들에 대한 결과를 보여줍니다. 이를 통해 다음과 같은 것을 알 수 있습니다. 1) deformable filter의 receptive field의 크기는 object의 크기와 상관이 있으며 이미지의 내용에 따라 deformation이 잘 학습되었음을 보여줍니다. 2) background region의 필터의 크기는 medium, large object의 사이쯤에 있으며 background region을 인식하기 위해서는 상대적으로 큰 receptive field가 필요하다는 것을 보여줍니다. 이러한 관찰은 다른 레이어에서도 보여집니다.

디폴트로 사용한 resnet-101 모델은 마지막 3x3 conv layer에 dilation 2를 적용한 atrous convolution을 적용 합니다. 우리는 dilation을 4,6,8 을 적용한 모델을 실험하고 이를 테이블 3에 보고했습니다. 1)  더 큰 dilation value를 쓸수록 성능이 증가하는 것을 관찰했고 이는 디폴트 resnet의 receptive field가 너무 작은 것을 나타냅니다. 2) optimal dilation 은 task에 따라 달라집니다. Deeplab은 6이 최적이었고 Faster R-CNN은 4가 최적이였습니다. 3) deformable convolution이 가장 좋은 성능을 보엿습니다. 이러한 관찰은 adaptive filter deformation이 효과적이고 필수적이라는 것을 보여줍니다.

<b>Deformable RoI Pooling </b> Faster RCNN, R-FCN 모두 Deformabl Roi-pooling을 적용 가능합니다. 테이블3에서 볼 수 있듯이 deformable Roi-pooling 만을 사용해도 주목할 만한 성능 향상이 보이며 특히 엄격한 임계값을 적용한 map@0.7 에서 많은 성능 향상을 보입니다. deformable convolution, Roi-pooling을 모두 사용하면 상당한 성능 개선을 얻을 수 있습니다.

<b>Model Complexity and Runtime</b> 테이블 4는 모델의 복잡도와 시간복잡도를 보고합니다. deformable convnet은 약간의 파라미터와 연산량만을 추가합니다. 이는 모델의 파라미터가 증가하여 성능이 개선된 것이 아닌 geometric transformation 을 모델링 할 수 있는 능력 때문에 성능이 개선된 것을 나타냅니다.

### 4.3. Object Detection on COCO

테이블 5에서 COCO test-dev set에서의 deformable convnet과 plain convnet간의 비교를 수행합니다. 우리는 먼저 resnet-101 모델을 사용하여 실험을 진행하였습니다. deformable class-aware RPN, Faster R-CNN, R-FCN 은 각각 25.8%, 33.1%, 34.5% 의 map@(0.5:0.95) 를 달성했습니다. 이는 plain convnet 보다 11%, 13%, 12% 높은 성능입니다. resnet-101을 aligned-inception-resnet으로 교체함으로써 이들의 feature representation 성능이 향상됩니다. deformable convnet을 추가함으로서 얻는 성능 향상도 그대로 유지됩니다. multiple image scale testing을 추가하고 (이미지의 짧은 면이 480, 576, 688, 864, 1200, 1400 이 되도록한다) iterative bounding box average를 수행함으로써 deformable R-FCN의 성능은 map@(0.5:0.95) 는 37.5% 까지 향상됩니다. deformable convnet 으로 얻어진 성능 개선은 이러한 bells and whistle과 보완적입니다.

### 5. Conclusion

이 논문은 간단하고, 효율적이고, deep model, end-to-end 학습이 가능한 deformable convnet을 제안하고 이는 dense spatial transformation을 모델링 가능하게 합니다. 우리는  dense spatial transformation을 모델링 하는 것이 가능하며 obeject detection, semantic segmentation과 같은  vision task를 위해 이를 학습하는 것이 효과적이라는 것을 보여줍니다.

### A. Details of Aligned-Inception-ResNet

원래의 inception-resnet 구조에서 여러개의 valid padding convolution/pooling 이 사용됩니다.이는 dense prediction task에 필요한 feature alignment 문제를 발생시키게 됩니다. output과 가까운 feature map의 cell의 경우 이미지에 정사영된 spatial location의 위치가 그 cell의 receptive field 의 중앙에 위치하지 않습니다. 반면에 task specific sub-network의 경우 일반적으로 alignment 가정하에 설계됩니다. 예를들어 semantic segmentation 에서 자주 사용되는 FCN의 경우 cell의 feature는 그 cell이 정사영된 image location의 pixel label을 예측하는데 사용됩니다.

이러한 문제를 다루기 위해 테이블 6에 나와있는 Aligned-Inception-ResNet 구조를 설계했습니다. 이는 feature의 차원이 변경될때 1x1 stride 2의 conv layer가 사용됩니다. 원래의 inception-resnet과의 2가지의 주요한 차이점이 있습니다. 1) Aligned-Inception-ResNet 은 feature alignment 문제를 겪지 않습니다. 이는 conv/pooling layer에서 적절한 padding을 사용함으로써 달성합니다. 2) Aligned-Inception-ResNet은 반복적인 모듈로 설계되어서 원래의 디자인보다 간단합니다. 

##### 읽어보고 싶은 논문
invariant, equivalence에 대해 다룬 논문이라고 한다.  
1. Understanding image representations by measuring their equivariance and equivalence










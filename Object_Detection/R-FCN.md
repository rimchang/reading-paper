# R-FCN:Object Detection via Region-based Fully Convolutional Networks


### Abstract
region-based, fully convolutional network를 통한 정확하고 효율적인 object detction 알고리즘을 소개합니다. 기존의 Fast/Faster R-CNN 같은 region-based detector와 은 per-region 으로 수행되어 비효율적인 반면 우리의 모델은 fully convolutional을 이용하여 전체의 이미지에서의 computation을 공유합니다. 이러한 목적을 위해, 우리는 <span style="color:RED">position-sensitivescore maps를 제안하고 이는 image classfication, object detection 에서의 translation-variance dilemma를 다룰 수 있도록 합니다.</span> R-FCN은 자연스럽게 Resnet 같은 fully convolutional image classfier를 backbone으로 사용할 수 있게 합니다. 101-layer Resnet을 이용하여 PASCAL VOC dataset 에서 좋은 결과 (83.6% map on 2007 set)을 보였습니다. 또한 170ms per image의 test-time 을 보였고 이는 Faster RCNN 보다 2.5~20배 정도 빠릅니다. 

### 1. Introduction

[Fast RCNN(6), SPP NET(8), Faster RCNN(18)] 같은 종류의 object detection 방법은 두가지 종류의 subnetwork로 나눠질 수 있습니다. 1, Region-of-Interest(ROI) pooling layer 이는 ROI와 독립적으로 하나의 네트워크를 공유합니다. 2, ROI-wise subnetwork 이는 computation을 공유하지 않습니다. 이러한 Alexnet, VGGnet 등에서 비롯된 역사적인 이유 때문인데 이들은 두가지의 subnetwork로 디자인되어졌습니다. spatial pooling layer와 그 이후의 fully-connected layer로 이루어져 있습니다. 전통적 classfier에서의 마지막 spatial pooling layer는 object detecion network에서의 ROI pooling 으로 생각 할 수 있습니다.

하지만 최근의 Resnet, Googlenet 과 같은 네트워크는 fully convolutional 로 디자인되어져 있습니다. object detection 구조에서도 all convolutional layer를 쓰려는 것은 자연스러운 움직임입니다. 그러나 경험적으로 실험된 결과 naive solution 은 낮은 detectoin accuracy를 가지는 것으로 나타났고 네트워크의 superior classfication accuracy 와는 다른 양상을 보입니다. 이러한 문제를 다루기 위해서 Restnet 논문에서 object detection을 위해 사용된 ROI-pooling layer는 자연스럽지 않게 삽입되어졌습니다. 이는 ROI-wise subnetwork의 accuracy를 높이긴 했지만 unshared per-ROI computation 때문에 속도를 낮췄습니다.

우리는 앞서 말한 자연스럽지 않은 디자인이 이미지 classfication 에서의 translation invariance의 증가 vs object detection 에서의 translation variance의 딜레마 때문에 발생한다고 주장합니다. image-level classfication 에서는 translation invariance가 도움이 됩니다. 이미지안에서 shift된 오브젝트를 구별해야만 합니다. 그래서 fully convolutional 디자인은 translation-invariant한 특성 때문에 때문에 선호가 됩니다.

반면에 object detection task는 translation-variant한 localization representation이 필요합니다. 예를들어 한 candidate box안의 오브젝트의 translation은 cadidate box가 얼마나 좋은지를 평가하는데 의미를 가집니다. 우리의 가정은 이미지 classfication에서의 deeper covoluional layer는 translation에 민감하지 않다는 것입니다. <span style="color:RED">이런 딜레마를 다루기 위해서 Resnet 논문에서의 object detection 구조는 convolutions 으로 ROI-pooling layer를 집어 넣었습니다.</span> 이러한 region-specific 연산은 translation invariance를 줄여줍니다. 그리고 post-ROI convolutional layer는 더이상 translation-invariant하지 않게 됩니다. 그러나 이런 디자인은 region-wise layer의 갯수 때문에 training, testing의 효율을 희생하게 됩니다.

이 논문에서는 우리는 object detection을 위한 R-FCN이라는 구조를 제안합니다. 우리의 네트워크는 shared, fully convolutional 구조를 가집니다. FCN이 translation variance하도록 하기 위해서 우리는 position-sensitive score map 이라는 것을 사용합니다. 이는 특수한 convolutional layer를 FCN의 출력으로 사용합니다. <span style="color:RED">각각의 score map은 (to the left of an object) 와 같은 상대적인 spatail postion을 부호화 합니다. </span> FCN의 마지막에 position-sensitive ROI pooling layer를 추가하였고 이는 추가적인 weight를 필요로 하지 않습니다. 전체 구조는 end-to-end로 학습가능합니다. 모든 학습 가능한 레이어는 convolutional 구조이고 이미지 전체에서 공유됩니다. 또한 object detection에 필요한 spatial imformation을 부호화 합니다. 

Resnet-101 을 사용하여 우리의 R-FCN 모델은 PASCAL VOC 2007에서 83.6% MAP, VOC 2012 에서 82% MAP을 달성했습니다. 또한 test-time speed로 이미지당 170ms를 당설하였고 이는 Faster RCNN보다 2.5 ~20배 정도 빠른 결과입니다. 

### 2. Our approach

<b>Overview</b>. R-CNN과 같이 우리는 two-stage object detection 전략을 적용했습니다. 1, region proposal 2, region classfication. [YOLO(17), SSD(14)] 와 같은 region proposal이 없는 방법이 존재하지만 region-based system은 높은 성능을 보이고 있습니다. 우리는 RPN을 통해 candidate region을 추출합니다. RPN은 그 자체로 fully convolutional 구조를 가집니다. [Faster RCNN(18)] 과 같이 RPN, R-FCN은 feature 를 공유합니다. 

RPN을 통해 ROI가 주어졌을때 R-FCN 구조는 object category와 background를 classfy 하도록 디자인 되어졌습니다. R-FCN의 모든 학습 가능한 wieght 들은 모두 convolutional 하며 전체 이미지에서 계산됩니다. 마지막 convolutional layer는 각 카테고리당 k^2이 position-sensitive score map을 출력합니다. 그래서 총 k^2(C + 1) 개의 아웃풋을 출력합니다. K^2 개의 score map은 kxk 의 spatial grid에 해당하는 갯수입니다. 이는 relative position 인데 예를들어 k x k = 3 x 3 이라면 이 9개의 score map은 (top-left, top-center,,,, bottom-right) 를 나타냅니다.

postion-sensitive ROI-pooling layer는 마지막 convolutional layer의 결과를 aggregate하고 각각 ROI에 대한 score를 생성합니다. Faster R-CNN(6), SPP-NET(8) 과 다르게 우리의 position-sensitive ROI layer는 일종의 selective pooling 입니다. 각각의 kxk bin은 kxk 개의 score map을 하나의 score map으로 통합합니다. end-to-end 학습방법으로 이 ROI layer는 마지막 convolutional layer가 postion-sensitive score map을 배울수 있도록 합니다.

<b>Backbone architecture</b> R-FCN의 backbone은 Resnet-101을 사용합니다. 다른 네트워크도 사용가능합니다. Resnet-101은 100개의 convolutional layer로 이루어져있으며 그 이후 global average pooling과 1000-class fc layer로 이루어져 있습니다. 우리는 average pooling layer와 fc layer를 제거하고 오직 feature를 얻기 위한 convolutional layer만을 사용합니다. 우리는 Imagenet에 pre-trained된 resnet-101을 사용합니다. 마지막 convolutional block은 2048-dim이며 차원을 줄이기 위해 1024 1x1 conv layer를 붙였습니다. 그 이후 ROI pooling을 붙였습니다.

<b> Position-sensitive score maps & Position-sensitive ROI pooling.</b> 각각의 ROI에 대해 position imformation을 명시적으로 나타내기 위해서 우리는 사각형의 ROI를 kxk bin으로 나눴습니다. WxH의 사이즈의 ROI가 있다고 했을때 각 bin의 사이즈는 W/K * h/k 입니다. 우리의 방법에서 마지막 convolutional layer는 각 카테고리에 대한 k^2 score map을 생성합니다. 각 (i, j) 번째의 bin에 대해서 우리는 position-sensitive ROI pooing operation을 정의합니다. 

  <img src="https://latex.codecogs.com/gif.latex?r_c%28i%2Cj%7C%5CTheta%29%20%3D%20%5Csum_%7B%28x%2Cy%29%20%5Cin%20bin%28i%2Cj%29%7D%20z_%7Bi%2Cj%2Cc%7D%28x%20&plus;%20x_0%2C%20y%20&plus;%20y_0%20%7C%20%5CTheta%29%20/n"/>

(i, j) 번째의 bin에 대한 특정 클래스에대한 response는 전체 k^2(c+1) feature map 에서 bin에 해당하는 feature를 평균낸것!!

r_c(i, j) 는 (i, j)번째의 bin에 대한 c번째 class category에 대한 결과입니다. z_i,j,c 는 feature K^2(C+1) 의 score maps입니다. (x_0, y_0) 은 ROI의 top-left corner를 나타냅니다. n은 bin안의 pixel의 갯수입니다. Θ는 모든 학습가능한 parameter를 나타냅니다. (i,j) 번째의 bin의 span은 다음과 같습니다.

  <img src="https://latex.codecogs.com/gif.latex?%5Cleft%20%5Clfloor%20i%5Cfrac%7Bw%7D%7Bk%7D%20%5Cright%20%5Crfloor%20%5Cleq%20x%20%3C%20%5Cleft%20%5Clceil%20%28i&plus;1%29%5Cfrac%7Bw%7D%7Bk%7D%20%5Cright%20%5Crceil%20%5C%2C%5C%2C%20%5Cleft%20%5Clfloor%20j%5Cfrac%7Bh%7D%7Bk%7D%20%5Cright%20%5Crfloor%20%5Cleq%20y%20%3C%20%5Cleft%20%5Clceil%20%28j&plus;1%29%5Cfrac%7Bh%7D%7Bk%7D%20%5Cright%20%5Crceil"/>

위의 수식은 피규어 1에 나와있으며 여기서 색은 (i,j) 의 하나의 쌍을 나타냅니다. 위 수식은 average pooling을 수행하지만 max-pooling도 수행 가능합니다.

  <img src="https://raw.githubusercontent.com/DuinoDu/CsdnBlogCode/master/20170409/2.png"/>

k^2 position-sensitive score는 각 ROI에 대해 투표합니다. 이 논문에서는 averaging을 통해 간단히 vote합니다. 그다음 softmax를 수행합니다. 이들은 cross-entropy loss와 ROI의 순위를 정할때 사용됩니다.

Faster/Fast RCNN과 비슷한 bounding box regression을 수행합니다. k^2(C+1) dim 의 convolutional layer 와 별도로 4k^2 dim의 bounding box regression layer를 추가합니다. position-sensitive ROI pooling은 이 4k^2의 map위에서 수행됩니다. 각 ROI에 대해 4K^2 dim vector를 출력합니다. 그다음 average voting을 통해 4 dim vector로 통합됩니다. 이 4 dim vecotre는 Fast RCNN 에서 나온 (t_x,t_y,t_w,t_h) 로 parameterize 합니다. 우리는 간단함을 위해 class-agnositc bounding box regression을 수행했지만 class-specific(4k^2C dim) 도 가능합니다.

이러한 position-sensitive score map의 개념은 [Instance-sensitive fully convolutional networks(3)] 에서 영감을 받았습니다. 우리는 더 나아가 position-sensitive ROI Pooling을 object detection에 적용하였습니다. 여기에는 학습하는 weight가 존재하지 않아 거의 cost-free region-wise computation을 가지며 training, inference 의 속도를 향상시킵니다.

<b>Training</b> 미리 계산된 region proposal이 있다면 R-FCN을 end-to-end 로 학습시키는 것은 매우 쉽습니다. [Fast RCNN(6)] 에 따라 우리의 loss function은 cross-entropy, box regression loss의 합으로 정의됩니다.   <img src="https://latex.codecogs.com/gif.latex?L%28s%2C%20t_%7Bx%2Cy%2Cw%2Ch%7D%29%20%3D%20L_%7Bcls%7D%28s_%7Bc*%7D%29%20&plus;%20%5Clambda%5Bc*%20%3E%200%5DL_%7Breg%7D%28t%2Ct*%29"/>. c\*는 roi에 할당된 gt label을 나타내며 0일때 background가 됩니다. classfication loss는 cross-entropy이며 regression loss는 smooth L1 loss 입니다. t\* 는 gt box를 나타내며 [c*>0] 은 background 라벨이 아닐때에만 활성화되는 indicator 함수입니다. 우리는 람다를 1로 설정했으며 IOU가 0.5 보다 높으면 positive, 나머지는 negative box로 설정했습니다.

학습중에 OHEM 을 적용하는 것이 매우 쉽습니다. resnet의 뒷단에 psroi-pooling을 넣음으로써 매우 적은 per-Roi 연산이 수행되어 비용이 매우 적은 example mining(OHEM) 이 가능합니다. 각 이미지당 N개의 proposal 이 있다고 하면 N개의 proposal에 대해 loss를 계산합니다. 그 후 모든 ROI(positive, negative 모두) 를 loss에 따라 정렬하고 높은 순서대로 B개의 roi를 선택합니다. 이렇게 선택된 B개의 example에 대해 backprop을 수행합니다. 우리의 아키텍쳐에서 per-Roi 연산은 매우 적기 때문에 forward 시의 연산량이 각 이미지당 몇개의 proposal 이 존재하는지에 영향을 받지 않게 됩니다. 이는 FAST R-CNN 에서의 OHEM 이 학습시간을 두배로 늘리는 것과 달리 우리의 아키텍쳐는 거의 영향을 받지 않습니다. 우리는 테이블3 을 통해 다양한 통계치들을 제공합니다.

(왜 Fast R-CNN에서의 OHEM 이 왜 학습시간을 두배로 늘리지..?? Fast R-CNN 에서의 per-Roi 연산도 매우 적지 않나? Fast RCNN 에서는 per-ROI 별로 뒤의 fc-layer를 거쳐야 하기 때문이네 R-FCN에서는.. feature map에서 바로 softmax를 적용한다!! Faster-RCNN+++ 에서는 conv5 연산이 per-Roi 별로 적용되게 된다.)

우리는 0.0005 weight decay, 0.9 momentum 을 적용하였고 기본적으로 single-scale 학습을 진행하였습니다. [6,18] 에 따라 인풋 이미지는 가장 짧은 면이 600 pixel이 되도록 리사이즈 합니다. 하나의 GPU는 1개의 이미지를 인풋으로 받으며 selected 128 ROI 에 대해 backprop을 수행합니다. 8개의 GPU를 사용했으며 미니배치 사이즈를 8개로 늘리는 효과를 가집니다. fine-tuning 을 위해 VOC 에서 20K 까진 0.001 그후 10K 는 0.0001 의 learning rate를 적용했습니다. R-FCN과 RPN의 feature를 공유시키기 위해 [18] 에서의 4-step 학습 방법을 사용했고 이는 RPN, R-FCN의 학습을 번갈아 가며 학습합니다.

<b>Inference. </b> 피규어 2에서 볼 수 있듯이 RPN, R-FCN에서 사용되는 shared feature가 계산됩니다. 그후 RPN은 roi proposal을 수행하며 R-FCN은 category-wise score를 계산하고 bonding box regression을 수행합니다. inference 동안 [18] 과의 공정한 비교를 위해 300 ROI 로 설정하여 비교합니다. 이들 결과는 NMS에 의해 후처리 되며 IOU의 임계값은 0.3을 사용합니다.

<b>À trous and stride</b> 우리의 fully convolutional 아키텍쳐는 semantic segmentation에서 사용되는 FCN에서 network 구조를 변경하는 것을 받아들입니다. 우리는 Resnet-101의 stride를 32 pixel 에서 16 pixel 로 줄여 score map resolution을 증가시킵니다. conv4 이전의 모든 layer는 변경하지 않습니다. conv5 block의 첫번째 layer의 stride를 2에서 1로 수정하였고 conv5 단계의 모든 conv filter는 'hole algorithm' 에 의해 변경되고 이는 감소된 stride를 보정해 줍니다. 공정한 비교를 위해 RPN conv4 단계의 마지막에서 수행되어 집니다. (conv4 feature는 R-FCN과 공유됩니다) 이를 통해 RPN은 trous trick에 영향을 받지 않습니다. 다음 테이블은 R-FCN의 부분적 실험의 결과를 보여주며 a trous trick은 map를 2.6% 가량 향상시킵니다.

#### a trous trick 에 대해서 알아볼 필요가 있다.

<b>Visualization</b> 피규어 3,4 에서 R-FCN에 의해 학습된 3x3의 position-sensitive score map을 시각화 했습니다. 이러한 score map은 object의 상대적인 위치에서 강하게 활성화 될 것으로 기대합니다. 예를 들어 "top-center-sensitive" score map은 object가 top-center 위치에 있을대 강한 반응을 보입니다. 만약 피규어 3에서 처럼 candidate box가 object를 정확히 덮고 있다면 대부분의 k^2 bin의 score가 강하게 활성화 되고 이들의 voting은 높은 score를 가질 것입니다. 반대로 candidate box가 object를 제대로 덮고 있지 않다면 k^2의 몇몇 bin이 활성화 되지 않을 것이며 voting socre는 낮게 나올 것입니다.

### 3 Related Work

R-CNN은 deep network에서 region proposal의 사용의 효과에 대해 보여주었습니다. R-CNN은 cropped, warped region에 대해 conv network 을 연산하여 region간에 연산을 공유하지 않습니다. SPPnet, Fast R-CNN, Faster R-CNN은 일종의 'semiconvoulutional'  로써 이는 conv subnetwork는 전체 이미지에서 연산을 공유하고 (RPN, ROI-pooling) 다른 subnetwork(fc-layer)는 개별적인 region에 대해 연산을 수행합니다.

fully convolutional 모델로 볼 수 있는 object detector도 존재합니다. OverFeat 는 shared conv feature map 에서 multi-scale sliding window 를 통해 객체를 검출 합니다. Fast R-CNN 에서는 region proposal을 대체할 수 있는 sliding window 방법을 사용합니다. single scale sliding window의 경우 하나의 conv layer로 대체 할 수 있습니다. Faster R-CNN 에서의 RPN은 multiple size의 anchor 에 대한 bounding box를 예측하는 fully convolutional detector 입니다. 원래 논문에서의 RPN은 class-agnostic 이였지만 [SSD(14)] 와 같이 class-specific 도 가능하며 우리도 이를 실험해 봤습니다.

다른 object detector의 종류는 fc-layer를 활용하여 전체 이미지에서의 object detection 결과를 얻습니다.

### 4 Experiments

### 4.1 Experiments on PASCAL VOC

우리는 PASCAL VOC 의 20개의 카테고리에 대해 실험을 진행했습니다. VOC 2007, VOC 2012 의 trainval set을 모두 사용했으며 VOC 2007 test set에 대해 평가했습니다. object detection accuracy는 map에 의해 측정됩니다.

<b>Comparisons with Other Fully Convolutional Strategies</b>

fully conolutional detector를 사용하는 것이 가능하지만 실험은 이들이 좋은 성능을 보이는 것이 non-trivial 하다는 것을 보여줍니다. 우리는 다음과 같은 fully convolutional 전략 혹은 almost fully convolutional 전략(per ROI당 하나의 fc-layer를 갖는 구조)을 사용했으며 resnet-101을 사용했습니다.

<b>Naïve Faster R-CNN.</b> introduction에서 언급했던 듯이 resnet-101의 conv layer를 shared feature를 계산하느데 사용하고 마지막 conv layer(conv5) 이후에 ROI-pooling을 적용했습니다. 연산량이 적은 21-class fc-layer를 각각 ROI 에 대해 수행되어 이는 almost fully convolutional 입니다. a trous trick은 공정한 비교를 위해 사용했습니다.

<b>Class-specific RPN.</b> [Faster RCNN(18)] 에서의 2-class(object or not) conv classfier 대신 21-class conv classifer 를 사용한 RPN을 학습시켰습니다. 공정한 비교를 위해 a torus trick 을 사용합니다.

<b>R-FCN without position-sensitivity</b> k=1 로 설정함으로써 R-FCN의 position-sensitivity 를 제거합니다. 이는 각 ROI에 대해 global pooling을 수행하는 것과 동일합니다.

테이블2 는 실험 결과를 보여줍니다. Resnet 논문의 standard Faster R-CNN이 76.4% map를 달성했습니다. 이는 conv4, conv5 사이에 ROI-pooling을 삽입 하였습니다. 비교 해보자면 naive Faster R-CNN (conv5 이후에 roi-pooling을 적용) 은 성능이 급격히 떨어져 68.9%의 map를 보입니다. 이는 conv4, conv5 사이에 ROI-pooling을 삽입하여 spatial information의 정보를 보존하는 것이 중요하다는 것을 보여줍니다. 비슷한 결과가 [19] 에서도 보고되었습니다.

class-specific RPN은 표준 Faster-RCNN 보다 9% 낮은 67.6% 의 map을 보여줍니다. 이 비교는 [6,12] 에서의 보고와 일치합니다. 실제로 class-specific RPN은 proposal을 위해 dense sliding window를 사용하는 Fast RCNN의 일종으로 볼 수 있으며 이는 [6,12] 에서 낮은 성능을 보인다고 보고되었습니다.

반면 우리의 R-FCN은 더 나은 accuracy를 보여줍니다. standard Faster RCN이 76.4% map를 보이는 반면 R-FCN은 76.6% map를 보여줍니다. 이 결과는 position-sensitive 전략이 object의 위치를 알기 위해 효율적으로 spatial information을 인코딩 하는 것을 보여줍니다. 또한 ROI-pooling이후 배워야 할 layer가 필요하지 않습니다.

position-sensitivity의 중요성은 k=1로 설정함으로써 보여집니다. 이때 R-FCN은 수렴하지 않았습니다. 이 경우에는 ROI 내에서 spatial information을 명시적으로 포착할 수 없습니다. 또한 naive Faster R-CNN에서 ROI-pooling의 출력이 1x1인 경우에도 수렴을 할 수 있지만 map가 61.7% 로 떨어지는 것을 보고합니다.

<b>Comparisons with Faster R-CNN Using ResNet-101</b>

다음으로 우리는 가장 강력한 경쟁 상대인 "Faster R-CNN + Resnet-101" 과 비교하며 이는 PASCAL VOC, MS COCO, ImageNet 밴치마크에서 최상위 성능을 보여줍니다. 우리는 k=7 을 사용하여 실험을 진행합니다. 테이블 3은 비교의 결과를 보여줍니다. Faster R-CNN은 10-layer의 sub-network를 통해 각각의 region을 평가하고 좋은 성능을 보여줍니다. 반면 R-FCN은 per-region cost가 매우 적습니다. 300 ROI을 사용한 테스트에서 Faster R-CNN은 0.42s 의 속도를 가지며 R-FCN은 2.5배 빠른 0.17s per image의 속도를 가집니다.  R-FCN은 또한 Faster R-CNN보다 빠르게 학습 가능합니다. 게다가 OHEM은 R-FCN의 학습에 연산량을 추가하지 않습니다. 이는 R-FCN을 2000 ROI에서 훈련하는 것을 가능하게 하며 Faster R-CNN은 이 경우 6배 느린 속도를 보입니다. 그러나 실험을 통해 2000개의 ROI에 대해 OHEM을 적용하는 것이 이득이 없을음 보입니다. 그래서 우리는 트레이닝, Inference 모두에 300 ROI만을 사용합니다.

테이블 4는 더 많은 비교를 보여줍니다. [8] 에서의 multi-scale training 방법에 따라 우리는 각각의 training iteration 에서 인풋 이미지의 크기를 (400, 500, 600, 700, 800) 에서 임의로 골라 리사이즈 합니다. 테스트 타임에는 600 pixel scale을 사용함으로 테스트 타임의 연산량을 추가하진 않습니다. 이때의 map는 80.5% 입니다. 또한 R-FCN을 MS COCO trainval set에서 학습한 다음 PASCAL VOC 에서 fine-tune 하였습니다. 이 경우 83.6% map을 얻을 수 있었고 이는 Faster R-CNN++ 와 비슷한 결과를 보입니다. 우리는 iterative box regression, context, multi-scale testing을 사용한 Faster R-CNN+++ 보다 20 배 빠른 0.17s per image의 inference 속도로 경쟁력 있는 결과를 얻었습니다. 이러한 비교는 PASCAL VOC 2012 test set에서도 관찰되었습니다.

<b>On the Impact of Depth</b> 다양한 깊이의 Resnet을 사용한 R-FCN의 결과를 보여줍니다. 50에서 101 로 깊이를 증가함에 따라 성능이 향상되었으며 resnet-152 에서는 성능 향상이 없었습니다.

<b>On the Impact of Region Proposals</b> R-FCN는 Selective search, edge boxes와 같은 region proposal 방법을 적용 가능합니다. 다음은 다양한 proposal 방법을 적용한 결과를 보여줍니다. R-FCN은 SS, EB를 적용해도 경쟁령 있는 성능을 보였고 이는 R-FCN의 일반성을 보여줍니다.

### 4.2 Experiments on MS COCO

다음으로 우리는 80개의 카테고리를 가진 MS COCO 데이터셋에서 실험을 진행합니다. 우리의 실험은 80K train set, 40k val set, 20k test-dev set으로 구성되어져 있습니다. 90k 까진 0.001, 다음 30k 까진 0.0001 의 learing rate를 적용하였고 8개의 미니배치를 사용했습니다. 우리는 4-step alternating training을 5-step으로 확장하였고 shared feature를 사용할때 성능이 약간 향상됩니다. 또한 2-step training 방법이 좋은 성능을 보이는 것을 보고했지만 feature가 공유되지는 않습니다.

테이블 6 을 통해 결과를 보여줍니다. single-scale R-FCN 은 48.9%/27.6% 의 성능을 보입니다. 이는 Faster R-CNN baseline (48.4%/27.2%) 와 유사하지만 2.5 배 빠른 test time을 보여줍니다. R-FCN이 작은 object에 더 좋은 성능을 보인다는 것이 주목할 만 하며 multi-scale (yet single-scale test) R-FCN은 val set에서 49.1%/27.8%  test-dev set 에서 51.5%/29.2% 의 성능을 보였습니다. COCO 데이터의 다양한 object scale을 고려하여 [9] 에 따라 multi-scale testing 을 수행하였고 testing scale은 {200, 400, 600, 800, 1000} 을 사용했습니다. 이 경우 53.2%/31.5% 의 성능을 보였습니다. 이 결과는 1위의 결과 (Faster R-CNN+++ with Resnet-101 55.7%/34.9%) 와 근접합니다. 약간의 성능이 떨어짐에도 우리의 방법은 더 간단하고 [9] 에서 사용된 context, iterative box regression 과 같은 bells and whistle을 사용하지 않았고 traning, testing 속도가 더 빠릅니다.

### 5 Conclusion and Future Work

우리는 object detection을 위한 간단하지만 효율적인 region-based fully convolutional network를 제시했습니다. 우리 시스템은 Resnet과 같은 최신의 image classfication backbone을 적용하였고 이들은 fully convolutional 방식으로 설게되어져 있습니다. 우리의 방법은 Faster R-CNN과 비교해 경쟁력 있는 성능을 달성하고 training, testing 속도가 더 빠릅니다. 

의도적으로 논문에 기술한 R-FCN 시스템을 간단하게 기술하였습니다. semantic segmentation 을 위해 개발된 FCN을 적용하는 방법과 [9,1,22] 과 같은 방법의 object detection의 확장이 있었습니다. 우리의 R-FCN 시스템도 이러한 방법들의 효과를 누릴 수 있을 것이라고 기대합니다.


### 정리
1. position-sensitive pooling layer 를 도입했다. 
2. 원래 faster-rcnn+++ 는 conv4 conv5 사이에 pooling layer를 집어넣엇음 이는 만약 region proposal이 2000개라면 2000개에 대해 conv5연산을 다 해야 한다는 뜻.. 근데 position-sensitive pooling 을 사용함으로써 conv5 이후에 pooling 할수 있게함으로써 연산량을 줄임!!

### 더 읽어볼 논문 

1. a trous trick
2. FCN을 적용한 논문
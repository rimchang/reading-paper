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

Faster/Fast RCNN과 비슷한 bounding box regression을 수행합니다. k^2(C+1) dim 의 convolutional layer에 4k^2 dim의 bounding box regression layer를 추가합니다. position-sensitive ROI pooling은 이 4k^2의 map위에서 수행됩니다. 각 ROI에 대해 4K^2 dim vectore를 출력합니다. 그다음 average voting을 통해 4 dim vectore로 통합됩니다. 이 4 dim vecotre는 Fast RCNN 에서 나온 (t_x,t_y,t_w,t_h) 로 parameterize 합니다. 우리는 간단함을 위해 class-agnositc bounding box regression을 수행했지만 class-specific(4k^2C dim) 도 가능합니다.

이러한 position-sensitive score map의 개념은 [Instance-sensitive fully convolutional networks(3)] 에서 영감을 받았습니다. 우리는 더 나아가 position-sensitive ROI Poolingdmf object detection에 적용하였습니다. 여기에는 학습하는 weight가 재하지 않아 거의 cost-free region-wise computation을 가지며 training, inference 의 속도를 향상시킵니다.

### 정리
1. position-sensitive pooling layer 를 도입했다. 
2. 원래 faster-rcnn+++ 는 conv4 conv5 사이에 pooling layer를 집어넣엇음 이는 만약 region proposal이 2000개라면 2000개에 대해 conv5연산을 다 해야 한다는 뜻.. 근데 position-sensitive pooling 을 사용함으로써 conv5 이후에 pooling 할수 있게함으로써 연산량을 줄임!!


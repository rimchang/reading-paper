

# Mask-RCNN

### Abstract

object instance segmentation을 위한 간단하고, 유연하고, 일반적인 프레임워크를 소개한다. 우리의 접근은 하나의 이미지에서 효과적으로 object를 검출하고 동시에 각각의 instance에 대한 mask를 생성한다. Mask R-CNN 이라는 방법은 Faster R-CNN의 확장으로 mask를 생성하는 branch를 추가한다. Mask R-CNN은 학습이 간단하고 원래의 Faster R-CNN에 5 fps의 오버헤드밖에 추가하지 않는다. 게다가 Mask R-CNN은 다른 task로 일반화가 쉽다 (human pose estimation).  우리는 instance segmentation, bounding-box object detection, person keypoint detection COCO 데이터셋 대회에서 최고의 성능을 보였다. 다른 트릭이 없이 Mask R-CNN은 존재하는 모델 중에서 가장 월등한 성적을 보였다. 우리는 우리의 간단하고 효율적인 접근방법이 견고한 베이스라인을 제공하고 instance-level recognition에서 도움이 될 것을 희망한다.

### 1, Introduction

vision 그룹에서는 짧은 시간동안 object detection, semantic segmentation 의 월등한 성과가 있어왔다. 그중 강력한 baseline으로 Fast/Faster RCNN, FCN 등이 많은 기여를 했다. 이들 방법들은 개념적으로 이해하기 쉽고 유연하고 강건한 성능을 제공하며 빠른 학습, Inference 시간을 제공한다. 우리의 목표는 이러한 것들을 instance segmentation을 위한 모델로 만드는 것이다.

instance segmentation은 매우 어려운 task 인데 이미지 안의 모든 object를 검출을 해야 하며 각각의 instance마다 정확한 segmentation을 해야하기 때문이다. 그래서 <span style="color:RED">개별의 object를 classfy하고 bounding box를 이용해 localization하는 object detection과 instance의 구분 없이 각각의 pixel을 고정된 카테고리로 분류하는 semantic segmentation 방법을 섞는다.</span>

이러한 상황에서 누군가는 좋은 결과를 달성하기 위해 매우매우 복잡한 모델을 사용해야 할 것이라고 생각할 지도모른다. 그러나 매우 놀랍도록 간단하고, 유연하고 빠른 방법을 만들었으며 이는 기존의 state-of-art 방법들을 뛰어넘는다.

<span style="color:RED">Mask-RCNN이라고 불리는 방법은 Faster R-CNN을 확장한 것으로 각각의 ROI 마다 segmentation mask를 추가한 것이다.</span> 동시에 병렬적으로 classfication, regression 을 수행하는 가지가 존재한다. mask를 수행하는 가지는 작은 FCN으로써 각각의 ROI마다 적용이 되어 pixel-to-pixel로 mask를 예측을 한다. Mask R-CNN은 구현하기 간단하며 보다 유연한 아키텍쳐 디자인의 Faster RCNN 의 구조 안에서 학습이 된다. 게다가 mask를 수행하는 가지는 매우 적은 계산비용이 들어 빠른 시스템과 실험을 가능하게 한다.

근본적으로 Mask RCNN은 Faster RCNN의 확장으로써 아직 mask branch는 좋은 결과를 보이지 않았다???(모르겠다) 가장 중요한 것은 Faster RCNN이 pixel-to-pixel alignment의 인풋, 아웃풋 구조를 가지도록 설계되지 않았다. <span style="color:RED">이것은 instance를 구분하기 위한 가장 중요한 연산중 하나인 ROIPOOL 때문인데 이는 feature extraction을 위해 거친 spatial quantization을 수행한다. </span>이러한 misalignment를 고치기 위해 우리는 간단하고 <span style="color:RED">quantization-free 레이어인 ROIAlign</span> 을 제안한다. 이들은 정확한 spatial location을 유지하게 된다. 매우 작은것 처럼 보이는 변화만으로 ROIAlign은 매우 큰 효과를 가져오는데 mask accuracy를 10% 정도에서 50%로 향상시킨다.

<span style="color:RED">두번째로, 우리는 mask, class 를 예측하는 것을 분리하는 게 필수적이라는 것을 발견했습니다.</span> 우리는 각각의 class에 대해 독립적으로 binary mask를 예측했는데 이는 class별로 이루어진 것이 아니였고 단지 네트워크의 ROI classfication 에만 의존합니다. 반면에 FCN은 per-pixel multi-calss categorization(각 픽셀마다 클래스 갯수만큼을 예측한다는 것인듯) 을 수행하는데 이는 segmentation과 classfication이 결합된 형태입니다. 우리의 실험에서도 결합된 형태는 instance segmentation에 좋지 않은 결과를 보였습니다.

별다른 옵션이 없이 Mask RCNN은 COCO instance segmentation task에서의 다른 뛰어난 모델들을 뛰어 넘었고 물론 2016 대회의 승자를 포함합니다. 다른 부산물로 우리의 방법은 COCO object detection task 또한 뒤어 넘었습니다. 몇몇의 제외된 실험을 통해 multiple basic instantiations(다수의 object를 생성하는 것??) 을 평가했으며 이는 우리의 모델의 강건함과 주요한 효과를 분석하게 해줬습니다.

우리의 모델은 GPU에서 200ms로 수행가능하며 COCO 데이터 셋에서 1~2일 정도로 학습 가능합니다. (8-GPU machine) 빠른 train, test 스피드를 가진며 유연하고 정확한 프레임워크라고 생각하며 미래의 instance segmentation에 큰 도움이 될 것이라고 생각합니다.

마지막으로 COCO keypoint dataset을 통해 human pose estimation을 수행하여 우리의 프레임워크의 일반성을 보였습니다. 하나의 keypoint를 one-hot binary mask로 봄으로써 Mask-RCNN의 최소한의 변경으로 task를 수행합니다. 별다른 트릭이 없이 2016 COCO key-point competition을 능가하였고 5fps라는 속도로 수행합니다. 그러므로 Mask RCNN은 instance-level recognition을 위한 유연하고 광범위한 모데링라고 볼 수 있으며 다른 복잡한 task로 쉽게 확장 가능합니다.

### 2, Related Work

<b>R-CNN</b> : object detection을 위한 Region-based CNN의 접근 방법은 다양한 숫자의 object regrion을 다룰 수 있게 해줍니다. 또한 같은 ROI에 대해 독립적으로 convolutional network를 평가합니다?? RCNN은 ROIPOOL을 이용하여 ROI를 피쳐맵 상에서 구할 수 있게 해주고 매우 빠른 속도와 정확도를 가지게 합니다. Faster RCNN은 RPN이라는 attention mechanism을 통해 더 진보한 흐름입니다. Faster RCNN은 유연하고 강건하며 현재의 뛰어난 프레임워크입니다.

<b>Instance segmentation</b> : RCNN의 효율성에 영향을 받아 많은 instance segmentation 방법들은 segment proposal에 기반합니다. 13 15 16 9은 bottom-up segment를 수행합니다. DeepMask와 33,8 들은 segment candiates를 제안하고 이를 Fast RCNN으로 classfication 합니다. 이러한 방법들은 <span style="color:RED">recognition후에 segmentation이 수행되고 이는 느리고 정확도가 떨어집니다. (cascade 하도록 모델링 한 경우인듯?)</span> 10 과같은 방법들은 complex mulitple-stage cascade 방법을 제안하는데 bounding-box proposal로 부터 segment proposal을 예측합니다. 이때 box proposal은 classfication 이후 계산됩니다. <span style="color:RED">대신에 우리의 방법은 mask와 class label을 parallel 하게 예측하며 이는 더 간단하며 유연합니다.</span>

비교적 최근에는 FCIS(fully convolutional instance segmentation)이라는 segment proposal과 object detection이 결합된 형태가 제안되었습니다. <span style="color:RED">이들의 기본적인 아이디어는 fully convoltionally network를 이용해 position sensitive channel을 예측하는 것입니다.</span> 이러한 channel은 object class, boxes, mask를 동시에 예측하며 시스템을 더 빠르게 만듭니다.하지만 FCIS는 겹쳐진 instance에 대해 안좋은 결과를 보이며 그림5와 같이 이상한 edge를 만들게 됩니다. 이는 instance segmentation의 근본적인 어려움을 보여줍니다.

### 3, Mask RCNN

Mask RCNN은 개념적으로 간단합니다. Faster RCNN은 각각의 candidate object에 대해 두가지 출력을 가집니다. 하나는 class label, 하나는 bounding box offset 입니다. 우리는 여기에 3번째 branch를 추가하고 이의 출력은 object의 mask입니다. Mask RCNN은 자연스럽고 직관적인 아이디어 입니다. mask branch는 class, box branch와 다르게 더 정교한 spatial layout을 필요로 합니다. 다음은 Mask RCNN의 주요한 요소를 소개할 것인데 이는 pixel-to-pixel alignment를 포함합니다. 이는 Fast/Faster RCNN에는 포함되어있지 않습니다.

<b>Faster RCNN</b> : Faster RCNN의 간단한 리뷰로 부터 시작합니다. Faster RNN은 두단계로 구성되어져 있습니다. 첫번째 단계는 RPN이라고 불리는 단계이고 이는 candidate object bounding boxes를 제안합니다. 두번째 단계는 Fast RCNN의 부분으로써 각각의 candidate box로 부터 피쳐를 뽑아내고 classfication, bounding box regression을 수행합니다. 두 단계에서 사용되는 feature는 빠른 inference를 위해 공유됩니다. Faster RCNN와 다른 프레임워크 들의 포괄적인 비교들을 다루는 것을 21에 언급해놨습니다.

<b>Mask RCNN</b> : Mask RCNN 또한 두가지 단계로 이루어져 있습니다. 첫번째 단계는 RPN과 동일합니다. 두번째 단계는 class와 box offset을 병렬적으로 예측합니다. Mask RCNN은 각각의 ROI마다 binary mask를 출력합니다. 이러한 것들은 classfication이 mask prediction에 의존적인 최근의 시스템과는 반대입니다. 우리의 접근방법은 classfication, regression을 병렬적으로 수행하는 Fast RCNN의 정신을 따릅니다. ( multi-stage의 원래의 RCNN을 매우 간단히 만든 것처럼)

학습동안에는 각각의 ROI 마다의 <span style="color:RED">multi-task loss를 L_cls + L_box + L_mask로 정의합니다. classfication loss, bounding-box loss는 Fast RCNN과 동일합니다.</span> mask branch는 각각의 ROI마다 km^2의 차원을 가집니다. 이는 mxm 해상도를 가지는 K binary mask 입니다. (K 클래스중의 하나의 클래스에 대해서) 이를 per-pixel sigmoid를 적용하기 위해서 <span style="color:RED">L_mask 를 average binary cross-entorpy loss로 정의합니다.  class k와 관련된 하나의 ROI에 대해서 L_mask는 k-th mask에 대해서만 정의됩니다(다른 mask들은 loss에 관여하지 않습니다)</span>

<span style="color:RED">우리의 L_mask에 대한 정의는 class끼리 경쟁하지 않고 네트워크가 각각 class에 대해 mask를 생성하도록 합니다. 우리는 output mask를 선택하는데 사용하기 위한 전용의 classfication branch에 의존합니다.</span> 이것이 mask, class prediction을 분리한 것입니다. 이러한 점이 FCN을 semantic segmentation에 적용할 때와 다른 것인데. FCN은 per-pixel softmax와 multinomial cross entropy loss를 사용합니다. 이 경우 class간의 mask들은 경쟁을 하게 됩니다. 우리의 경우 per-pixel sigmoid, binary loss를 사용하여 이러한 일이 발생하지 않습니다. 이러한 구조가 좋은 결과를 낸다는 것을 실험으로 나타냈습니다.

<b>Mask Representation</b> : <span style="color:RED">하나의 mask는 하나의 object spatial layout을 부호화 한다. 그래서  fc layer에 의해 짧은 output vector로 붕괴되는  class label, box offset과 달리 mask의 spatial structure를 뽑아내는 것은 convolution에 의해 제공되는 pixel-to-pixel 대응으로 다뤄질 수 있다. (이 부분.. 잘 이해가 안간다) (fc layer를 거침으로써.. spatial information들이 없어지게 되는데.. 이를 fully convolution을 통해 spatial information들을 보존할 수 있다)</span>

특별히 우리는 mxm mask를 FCN 을 사용하여 예측한다. 이것은 각각의 mask branch가 lacks spatial dimension이 일어나는 백터 representation 없이 mxm object spatial layout을 유지하도록 한다. mask prediction을 위해 fc layer에 의존하는 이전의 방법들과 다르게.. 우리의 fully convolutional representation은 적은 파라미터를 필요로 하고 더 좋은 정확도를 가진다. (fc layer 대신 FCN을 사용하여 spatial dimension이 붕괴되는 것 없이 할수 잇다는 뜻인듯)

이러한 pixel-to-pixel의 동작은 매우 작은 feature map 을 가지는 ROI feature 이더라도 잘 aligned된 피쳐인 것을 요구한다. 이들은 per-pixel spatial correspondence를 정확하게 유지해야 한다. 이러한 동기는 ROIAlign layer를 개발하게 하는 동기가 되었고 이는 mask prediction의 중요한 요소로 작동한다

결국.. 하고싶은 말은 기존의 ROIPOOL은.. 정확한 spatial information을 유지할 수 없으므로.. ROIAlign 이라는 것을 만들게 되었다는 건데.. 왜이렇게 영어가 어렵냐..

ROIAlign : ROIPool 은 각각의 ROI에서 고정된 사이즈의 작은 feature map을 뽑아 내는데 사용되는 표준적인 연산이다.

1, ROIPool은 먼저 float number ROI를 discreate 한 거칠거칠한 feature map으로 양자화 한다.  
2,이 양자화된 ROI는 그들이 양자화 될 spatial bin으로 다시 나눠지게 된다.  
3, 마지막으로 각각의 bin 안의 feature value 들은 aggregate 되게 된다 (보통 max pooling)  
 
만약 feature map stride가 16이라면 continuous coordinate x는 [x/16] 과 같이 수행된다. [,]는 rounding function이다. 그리고 7x7 같은 bin으로 나눠지고 양자화가 일어나게 된다. 이러한 양자화는 ROI와 extrated feature 사이의  misalignments을 야기하게 된다. 이러한 것이 classfication 같은 small translation에 강건한 task에는 큰 영향이 없을지 모르지만 pixel-accurate mask 같은 것에는 매우 부정적인 효과를 일으킨다.

이러한 것을 다루기 위해 ROIPool의 거친 양자화를 제거하기 위한 ROIAlign 이라는 것을 제안한다. 이러한 것은 인풋과 함께 extracted feature를 재조정 하게 된다. <span style="color:RED">우리의 제안은 매우 간단한데 ROI 경계나 bin에서의 어떠한 양자화를 피한다.([x/16] 을쓰지 않고 x/16을 그대로 사용한다) 각각의 ROI bin에서 샘플링된 4개의 location에서 정확한 값을 계산하기 위해 bilinear interpolation을 사용한다. 그후 4개의 location을 max하거나 average한다.</span> ( 4개의 location을 샘플링하기 때문에 우리는 max or average pooling으로 평가할 수 있다. 사실 각 bin의 center에 있는 하나의 값으로 보간하는것은 매우 효과적이다. 4개이상의 location을 샘플링 하는 것은 성능이 안좋아 지는 것을 발견했다)

ROIAlign은 매우 큰 성능 향상을 보인다. 또한 ROIWarp 연산과 비교했다. ROIAlign과 다르게 ROIWarp는 alignment 문제를 간과했고 ROIPool 처럼 양자화 되도록 구현되었다. 그래서 ROIWarp가 bilinear resampling 될 수 있더라도 별로 좋지 않았다.


### Mask-RCNN 리뷰 영상

https://www.youtube.com/channel/UChWUtAAsYbTXe0jXGrhb-YQ/playlists

### 1, introduction
- segmentation : 하나의 이미지를 여러개의 part로 나누는 것, 하지만 각 part가 어떤 것을 나타내는지를 알려고 하지는 않는다. (part를 나누지만 각 part의 semantic 정보를 알려고 하지는 않는것)
- semantic segmentation : 하나의 이미지를 semantically meaningful part로 나누려고 한다. 그리고 각 part를 미리 정의된 class 중 하나로 classify 하는것. 하지만 같은 class의 다른 object/instance를 구별하지는 않는다.
- instance segmentation : 같은 class의 다른 object/instance 또한 구별하려고 한다.

### 2, Related work

<b>MNC</b>

1, Mask : 하나의 mask 안에 모든 클래스가 포함되어진다. (CNN을 이용하기 때문에 translation-invariant하다.)  
2, ROI-Warping 을 사용  
3, classfication이 mask에 의존적이다. (cascade task? mask를 한다음 mask를 이용하여 classfication 한다)  

<b>FCIS</b>

1,mask : position에 민감한 mask를 사용 (position에 민감함으로?? translation-variant 하다)  
2,  
3, join mask & classfication (그림에서 위쪽이 mask, 밑쪽이 classfication에 쓰이는 것이라고 함)  

<b>Mask RCNN</b>

1, Mask : 각 object에 대한 binary mask (하나의 object에 대해.. k개의 클래스만큼의 mask를 만든다)??  
2, ROIALign을 사용한다.  
3, branch를 통해 독립적으로 일어난다. 그림에서 위에 튀어나온 branch는 classfication, regression에 사용되고 뒤쪽은 mask에 사용되는듯

### 3, Architecture

Mask RCNN = Faster RCNN + FCN
FCN

7x7x4096 => 1x1 conv 를 거쳐서 7x7x21 의 feature map을 얻는다. 이 7x7x21 을 upsampling(bilinear interpolation) 해서 224x224x21 의 output을 얻게 된다. 여기서 21은 VOC 20 class + background

bilinear interpolation을 어떤식으로 하는지가 ... 명확하지가 않다.

#### FCN과의 가장 큰 차이

1, ROIAlign을 사용함으로써 mask accuracy를 10%에서 50%로 올릴수 있었음   
2, decouple mask and class prediction : mask branch는 각 ROI마다 K개의 MASK를 예측할 수 있지만.. k-th mask만을 사용한다. 이때 k는 classfication branch에서 예측된 class이다.

#### HOW ROI Pooling, ROI Warpng, ROI align work

https://www.youtube.com/watch?v=XGi-Mz3do2s&list=PLkRkKTC6HZMxZrxnHUDYSLiPZxiUUFD2C&index=4

1, ROI Pooling 이 예제에서,, 665x665 이미지가 665/32 = 20.78 인데 버림해서 20x20 의 feature map을 가지게 된다 이때 0.78 이라는 값이 손실되게 되는데.. 이를 원본 이미지에서 손실량을 살펴보면 0.78*32=24.96 의 픽셀이 손실되게 된다.

또한 이를 pooling 할때도.. 엄청난 픽셀이 손실되게 된다. 2단계의 양자화를 거치면서 손실량이 매우 커지게 된다.

2, ROIAlign : 여기서는 floating value를 그대로 사용하게 된다. 그렇다면 어떻게 floating box를 가지고 feature 맵을 계산하냐? bilinear inteopolation을 사용하여 계산을 하게된다. 각각 그리드의 4귀퉁이를 bilinear 를 통해 샘플링을 하게 된다. 그다음 4귀퉁이를 max or average pooling 해서 그리드에 할당??(정확하지 않다)

### 4, implementation details

1, Faster RCNN의 하이퍼파라미터를 그대로 사용  
2, 기본 구조 : Resnet50, resnet101, FPN  
3, 가장 짧은 사이드를 800로 맞춤  
4, 8GPU, 2장씩 사용  
5, 32(RESNET50, FPN), 44(RESNET101) end-to-end 사용하지 않음  
6, 한장의 이미지당 195ms가 걸림 + 15ms(resizing)  
7, COCO dataset 사용  


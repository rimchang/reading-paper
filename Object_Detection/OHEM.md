# Training Region-based Object Detectors with Online Hard Example Mining

### Abstract

object detection 영역은 region-based Convnet의 영향을 많이 받고 있습니다. 그러나 이들의 training 과정은 여전히 많은 휴리스틱과 조정해야하는 하이퍼파라미터가 있습니다. 우리는 간단하지만 놀라운 효과를 끼치는 online hard example mining(OHEM) 이라는 것을 소개합니다. 우리의 동기는 detection dataset이 많은 수의 easy example과 적은 수의 hard example을 포함하고 있는 것과 같습니다. 자동적으로 이러한 hard-example을 서택하는 것은 트레이닝을 더 효과적으로 만들 수 있습니다.OHEM은 보통 사용되는 휴리스틱과 하이퍼파라미터를 제거할 수 있는 간단하고 직관적인 알고리즘입니다. 더 중요한 것은 이러한 알고리즘이 PASCAL VOC 2007, 2012에서 성능의 향상을 보였습니다. 알고리즘의 효과는 MS COCO 같은 보다 크고 복잡한 데이터셋에서 많은 증가를 보였습니다. 

### Introduction

Image classfication과 object detection은 컴퓨터 비전에서의 두가지 근본적인 task 입니다. object detection은 일종의 reduction을 통해 종종 학습이 되어지는데 이러한 reduction은 object detection문제를 image classfication 문제로 다루는 것입니다. object detection을 reduction하여 다루는 것은 새로운 문제를 야기하는데 그것은 training set에 foreground, background exmaple의 매우 큰 불균형입니다. DPM 같은 sliding-window object detector의 경우 이러한 불균형은 하나의 object당 100000개의 background example정도 될 수도 있습니다. 최근의 object-proposal-based detecotr(Fast, Faster RCNN)의 경우 이를 완화시켰지만 여전히 70:1 정도의 불균형을 보입니다. object detection을 위한 불균형을 완화시키고 빠른 학습, 높은 성능을 가지는 learning technique은 오픈 이슈로 남아 있습니다.

hard negative mining은 새로운 방법이 아닌 standard solution입니다. 이는 원래 bootstrapping이라는 20년 전부터 존재하던 방법입니다. object detection을 위한 bootstrapping은 1990년 중반에 sung and poggio에 의해 소개되었습니다. 그들의 주요 아이디어는 false alarm을 유발하는 background example을 점차적으로 증가(bootstrap) 하는 것입니다. 이러한 전략은 현재의 고정된 traning set만을 이용하는 것이 아닌 iterative training 알고리즘을 야기하는데 bootstrapped training set을 통해 업데이트된 모델이 false positive(false를 positive로 분류 object detection에서는 background or other object를.. object라고 분류.)를 더 잘 찾도록 하는 것입니다. 이러한 과정은 일반적으로 모든 object example과 small, random background example로 구성된 training set에서 시작합니다.

bootstrapping은 object detecion 연구에서 많이 사용되어져 왔습니다. Dalal and Triggs는 pedestrian detection을 위한 SVM을 학습시킬때 사용했습니다. Felzenszwalb 는 전체의 데이터셋에 정의된 SVM의 global optiaml solution에 도달하기 위한 bootstrapping의 형태를 제공합니다. 이들의 알고리즘은 hard negative mining이라고 불리며 SVM을 학습시킬때 자주 이용됩니다. Bootstrapping은 다른 모델을 학습시킬때도 자주 적용됩니다. shallow neural network, boosted decision tree에 사용되었습니다. R-CNN, SPPnet 같은 최근의 detection model에도 SVM을 학습시킬때 사용되었습니다.

최근의 state-of-the-art obejct detector인 Fast, Faster RCNN에서 bootstapping을 사용하지 않는 것은 이상해 보일 수 있습니다. <span style="color:RED">이들이 bootstrapping을 사용하지 않는 이유는 SGD를 이용하는 Convnet 학습과정이 online learning 알고리즘을 적용하기 때문에 bootstrapping을 사용하기 힘듭니다.</span> Bootstrapping과 이들의 변형들은 앞에서 말한 대안적 탬플릿에 의존합니다. (a) 어떤 기간동안 고정된 모델이 새로운 example을 추가하는데 이용되어야 합니다. (여기서 고정된 모델이라는 것이 중요한듯) (b) 그다음 어떤 기간동안 그 모델이 새로운 fixed active training set에 의해 학습되어야 합니다. SGD를 이용한 Convnet detector의 학습은 일반적으로 매우 많은 SGD 단계를 필요로 하고 일정 기간동안 모델을 고정하는 것은 매우 느린 속도를 야기합니다. 대신에 우리가 필요한 것은 purely online form of hard example selection 입니다.

이 논문에서는 우리는 OHEM 이라고 불리는 online hard example mining 이라는 새로운 bootstrapping 기술을 제안합니다. 이 알고리즘은 현재의 loss에 의존하는 non-uniform, non-stationary 분포를 에서 샘플링된 training example을 사용하는 SGD를 수정한 것입니다. 이 방법은 하나 혹은 2개의 batch image이지만 수천개의 candidate example 을 가지는 detection-specific problem 에서 매우 큰 장점을 가집니다. 이 candidate example은 favor diverse, high loss instance에 의해 subsampled 됩니다.?? backpropagation은 여전히 모든 candidate의 작은 부분집합 만을 사용하기 때문에 효율적입니다. 우리는 보통의 Fast RCNN 에 OHEM 을 적용하였고 세가지의 장점을 보여줍니다.

- region-based Convnet에서 사용되던 휴리스틱과 하이퍼파라미터를 제거합니다.

- 안정적이고 높은 map의 향상을 보입니다.

- MS COCO 데이터셋 같은 더 크고 복잡한 training set에 더 큰 효율성을 보입니다.

게다가 이러한 장점은 multi scale testing, iterative bounding box regression 같은 방법과 상호 보완적입니다. 이들 트릭을 모두 사용하여 OHEM은 PASCAL VOC 2007(78.9%), 2012(76.3%) 의 성능을 보였습니다.

### Related work

object detection 은 오래되고 가장 근본적인 컴퓨터 비전 문제중 하나입니다. 데이터셋의 bootstrapping은 일반적으로 hard negative mining이라고 불리며 다양한 object detector에서 나타납니다. 이들은 detection scoring을 위해 SVM을 사용하고 feature extraction을 위해 Convnet을 사용했습니다. Fast RCNN, Faster RCNN은 SVM을 이용하지 않았는데 이 때문에 hard example mining 방법을 적용하지 못했습니다. 이 논문에서는 이러한 문제를 online hard example mining 알고리즘을 통해 다뤄 optimization, detection accuracy을 향상시킵니다. 우리는 간단한 hard example mining과 modern Convnet을 기반으로한 object detection에 대한 소개를 하며 둘을 같이 사용한 논문에 대한 소개를 할 것입니다.

<b>Hard example mining</b> 일반적으로 많이 사용되는 두가지 종류의 hard example mining 알고리즘이 있습니다. 첫번째는 SVM을 사용할때의 방법입니다. 이 경우에는 먼저 working set을 고정해 놓고 SVM을 수렴시킵니다. 그다음 working set을 업데이트 하고 다시 SVM을 학습시킵니다. 업데이트를 위한 룰은 다음과 같습니다. 현재의 model의 margin에 의해 잘 분류되는 easy-example은 working set에서 제거합니다. 반대로 현재의 model의 margin에 의해 어렵다고 판단되는 hard-example을 추가시킵니다. 이러한 규칙은 global SVM solution에 수렴합니다. 중요한 것은 이러한 working-set은 전체 학습 데이터의 일부분 입니다.

두번째 방법은 shallow neural network, boosted decision tree에 사용되는 non-SVM 방법입니다. 이 알고리즘은 positive example과 random negative example로 부터 시작합니다. 모델은 이 데이터셋에 먼저 수렴시키고 그 다음 더 많은 false postivie를 가진 큰 데이터셋에 학습이 됩니다. false postivie 들은 그 뒤에 training set에 추가가 되며 모델은 다시 학습됩니다. 이러한 과정은 한번 반복되며 수렴한다는 증명은 되지 않았습니다.

<b>Convnet-based object detection</b> 지난 3년 동안 object detection 분야에서 큰 성과가 있었습니다. 이러한 향상은 Imagenet classfication에서의 deep Convnet 때문에 이뤄질 수 있었습니다. RCNN, Overfeat detector는 이러한 흐름을 이끌었고 PASCAL VOC, Imagenet detection에서 큰 성과를 냈습니다. Overfeat는 sliding-window detection 방법입니다. 반면 RCNN은 selective search 같은 region proposal 방법을 사용합니다. RCNN과 같은 방법들 때문에 우리의 논문이 만들어졌습니다.

<b>Hard example selection in deep learing</b> [22, 27, 33] 같은 우리의 작업과 비슷한 딥러닝을 위한 hard example 방법이 있습니다. 우리의 접근방법과 비슷하게 이들의 방법은 그들의 selection을 각 데이터 포인트의 현재의 loss에 기반을 둡니다. [27] 은 독립적으로 그들만의 loss를 통한 hard positive, negative example을 선택합니다. positive pair가 주어졌을때 [33] 은 triplet loss를 사용해 hard negative pathc를 찾습니다. 우리의 방법과 비슷하게 [22] 은 미니배치 SGD 방법을 위한 hard example 방법을 제안합니다. 그들의 selection은 loss를 기반으로 하지만 image classfication을 중점으로 둡니다. 우리는 region-based object detection을 위한 hard example selection에 중점을 둡니다.


### Overview of Fast RCNN

우리는 먼저 Fast RCNN 프레임워크를 요약합니다. FRCN은 하나의 이미지와 ROI를 인풋으로 받습니다. FRCN 네트워크는 두가지 부분으로 나눠집니다. 몇가지 convolution, max-pooling으로 이루어진 conv network와 ROI-pooling과 fc layer들로 이루어진 ROI 네트워크로 나눠집니다. 

inference 동안 conv network는 인풋 이미지의 차원에 따라 달라지는 conv feature map을 생성합니다. 그다음 object proposal을 위해 ROI-pooling layer는 conv-feature map을 fixed-length feature vector로 바꿉니다. 각각의 feature vector는 fc layer의 인풋으로 들어가며 classfication score, regression coordinate를 출력합니다.

우리가 base object detector로 FRCN을 선택한 이유는 다음과 같습니다. 첫번째로 이러한 두 단계의 네트워크가 SPPnet, MR-CNN등에서도 사용되었습니다. 그러므로 우리가 제안한 알고리즘 또한 SPPnet, MR-CNN등에 사용가능합니다. 두번째로 기본적인 가정이 비슷합니다. FRCN은 전체 conv network를 학습하는 것을 허락합니다. 반면에 SPPnet, MR-CNN은 conv net을 고정하여 사용합니다. 마지막으로 SPPnet, MR-CNN은 분리된 SVM classfier를 학습하게 위해서 feature를 cache하는 것을 필요로 합니다. 여기에는 online hard negative mining이 필요없고 그냥 hard negative mining을 사용하면 됩니다. FRCN은 ROI network 그 자체를 classfier를 학습하는데 사용합니다. [Fast RCNN 14]의 논문에서는 마지막 단계의 SVM classfier가 필요없다는 것을 보여줍니다.

### Training

많은 deep network와 비슷하게 FRCN은 SGD를 통해 학습됩니다. 각 ROI당 loss는 object or background를 잘 예측했는지에 대한 classfication log loss와 정확한 bounding box를 예측하도록 격려하는 localization loss로 구성되어져 있습니다.

conv network의 computaion을 공유하기 위해서 SGD 미니배치는 계층적으로 만들어져 있습니다. 각 미니배치 동안 N개의 이미지가 먼저 샘플링 됩니다. 그 다음 B/N 개의 ROI가 각 이미지에서 샘플됩니다. Fast rcnn에서는 N=2 B=128 로 설정되었습니다. ROI를 샘플링 하는 과정은 약간의 휴리스틱으로 구성되어져 있는데 그것은 밑에 설명합니다. 우리의 공헌중 하나는 이러한 휴리스틱과 그들의 하이퍼파라미터를 제거한 것입니다.

<b>Foreground ROIS</b> ROI들중 foreground라고 라벨링 되는 것들은 ground-truth box와의 IOU가 0.5 이상되는 ROI 입니다. 이는 PASCAL VOC object detecion의 evaluation에서 사용된 수치입니다. 다른 논문에서도 같은 수치를 사용했으며 우리도 똑같은 수치를 사용합니다.
 
 <b>Background ROIS</b> ROI중 background라고 라벨링 되는 것은 gt box와의 IOU가 [0.1 , 0.5) 사이의 ROI들입니다. FRCN, SPPnet 에서 0.1을 low-backgorund threshold로 사용하였습니다. <span style="color:RED">FRCN 에서의 가정은 0.1 이라는 수치가 hard negative mining의 근사치라는 것입니다. 그들의 가정은 ground truth와 좀 겹치는 region이 더 hard, confusing example일 것이라는 것입니다. </span> 우리는 5.4 섹션에서 이러한 휴리스틱이 모델의 수렴과 accuract에 도움이 된다는 것을 보입니다. 하지만 이는 suboptimal 인데 왜냐하면 [0.1 0.5) 라는 수치는 많지는 않지만, 중요하고 어려운 background region을 무시하기 때문입니다. 우리의 방법은 low-backgorund threshold을 제거합니다. (0.1 을 low-threshold로 사용하기 때문에.. 사용되지 않는 background중에 중요한 애들이 있을 수 있기에 suboptimal일 뿐이라는 소리인듯)

 <b>Balancing fg-bg ROIS/b> fg-bg의 불균형을 다루기 위해 적용된 휴리스틱은 fg-bg를 1:3 비율로 rebalance하느 것입니다. 이때 background pathces를 랜덤으로 undersampling 합니다. 그래서 미니배치안의 25%가 fg ROI가 되도록 합니다. 우리는 이가 FRCN에서의 주요한 하이퍼 파라미터라는 것을 알아냈습니다. 이를 제거하거나 증가하면 map가 3% 가량 낮아졌습니다. 우리의 제안에서는 이러한 하이퍼파라미터를 제거할 수 있고 더이상 ill effect를 일으키기 않습니다.
 
 ### Our approach
 
 우리는 간단하지만 효과적인 OHEM을 소개합니다. 이는 Fast R-CNN의 학습에 효과적이며 다른 Fast R-CNN 스타일의 알고리즘에 적용 가능합니다. 우리는 기존의 미니배치를 생성하는 방법이 효과적이지 않고 suboptimal에 도달한다는 것을 보였고 우리의 방법을 기술할 것입니다.
 
 ### Online hard example mining
 
 hard example mining을 위한 번갈아 수행되는 두가지 방법을 기억해봅시다. (a) 일정기간 동안 fixed model이 active training set에 추가할 새로운 example을 찾아냅니다.
 (b) 그다음 일정기간 동안 active training set을 가지고 모델을 학습시킵니다.
 
 R-CNN, SPPnet에서 SVM을 학습시키는 방법은 다음과 같습니다. 
 (a) 10개 혹은 100개 정도의 숫자의 이미지를 active traing set이 threshold size에 도달할때 까지 탐색합니다.
 (b) active set을 가지고 SVM을 학습시킵니다. 이러한 과정은 active training set이 모든 support vector를 포함할때 까지 반복합니다. 이러한 방법을 FRCN에 적용하면 학습속도가 느려집니다. 이 이유는 10개 혹은 100개의 이미지를 탐색하며 example을 sampling 하는 동안 학습이 일어나지 않기 떄문입니다.
 
 우리의 주요한 관찰은 이러한 교차되는 단계가 어떻게 SGD를 이용한 FRCN 학습과 결합할 수 있을지입니다. <span style="color:RED">주요한 키는 각 SGD의 iteration이 작은 숫자의 이미지에서 일어나지만 각 이미지가 각 이미지가 포함한 ROI가 휴리스틱하게 샘플링된 것이 아닌 우리가 hard example로 선택하도록 하는 것입니다. </span>이러한 전략을 하나의 미치배치에 SGD를 "freezing" 모델로 취급할 수 있습니다. 그래서 모델은 기존의 SGD 접근방법과 같은 빈도로 업데이트 됩니다.
 
 더 구체적으로는 OHEM은 다음과 같이 진행됩니다. 
 
 1. t번째 SGD iteration에서의 하나의 인풋 이미지에 대해 먼저 conv feature map을 계산합니다. 
 2. ROI network는 위의 feature map을 이용해 모든 ROI를 계산합니다. 휴리스틱한 미니배치에 대해서가 아닌 모든 ROI를 계산합니다.
 
 2번째 스탭은 오직 ROI-pooling, fc layer, loss 계산만을 포함하는 것을 기억하세요. loss는 각 ROI에 대해 현재의 네트워크가 얼마나 잘 작동하느냐는 나타냅니다. 
 3. Hard example은 각 ROI를 loss로 sorting합니다. 그 후 B/N 개의 네트워크의 loss가 높은 example을 선택합니다.
 
 대부분의 forward computation은 conv feature map을 통한 ROI를 통해 공유됩니다. 그래서 추가적인 computation은 매우 적은 ROI에 대한 forawrd 입니다. 게다가 적은 ROI만이 선택되어 모델의 업데이트에 사용이 되므로 backward pass의 계산은 전과 같습니다.
  
  그러나 약간의 문제가 있는데 비슷한 위치의 많이 겹친 ROI는 매우 높은 correlated loss를 가진다는 것입니다. 게다가 이러한 겹쳐져 있는 ROI는 conv feature map의 같은 region에 해당할 수 잇습니다. (ROI-pooling 과정의 해상도 문제 때문에) 이는 loss double counting을 일으킬 수 있습니다. 이러한 redundant, correlated region을 다루기 위해 우리는 이러한 것들을 제거하기 위한 nms를 수행합니다. ROI와 그들의 Loss의 리스트가 주어진다면 NMS는 가장 높은 loss를 가지는 ROI부터 시작하여 선택된 ROI와 많이 겹치는 낮은 Loss를 가지는 ROI들을 제거합니다. 우리는 이때 0.7의 임계값을 적용했습니다.
  
  우리는 이러한 과정이 더이상 데이터 불균형을 위한 fg-bg 의 비율이 필요없어지는 것에 주목합니다. 만약 어떤 class가 잘 분류가 안되면? 그 loss가 매우 높을 것이고 이는 더 샘플링될 가능성이 높을 것입니다. fg-ROI가 매우 쉬운 경우가 있을 수 있습니다. (car 이미지의 매우 표준적인 각도) 이런 경우 네트워크는 미니배치를 bg-region으로 자유롭게 사용할 수 있습니다. 그리고 만약 bg가 매우 전형적인 (하늘, 잔디) 같은 것이라면 모든 미니배치가 fg region을 위해 선택 될 수 있습니다.
  
  ### Implementation details
  
  FRCN detectore의 많은 OHEM이 존재 합니다. 각각은 다른 trade-off를 가집니다. 하나의 명백한 방법은 loss layer를 hard example selection을 위해 변경하는 것입니다. loss layer를 모든 ROI에 대해 계산하도록 할 수 있고 hard ROI를 선택하기 위해 Loss로 sorting 합니다. 그다음 non-hard ROI의 loss를 0 으로 만듭니다. 이러한 간단한 방법으로 인해 이 구현은 ROI network는 여전히 메모리를 할당하고 모든 ROI에 대해 backward pass를 수행합니다. 하지만 Loss가 0 인 ROI는 gradiant 업데이트가 일어나지 않을 뿐입니다. (현재의 딥러닝 프레임워크의 한계이다)
  
  이를 극복하기 위해서 우리는 Figure 2와 같은 구조를 제한합니다. 우리의 구현은 두개의 ROI nerwork를 유지합니다. 하나는 오직 readonly-network일 뿐입니다. 이것이 의미하는 것은 readonly ROI network는 기존의 네트워크와 다르게 오직 forward pass만을 수행한다는것입니다. 한번의 SGD iteration에 대해서 conv feature가 주어져 있다면 read-only network는 forawad pass를 수행하고 모든 ROI에 대해 loss를 계산합니다. 그 다음 hard example을 선택하여 이는 평범한 ROI-network의 인풋으로 들어갑니다. 이 네트워크는 선택된 hard-example 에 대해서만 forward, backward pass를 계산하고 그라디언트를 축적하고 backward pass를 수행합니다. 실용적으로는 우리는 N개의 이미지에서 R개의 ROI를 뽑았고 readonly ROI network의 배치사이즈는 R 평범한 ROI network의 배치사이즈는 B 입니다.
  
  우리는 두가지 방법을 모두 Caffe를 통해 구현했으며 우리의 구현은 하나의 미니배치에 대해 N번의 forward-backawrd를 수행합니다. FRCN에 따라 우리는 N=2, B=128을 사용했으며 이러한 설정에 따라 제안된 구조는 첫번째 옵션과 비슷한 메모리 사용을 가집니다. 하지만 두배더 빠릅니다. 
  
  ### Analyzing online hard example mining
  
  이 섹션은 FRCN과 OHEM을 적용한 것을 비교합니다. 또한 우리는 OHEM을 사용한 FRCN과 모든 ROI를 사용한 것과 비교합니다. (B개의 hardest example을 사용하지 않고)
  
  ### Experimental setup
  
  우리는 두가지 표준적인 Convent 구조에 대해 실험을 진행 했습니다. Alexnet의 wider version인 VGGM, 그리고 VGG16 입니다. 모든 실험은 PASCAL VOC07 데이터셋에서 수행되었습니다. 학습은 trainval set, test는 test set에 진행되었습니다. 여기서 기술되지 않은 다른 것은 모두 FRCN의 세팅을 따랐습니다. 우리는 모든 방법을 80K 미니배치만큼 학습시켰고 초기의 lr은 0.001로 설정하고 30k마다 0.1씩 내렸습니다. baseline에 대한 보고는 테이블1에 되어져 있고 다른 방법에 비해 좀 더 좋은 성능을 냈습니다.
  
  ### OHEM vs heurisitc
  
  표준적인 FRCN은 0.1의 bg_lo를 사용하여 휴리스틱 hard mining을 합니다. 이러한 휴리스틱의 중요함을 테스트하기 위해 우리는 FRCN을 bg_lo=0 으로 수행했습니다. 그 결과 VGGM은 map가 2.4 정도 떨어졌고 VGG16은 거의 비슷했습니다. OHEM과 비교한 결과 OHEM은 bg_lo=0.1 보다 2.4 가량의 map을 올릴 수 있었습니다. 이 결과는 휴리스틱한 버젼이 sub-optimal 이라는 것을 보여주고 우리의 hard mining 방법이 효과적이라는 것을 보여줍니다.
  
  ### Robust gradient estimate
  
  N=2 이미지를 사용해서 하나 걱정되는 것은 unstable gradient, slow convergence 였습니다. 이는 ROI의 highly correlated가 유발할 수 있습니다. (원래 미니배치는 각 샘플이 상관이 없어야 한다) FRCN은 이는 실무적으로는 별로 문제가 되지 않는다고 보고했습니다. 하지만 우리의 학습과정에서는 우리가 같은 이미지안에서 high loss example을 사용하므로 문제가 될 수 있습니다. 이러한 것들 다루기 위해 우리는 N=1에서 부터 점점 증가시키며 실험을 진행 했습니다. 그 결과 N=1 에서 1% 정도의 map가 떨어졌습니다. 하지만 우리의 학습과정에서는 map가 거의 같았습니디ㅏ. 이러한 것은 OHEM이 매우 강건하고 gpu 메모리를 줄이기 위해 낮은 N을 사용해도 된다는 것을 보여줍니다.
  
  ### Why just hard examples, when you can use all?
  
  OHEM 은 이미지의 모든 ROI를 고려하는 것이 중요하고 그다음 hard example을 고려하는 것이 중요하다는 가정입니다. 그러나 만약 우리가 모든 ROI를 트레이닝에서 고려한다면? not just the hard one? easy example은 low loss를 가져올 것입니다. 그리고 이는 강한 gradient를 가져오지 않습니다. 학습과정은 hard example에 중점을 둬야 합니다. 이러한 것을 비교하기 위해서 우리는 FRCNN을 매우 큰 B=2048, bg_l0=0으로 실험했습니다. 여기서 매우 큰 미니배치를 가지기 때문에 lr를 바꾸는 것이 중요합니다 우리는 VGG16에 대해 0.003, VGGM에 대해 0.004를 사용했습니다. 결과는 테이블 1에 기술되어져 있습니다. VGG16, VGGM 모두 map가 1정도 증가했습니다. 하지만 우리의 접근 방식이 모든 ROI를 사용하는 것보다 1% 정도 더 높습니다. 게다가 우리의 방법이 더 빠릅니다. (여기서 말하고자 하는 것은 물론 all ROI를 고려하면 더 좋겠지만.. 실제로 해보니 별로다. 그리고 hard example만 고려하는게 더 빠르다!!)
  
  ### Better optimization
  
  마지막으로 우리는 다양한 FRCN 학습방법들과 training loss를 분석했습니다. 이것이 중요한 것은 샘플링 과정에 의존하지 않고 그래서 다른 방법들과 올바른 빅가 가능합니다. 이를 하기 위해서 우리는 매번 20k 마다 모델을 저장하고 trainval set에 대해 average loss를 계산했습니다. 이 것은 샘플링 방법에 의존하지 않습니다.
  
  Figure 3는 VGG16의 average loss를 보여줍니다. bg_lo=0이 가장 높은 loss를 보이는 것을 볼 수 있습니다. 반면에 bg_lo =0.1은 더 낮은 loss를 보입니다. B=2048의 경우는 bg_l0=0.1 보다 낮은 loss를 보입니다. 우리의 OHEM은 가장 낮은 loss를 보였고 OHEM이 다른 학습 방법들 보다 낫다는 것을 보여줍니다.
  
  ### Adding bells and whistles
  
  우리는 OHEM을 적용함으로써 일관된 성능향상을 보여줬다 이번 섹션에서는 다른 bells and whistels 을 통해 성능을 높이는 것을 보여준다. OHEM 은 두가지 방법을 사용했다.
  
  <b>Multi-Scale</b> 우리는 SPPnet 에서 적용된 multi-scale 방법을 적용했다. scale은 이미지의 가장 작은 side로 정의되어진다. 학습동안 하나의 scale이 random하게 선택되고, test time에서는 모든 스케일에 대해 진행되게 된다. VGG16 네트워크에서 우리는 480, 576, 688, 864, 900을 학습동안 사용하고 480, 576, 688, 864, 1000 을 테스팅 동안 사용했다. 이러한 scale과 cap은 gpu 메모리 제약 떄문에 선택된 것이다.
  
  <b>Iterative bounding-box regression</b> bbox voting 방법을 적용했다. 네트워크는 각각의 proposal ROI에서 score를 얻고 R_1 box를 relocalization 하여 R_2 boxes를 얻는다. 이 R_1, R_2의 합집합을 마지막 R_F로 사용한다. 이를 0.3의 임계값의 NMS을 통해 R_F_NMS을 얻고 각각의 box r_i에 대해 weighted voting을 수행한다. 
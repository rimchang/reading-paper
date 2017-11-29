# T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos

### Abstract

지난 2년동안 object detection을 위한 최신 기술들의 성능이 크게 개선되었습니다. GoogleNet, VGG와 같은 강력한 뉴럴네트워크의 도입과 R-CFNN, Fast, Faster RCNN 과 같은 모델의 성공은 성능의 향상에 필수적인 역할을 했습니다. 이들이 still image에 매우 효과적이지만 비디오에서의 object detection을 위해서 설계되진 않았습니다. 비디오에서의 temporal, contextual information은 이러한 모델에서 활용되지 않았습니다. 이 논문에서는 우리는 비디오에서 얻어진 tubelet을 통해 temporal, contextual information을 활용 할 수 있는 딥러닝 프레임워크를 제안합니다. 이는 기존의 still-image detection의 성능을 크게 올렸습니다. 이는 T-CNN 이라고 불리며 tubelet을 활용한 CNN 네트워크 입니다. 제안된 프레임워크는 2015 이미지넷 VID task 에서 우승을 하였으며 코드는 공개가 되어있습니다.

### 1 INTRODUCTION

지난 2년 동안 새로운 CNN구조와 다양한 object detection 프레임워크의 성공에 따라 object detection의 성능이 급격하게 향상되었습니다. R-CNN이라는 최신의 프레임워크와 그의 확장들 [4,5] 는 region proposal 에서 conv feture를 추출하고 proposal의 라벨을 예측합니다. DeppID-Net[8] 은 R-CNN을 box pre-traing, cascade on region proposal, deformation layer, context represenation 등을 통해 성능 올렸으며 최근에 ImageNet은 새로운 챌린지인 VID 를 도입하였고 이는 비디오에서의 object detection을 다루게 됩니다. 이 챌린지에서 object detection 시스템은 비디오의 모든 프레임에서 객체에게 라벨을 할당하고 bounding box를 그려야 합니다. VID는 비디오 분석의 다양한 분야에 활용됩니다.

still 이미지에서의 이들의 성공에도 불구하고 이러한 still-image object detection 프레임워크는 비디오를 위해 디자인 되지는 않았습니다. 비디오에서의 핵심 요소중 하나가 temporal information 이며 비디오에서의 object의 location, appearance는 반드시 temporally consistent 해야 합니다. 예를 들어 detection 결과인 bounding box location, detection confidecne가 시간에 따라 급격하게 변해선 안됩니다. 그러나 still-image object detection 프레임워크를 비디오에 바로 적용하면 인접한 프레임 사이의 detection confidence가 급격한 변화를 보이며 large long-term temporal variation이 존재합니다. 이에 대한 예들은 피규어 1에 보여집니다.

temporal consistency를 향상시키기 위한 우리의 직관은 detection 결과를 인접 프레임에게 전달시켜 detection result의 갑작스러운 변화를 줄이는 것입니다. 만약 특정 프레임에 하나의 object가 존재한다면 인접 프레임은 비슷한 위치에 같은 object를 가지고 있을 가능성이 높으며 비슷한 confidence를 가질 가능성이 높습니다. 다른 말로 하면 detection reuslt가 motion information에 따라 인접한 프레임으로 전파되어 잘못된 detection을 줄여 줄 수 있습니다. 이러한 결과로 나온 복제된 box 들은 nms를 통해 쉽게 제거할 수 있습니다.

temporal consistency를 향상시키기 위한 다른 직관은 detection 결과에 long-term 제약을 부여 하는 것입니다. 피규어 1에서 볼 수 있듯이 bounding box의 sequecne에 대한 detection score는 매우 큰 변동을 가지고 있습니다. 이러한 box seqeuecne, 즉 tubelet은 tracking 혹은 spatio-temporal object proposal 알고리즘[9]에 의해 생성 될 수 있습니다. tubelet은 하나의 단위로 취급되기 위해 long-term 제약을 적용할 수 있고 몇몇의 positive bounding box에 대한 낮은 detection confidence는 moving blur, bad posse, 특정한 pose에 대한 트레이닝 셋의 부족 등 때문에 일어날 수 잇습니다. 그러므로 tubulet 안의 대부분의 bounding box가 높은 confidence detection score를 가진다면 그 안의 몇몇 low confidence score를 가지는 box들은 tubelet의 long-term consistency 때문에 높아져야만 합니다.

temporal information외에도 contextual information이 비디오에서의 주요한 요소기도 합니다. image context information 들이 연구되었고 still-image detection 프레임워크에 사용되기도 하였지만 비디오의 경우는 수백장의 이미지의 집합이기 때문에 훨씬 풍부한 contextual information을 가지고 있습니다. 피규어 1(b) 를 살펴보면 비디오에서의 몇몇 프레임은 background object라고 잘못 분류 (false positive) 할 수 있습니다. (그림에서 2번째를 보면 거북이가 있는데 검출을 하지 못했음 즉 background라고 분류했다)  single 프레임에서의 contextual information은 이러한 false positive를 알아내기엔 충분하지 않습니다. 그러나 비디오 클립 내에서의 대부분이 high-confidence detection 결과를 보이는 것을 고려할때 이러한 false positive는 일종의 아웃라이어로 취급될 수 있으며 이들의 detection confidence는 suppressed 할 수 있습니다. (즉 여기서 하고싶은 말은 비디오 전체의 contextual infromation을 사용하여 에러를 smooth하자)

이 논문은 3 단계로 이루어져 있습니다. 1) 우리는 기존의 성공적인 still-image detection 프레임웤르르 확장하고 비디오에서의 일반적인 object detection 문제를 해겨라려고 하며 tubelet으로 부터 temporal, contextual information을 활용할 수 있도록 한다. 이는 T-CNN이라고 불리며 tubelet with CNN 의 의미이다. 2) 제안된 detection 프레임워크에서는 temporal inforamtion을 효율적으로 사용 가능합니다. 이는 인접한 프레임으로 detection 결과를 전파하고(locally) tracking 알고리즘으로 부터 생성된 tubelet을 통해 detection confidence를 조정 합니다(globally). 3) Contextual information은 비디오 클립안의 detection 결과들 중 low-confidece를 가지는 score들을 억제하는데 사용됩니다. 이 프레임워크는 2015 VID 에서 우승하였고 공식 코드도 사용 가능합니다.

### 2 RELATED WORK

<b>Object detection from still images.</b> object detection의 최신 방법들은 CNN을 기반으로 합니다. [3] 에서 Girshicsk 는 다단계 파이프 라인인 RCNN이라는 것을 제안하여 CNN을 region proposal을 예측하도록 학습시킵니다. 이는 detection problem을 몇몇 단계로 분해하는데 bounding box proposal, CNN pre-training, CNN fine-tuning, SVM training, bounding box regression으로 분해합니다. 이 프레임워크는 좋은 성능을 보였고 다른 논문에도 광범위 하게 사용됩니다. [1] 에서 Szegedy는 GoogleNet이라는 22레이어 'inception' 모듈을 제안하여 R-CNN의 CNN 부분을 대체하였고 2014 object detection task에서 우승을 하였습니다. [8] 에서 Ouyang은 deformation constrained pooling layer를 제안하고 box pre-training 전략을 제안하였고 50.3%의 accuracy를 얻었습니다. RCNN의 학습을 빨그ㅔ 하기 위해 Fast RCNN이 제안되었고 여기서는 이미지 패치를 더이상 고정된 사이즈로 리사이즈 하지 않았습니다. 대신에 마지막 conv layer에서 ROI pooling을 사용하여 해당하는 feature 레벨에서 crop을 합니다. Faster RCNN 에서는 RPN을 통해 region proposal을 수행하며 전체 프레임워크는 end-to-end 방식으로 훈련됩니다. 이모든 파이프라인은 still-image 에서의 obejct detection을 위한 것입니다. 이들을 비디오에 frame-by-frame 방법으로 적용하면 이들은 몇몇 positive sample들을 잘못 분류할 것이며 이는 비디오의 특정 프레임에서 object가 이상한 포즈를 취하기 때문입니다.

<b>Object localization in videos</b> object localization 과 co-localization과에 관한 연구도 있습니다. 이러한 task는 매우 비슷해 보이지만 VID task가 더 어렵다는 것에 주목합니다. 두 문제 사이의 몇가지 중요한 차이가 있습니다. 1) 목표 : localization 문제는 각각 비디오가 오직 하나의 known (weakly supervised setting) 혹은 unknown class (unsupervised setting) 이 있다고 가정하며 테스트 프레임에 대해 하나의 object에 대한 localizing이 필요합니다. VID 에서는 각 프레임에서 몇개의 object가 있는지 알 수 없어 VID task가 더 현실세계와 가깝습니다. 2) Metric : Localization metric은 (보통 CorLoc) 는 보통 (co)localization 을 평가하기 위해 사용됩니다. 반면에 VID 에서는 MAP가 사용됩니다. 이러한 차이를 고려할때 우리는 VID task가 더 어렵고 실세계과 가깝다고 생각합니다. object localization에 관련된 이전의 연구들은 VID에 바로 적용할 수가 없습니다.

<b>Image classification.</b> 지난 몇년간 image clasfication의 성능이 비약적으로 향상되었습니다. 이는 ImageNet과 같은 매우 큰 데이터 셋과 새로운 구조 때문입니다. object detection의 모델들은 보통 ImageNet classfication task에서 pre-train됩니다. BN layer는 [7] 에서 제안되었으며 미니 배치사이간의 statistical variation을 줄여주고 학습이 더 잘되게 합니다. Simonyan은 VGG-19 19-레이어 네트워크를 제안했으며 3x3 커널을 사용합니다. 이는 detection, action recognition, semantic segmentation등에서 광범위하게 사용됩니다.

<b>Visual tracking. </b> object tracking은 수십년간 연구되었습니다. 최근에는 CNN이 object tracking에 사용되었고 인상적인 tracking 결과를 달성했습니다. [30] 에서 Wangh은 object-specific traker를 제안하였고 이는 ImageNet pre-trained CNN에서 가장 영향력 있는 feature를 실시간으로 선택하여 tracking을 수행합니다. 이는 기존의 최신 tracker를 매우 큰 차이로 뛰어넘었습니다. [31] 에서 Nam 은 multi-domain CNN을 학습하였고 tracking object를 위한 일반적인 representation을 배웠습니다. 새로운 target을 추적할때 pre-trained CNN의 weight를 공유하고 맨 뒤의 binary classfication layer만 다른 새로운 네트워크가 추가되고 이는 online update 됩니다. tracking은 분명하게 VID와 다르네 이는 첫번째 프레임에 이미 object에 대한 localization이 되어져 있다고 가정하며 class label을 예측할 필요가 없습니다.

### 3 METHODS

이번 섹션에서는 우리는 VID task를 설명하고 우리의 전체 프레임워크에 대해 설명합니다. 그 후 주요한 요소들을 자세히 설명합니다. 섹션 3.3 에서의 우리의 still-image detector에 대한 설명을 하며 섹션 4에서는 어떻게 multi-context infromation을 사용하여 false positive detection을 억제하는지를 보이고 motion information을 사용해 false negative를 줄입니다. global tubelet rescoring이 섹션 3.5에 소개됩니다.

### 3.1 VID task setting

VID task는 DET 테스크와 매우 비슷합니다. VID task는 detect해야할 30개의 클래스로 구성되며 이는 DET task의 200개의 클래스의 부분집합입니다. 각 비디오 클립의 모든 프레임은 완전히 라벨링이 달려져 있습니다. 각각의 비디오 클립마다 알고리즘은 몇개의 예측을 수행해야 합니다. frame index fi, calss label ci, confidence score si, bounding box bi. VID 에 대한 evaluation은 DET task과 똑같습니다. 우리는 모든 클래스에 대한 map를 evaluation metric으로 사용합니다.

### 3.2 Framework overview

제안된 프레임워크는 피규어 2에 나와있습니다. 이는 4가지 주요 요소로 구성되어져 있습니다. 1) still-image detection, 2) mutli context suppression and motion-guided propagation 3) temporal tubelet re-scoring 4) model combination

### Still-image object detection.

우리의 still-image object detector는 DeepID-Net, Craft 를 사용하며 이는 DET, VID 트레이닝 셋 모두에서 학습되었습니다. DeepID-Net[8] 은 R-CNN의 확장이며 CRAFT는 Faster R-CNN의 확장입니다. 두 프레임워크 모두 object region proposal 단계와 region proposal scoring을 포함합니다. CRAFT와 가장 큰 차이점은 proposal generation, classfiation이 합쳐져서 하나의 end-to-end 네트워크라는 것입니다. still-image object detector는 개별 프레임에 대해 적용됩니다. 

<b>Multi-context suppression.</b> 이 과정은 먼저 still-image detection score를 내림차순으로 정렬합니다.(전체 비디오에 대해서) 높은 순위의 detection score를 가지는 class는 high-confidence class로 취급되며 다른 것들은 low-confidence class로 취급됩니다. low-confidence를 가지는 detection score들은 false positive를 줄이기 위해 억제됩니다.

<b>Motion-guided Propagation</b> still-image object detection에서 몇몇 object은 특정한 프레임에서 잘못 분류되지만 인접 프레임에서는 검출 될 수 있습니다. motion-guided propagation은 optical flow와 같은 motion information을 사용하여 detection result를 인접한 프레임으로 locally propagate하며 false negative를 줄일 수 잇습니다. (false negative : nagative로 예측했는데 틀렸다. 즉 label 이 True인데 false로 예측해버림.)

<b>Temporal tubelet re-scoring.</b> still-image detector의 high-confidence detection 으로 부터 시작하여 우리는 tracking 알고리즘을 적용해 tubelet을 얻습니다. tubelet은 detection score에 따라 positive/negative sample로 판별되어 집니다. positive score는 higer range로 매핑되고 negative socre는 lower range로 매핑되므로 score margin을 높일 수 있습니다. (뭔소리지. 0.5 이상은 positiv, 0.1~0.5는 negative 뭐 이런식으로 했다는 건가?)

<b>Model combination.</b> DeepID-Net, CRAFT의 proposal에 대해서 두 경우 모두 tubelet re-scoring, motion guided propagation 을 사용한 detection result를 모아서 min-max mapping 시켜 (0,1) 로 만듭니다. 그다음 두개 모델의 결과를 합친다음 0.5가 임계치인 nms를 적용하여 마지막 결과를 얻습니다.

### 3.3 Still-image object detectors

우리의 still-image detector는 DeepID-Net, CRAFT를 적용했습니다. 두개의 detector는 다른 region proposal, pre-train model, training 방법을 가집니다.

### 3.3.1 DeepID-Net

<b>Object region proposal</b> DeepID-Net에서의 region proposal은 selective search, Edge Boxes에 의해 얻어집니다. cascaded selection process를 통해 false positive box를 제거하며 pre-train Alexnet을 사용합니다. 모든 proposal box는 200개의 ImageNet detection class score로 라벨링 되어집니다. classfication score가 임계치 보다 낮은 box는 negative sample로 할당되고 쉽게 제거될 수있습니다. 이 과정을 통해 94% 의 proposal box를 제거하지만 recall이 90% 정도를 얻을 수 있습니다.

<b>Pre-trained models</b> ILSVRC 2015는 두가지 트랙을 제공합니다. 1) provided data track , ILSVRC 2015 데이터셋 만 사용가능합니다. classfication, localization, DET, VID, Place2 데이터만 사용 가능합니다. 2) for external data track, 추가적인 데이터를 사용 가능합니다.

provided data track을 위해 우리는 VGG, GoogleNet을 BN을 사용하여 학습시켰고 external data track을 위해서는 ImageNet 3000-class data를 사용했습니다. pre-train은 image-level annotation이 아닌 object level annotation을 사용했습니다.

### 3.4 Multi-context suppression (MCS) and motionguided propagation (MGP)

<b>Multi-context suppression (MCS).</b> still-image object detector를 비디오에 바로 적용하는 것의 하나의 한계는 비디오 클립에서의 contextual information을 무시하는 것입니다. 비디오의 각 프레임에서의 detection 결과는 강한 상관관계가 있으며 (프레임간의 강한 상관관계가 있다 한 프레임에 오브젝트가 있다면 인접한 프레임의 비슷한 위치에 오브젝트가 있을 것이다) 우리는 이러한 특징을 사용해 false positive detection(positive로 예측했는데 틀림. 즉 background를 foreground로 예측)을 억제 할 수 있습니다. VID 데이터셋에서의 비디오 snippet이 몇개인지 모르는 클래스를 포함하고 있지만, 몇몇 통계치를 발견했는데 비디오는 보통 매우 적은 클래스를 포함하고 같이 존재하는 클래스는 매우 상관관계가 높다는 것을 발견했습니다. 따라서 한 비디오 클립 에서의 이러한 detection 결과의 통계치들은 false positive를 구별하는데 도움이 될 것입니다.

예를들어 피규어 3에서 몇몇 프레임이 예로 나와있습니다. 몇몇 프레임에서 false positive detection이 큰 detection score를 가질 수 있습니다. 오직 context information만을 사용해서는 이러한 것들을 positive sample로 부터 구별할 수 없습니다. 그러나 다른 프레임의 detection 결과를 같이 고려하면 우리는 비디오 전체에서 주요한 high-confidence detection class를 결정할 수 있고 high-confidence가 아닌 false positive detection들을 아웃라이어로 볼 수 있습니다.

각 프레임마다 우리는 수백개의 region proposal을 가지고 있으며 각각은 30개의 클래스의 detection score를 갖습니다. 각 비디오 클립마다 우리는 모든 detection score를 내림차순으로 정렬합니다. (비디오 클립안의 모든 박스에 대해서) 임계값을 넘는 detection score를 가지는 class를 high-confidence class로 할당하고 나머지를 low-confidence class로 할당합니다. high-confidecne class의 detection score는 그대로 유지하고 low-confidence class의 socre는 상수를 뺌으로써 억제합니다. threshold와 빼줄 값은 validation set을 통해 찾게 됩니다.

<b>Motion-guided propagation (MGP). </b> multi-context suppression 과정은 false positive detection을 상당히 감소 시킬 수 있지만 false negative(negative로 예측했는데 틀림, 즉 foreground를 다른 object나 back ground로 예측)를 감소 시킬 수는 없습니다. false negative는 일반적으로 몇가지 이유 때문에 발생합니다. 1) object 를 충분히 덮을 수 있는 region proposal이 존재하지 않는다 2) bad pose, motion blur 때문에 detection score가 낮아진다.

이러한 false negative는 인접한 프레임에 더 많은 detection을 추가하여 극복 될 수 있습니다. 이는 인접한 프레임이 매우 높은 상관관계가 있기 때문입니다. detection 결과는 spatial location, detectionsocre 모두에 큰 상관을 가지고 있습니다. 예를 들어 object가 멈춰있거나 느리게 움직이는 경우 이 object는 인접한 프레임의 비슷한 위치에 나타나야만 합니다. 이러한 이유 때문에 detection 결과인 box와 score를 인접한 프레임에 전달하여 detection 결과를 향상시키고 false negative를 주이고자 합니다.

motion information에 따라 detection 결과인 bounding box를 전파하기 위해 우리는 motion-guided 라는 방법을 제안합니다. 각각 region proposal에 대해 bounding box안의 mean flow vector 를 계산하고 계산된 mean flow vector에 맞게 이동한 box coordinate, detection score을 인접 프레임에 전파합니다. 피규어 4에 이러한 예를 볼 수 있습니다.

### 3.5 Tubelet re-scoring

MGP(motion-guided propagation) 은 still-image detector으로 부터 short dense tubelet을 생성하게 됩니다. 이는 false negative 를 상당히 줄여주지만 short-term temporal constraint를 가지고 있으며 최종 detection result만을 consistency 하게 만들어 줍니다. long-term temporal consistency를 높이기 위해 우리는 더 긴 tubelet이 필요합니다. 따라서 우리는 tracking 알고리즘을 통해 긴 tubelet을 생성하고 tubelet 주위의 still-image object detection 결과를 보정해줍니다. 피규어 2에서 볼 수 있듯이 tubelet re-scoring은 3가지 부분으로 이루어져 있습니다. 1) 높은 신뢰도의 tracking 알고리즘 2) spatial maxpooling, 3) tubelet calssfication

<b>High-confidence tracking</b>  비디오 클립내의 각각의 객체에 대해 high-confidence를 가지는 detection proposal을 양방향으로 tracking 합니다. tracker 는 [30] 을 사용했으며 우리의 실험에서는 object pose, scale change에 대해 로버스트 한 성능을 보였습니다. tracking의 start bounding box는 anchor라고 불리며 이는 confidence가 가장 높은 detection proposal로 선택합니다. anchor 에서 시작하여 완전한 tubelet을 얻기위해 양방향으로 tracking을 수행합니다. tracking이 수행되는 동안 추적되고 있는 box는 drift 하거나 background, 다른 object로 향하거나 target object의 pose, scale의 변화에 적응하지 못할 수 있습니다. 따라서 tracking confidence가 임계값(우리의 경우는 0.1) 보다 낮아지면 tracking을 멈추게 되며 이는 false positive tubelet을 줄여줍니다. tubelet을 얻은 후에 new tracking을 시작하기 위해 남아있는 detection(still-image detection 이겠지) 에서 새로운 anchor를 선택합니다. 일반적으로 high-confidence detection은 spatially, temporally clustering  되는 경향이 있습니다. (detecion 결과들을 살펴보면... 몇몇 곳에 모여있는 경우가 많다 이를 temporally 하게 봐도.. 그런 경우가 많다는 뜻인듯) 그러므로 가장 confidence가 높은 detection 결과를 새로운 anchor로 선택하는 것은 같은 object에 대해 overlap이 큰 tubelet을 생성하는 경향이 있습니다.
 
 ###### 그래도 같은 object라는 것을 ... 보장하지 못하는데 이를 개선할 방법이 없을까?
 
 redundancy를 줄이고 최대한 많은 object를 커버하기 위해 우리는 NMS와 비슷한 작업을 수행합니다. 현재의 tubeelt과 IOU가 0.3 보다 낮은 detection은 새로운 anchor로 선택되지 않습니다. 이러한 tracking suppression 과정은 모든 남아있는 detection이 임계값보다 낮이 질 때 까지 반복적으로 수행합니다. 각각의 비디오 클립에 대해 이러한 tracking process는 VID 30 클래스에 대해 수행됩니다. (? 트래킹을 30개 클래스에 대해 한다는 소리인가..??? 이상한데 뭔가 같은 클래스가 비디오에 2개 있으면? 그냥.. 30개 클래스를 찾을 수 있도록 수행한다는 건가)
 
 <b>Spatial max-pooling. </b> 각각 클래스에 대한 tracking 이후에 우리는 high-confidence anchor에 대한 tubelet을 가지게 됩니다. 나이브한 접근방법은 tubelet 안의 모든 bounding box를 still-image object detector를 이용해 분류하는 것입니다. tracking을 통해 얻어진 tubelet과 still-image object detector의 결과는 다른 통계치를 가지기 때문에 tracking으로 부터 얻어진 bounding box에 still-image detector를 적용하면 detection score가 정확하지 않습니다. 또한 tracking의 실패로 인해 tracked location이 최적이 아닐 수 있습니다. 그러므로 tracked tublet에 적용되는 still-image detection은 신뢰할수가 없습니다.
 
 그러나 tubeelt과 공간적으로 가까운 detection을 유용한 정보를 제공 할 수 있습니다. spatial max-pooling 과정은 tubelet box proposal을 still-image object detection의 high-confidence box로 대체하는 것입니다.
 
각각의 tubelet box에 대해 먼저 still-image object detector로 부터 detection을 얻고 tubelet box와 IOU가 0.5 이상인 detection들을 선택합니다. 이중 maximum detection score를 가지는 detection들을 남기고 tracked bounding box를 대체합니다. 이러한 프로세스는 object detection 에서의 일반적인 NMS 프로세스를 사용하는 것입니다. tubelet box가 실제 positive box이지만 low detection score를 가진다면 이러한 프로세스는 detection score를 높여 줄 것입니다. IOU 임계값을 더 높게 사용하는 것은 tubelet box를 더 믿는 다는 것이며 가장 극단적인 케이스인 IOU =1 을 임계값으로 선택하면 still-image object detector를 고려하지 않고 tubelet box에만 완전히 의존한다는 뜻입니다.
 
<b>Tubelet classification and rescoring.</b>  High-confidence tracking과 max-pooling 프로세스(tubelet과 still-image detection 중 가장 많이 겹치는 아이들로 대체 하는 프로세스) 는 long sparse tubelet(왜 sparse?) 을 생성하고 temporal rescoring의 후보군이 됩니다. temporal rescoring의 주요한 아이디어는 tubelet을 positive/negative sample로 판별하고 score margin을 높이기 위해 detection score를 다른 범위로 매핑 하는 것입니다. (라벨은.. gt와의 IOU를 통해 할당해서 매겼을려나? 베이지안이면 unsupervised 를 했다는 것일려나...?)

tubelet classfication을 위한 인풋이 오직 detection score만을 포함하고 있기 때문에 tubelet calssfication의 feature는 매우 간단합니다. 우리는 tubelet detection score의 다양한 통계치인 mean, median, top-k(tubelet 의 score중 큰 kth score). 베이지안 classfier를 학습하여 통계치를 feature로 사용해 tubelet을 판별하도록 하였고 이 결과는 피규어 5에 나와있습니다. 우리의 실험에서는 top-k feature가 가장 좋은 성능을 보였습니다.
 
 classfication 후에 positive sample의 score는 (0.5 ~ 1) 로 매핑하였고 negative는 (0 ~ 0.5) 로 매핑하였습니다. tubelet detection score는 global 하게 변화되어 positive, negative tubelet의 margin 이 증가합니다. (베이지안 classfier를 로 positive, negative인지를 판별하고.. 억지로?? 강제로??? 범위 안으로 매핑 하는듯 ?)


### 4 EXPERIMENTS

### 4.1 Dataset

 1, 트레이닝 데이터에는 완전히 라벨리잉 달린 3862 개의 비디오 snippet이 존재하며 snippet 마다 6개에서 5492개의 프레임이 존재합니다. 2, validation 데이터는 라벨링이 달려 있는 555개의 비디오 snippset이 존재하며 11프레임에서 2898 프레임을 가집니다. 3, test 데이터는 937개의 snippset이 존재하며 ground truth는 공개되지 않았습니다. 공식 test 서버는 competetion을 위해 사용되며 사용에 제한이 있습니다. 우리는 object detection task에 보통 사용되는 validation 데이터에 대한 평가를 보고합니다. 끝으로 ILSVRC 2015에 참가한 상위 팀들의 결과를 보고합니다.
 
### 4.2 Parameter Settings

<b>Data configuration.</b> 우리는 DET, VID의 트레이닝 데이터를 섞음으로써 still-image object detector DeepID-Net 에 미치는 영향에 대해 설명합니다. 최상의 데이터 구성은 DeepIDNet, CRAFT에 모두 적용됩니다. VID 트레이닝 데이터가 DET 보다 더 많은 프레임을 가지기 때문에 DET의 샘플은 모두 사용하고 VID 에서 프레임을 샘플링하여 사용합니다. still-image object detector 를 학습하기 위해 우리는 VID 데이터를 사요하는 다양한 비율에 대해 실험을 하였습니다.

우리는 GoogleNet을 finetuning 하기 위해 몇몇의 data configuration을 조사했습니다. 테이블 1을 살펴보면 DET, VID의 비율을 2:1 로 사용하는 것이 still-image detector DeepID-Net 이 최상의 성능을 보였고 우리는 모든 모델을 이러한 2:1의 비율을 사용하여 fine-tune 하였습니다.

fine-tunning 외에도 SVM의 학습에 대한 data configuration을 조사했습니다. 이들의 성능은 테이블2에 나와있으며 DET, VID 모두를 사용하고 positive, negative sample 을 사용하는 것이 가장 좋은 성능을 보였습니다.

비디오 프레임 사이에 redundancy 가 있기 때문에 우리는 testing 타임에는 비디오 프레임을 factor 2로 샘프링하며 still-image detector를 나머지 프레임에 적용했습니다. (24 fps를 12fps 로 샘플링했다는 뜻인듯?) MCS, MGP, re-scoring 단계가 수행됩니다. upsampled frame의 detection box은 MGP에 의해 interpolation 됩니다. 우리는 validation set에서 프레임 샘플링에 따른 성능 차이를 관찰 하지 못했습니다.

결론적으로 우리는 VID 프레임을 DET 이미지의 갯수의 절반정도가 되도록 샘플링하고 이를 합쳐서 CNN 모델들을 fine-tune 합니다. DET, VID 이미지의 positive, negative sample을 모두 사용하여 학습을 시킵니다.

<b>Hyperparameter settings.</b> motion-guided propagation의 경우 테이블3 을 통해 다양한 progation window에 대한 성능을 보여줍니다. optical flow vector를 사용하지 않는 경우와 비교할때 같은 propagation window에서 MGP가 더 좋은 성능을 보여줍니다. 이는 MGP가 더 정확한 location의 detection 을 생성한다는 것을 의미합니다. 7 프레임 (앞뒤로 3프레임씩과 중간 1프레임) 은 경험적으로 window size 로 설정되었습니다. 

multi-context suppression 에서 비디오에서의 모든 bounding box에서의 상위 0.0003 클래스는 high-confidence class로 할당되고 low-confidence class의 경우 0.4를 빼게 됩니다. 이 하이퍼파라미터는 validation set에서 greedy search를 통해 찾아집니다.

<b>Network configurations.</b> DeepID-Net, CRAFT는 GoogleNet, VGG 모델을 기반으로 합니다. multi-scale, multi-region 기술은 score averaging 을 위해 여러개의 모델을 사용합니다. DET task에서 학습된 DeepID-Net의 경우 위에서 언급된 data configuration, multi-scale, multi-region, score average 기술들을 사용하여 70.7% 로 향상 시킬 수 있습니다.

### 4.3 Results

<b> Qualitative results.</b> 피규어 6에는 몇가지 qualitative result가 보여집니다. 우리가 제안된 프레임워크가 다음의 몇가지 특성을 따릅니다. 1) bounding box는 object에 매우 딱 맞습니다. 2) motion-guided propagation, tracking을 통해 false negative 를 줄이고 인접한 프레임간에 consistent detection 결과를 얻을 수 있습니다. 3) 이미지의 장면이 매우 복잡할 지라도 false positive 를 줄일 수 있습니다. 이는 multi-context information을 사용하여 negative class의 score를 억제할 수 있습니다.

<b>Quantitative results</b> provided-data, additional-data 에 대한 결과는 테이블 4에 나와있습니다. 이 결과는 validation set 으로 부터 얻어졌습니다. 우리는 still-image object detector가 65%~70% 정도의 map를 얻음을 알 수 있었습니다. MCS, MGP, tubelet re-scoring 을 사용하여 이를 6.7% 정도 향상시킵니다. 최종적으로 모델을 앙상블 함으로써 성능을 더 향상시킬 수 있습니다.

우리는 ILSVRC2015 에서 provided-data 트랙에서는 1위 를 하였고 30개중 28개의 클래스에서 가장 좋은 성능을 보였습니다 additional-data track 에서는 30개 클래스 중 11개에서 가장 좋은 성능을 보여 2위를 차지 했습니다. 테이블 5에는 validation set에 대해 제출된 AP 리스트들이 나와있으며 테이블 6에는 테스트 데이터에 대한 상위팀들과 우리팀의 결과가 있습니다.

### 5 CONCLUSION

이 논문에서는 우리는 temporal, contextual information을 활용하는 object detection 프레임워크를 제안합니다. 이 프레임워크는 VID 챌린지에서 우승하였고 월등한 성능을 보였습니다. 구성 요소에 대한 분석이 자세하게 조사되었고 코드는 공개되어있습니다. 

VID task는 새롭고 연구가 덜된 분야입니다. 우리가 제안한 프레임워크는 stil-image object detection 프레임워크를 기반으로 하였고 비디오를 위한 중요한 요소들이 설계되었씁니다. 우리는 이러한 요소이 end-to-end 시스템이 될 수 있으며 이것이 향후 연구 방향이라고 믿습니다.

### 정리

short tubelet 생성

1. Multi-context suppression (MCS) : 비디오 클립안에서 score 를 정렬하고 잘 나타나지 않는 클래스의 score를 억제한다. 
2. MGP(motion-guided propagation) : optical-flow를 통해 인접 프레임으로 detection 결과를 전달. shor-term tubelet을 생성가능

long tubelet 생성

3. tracking 알고리즘을 사용하여 long-term tubelet을 생성가능 still-image detection의 결과로 대체.
4. tubelet re-scoring 을 통해 long-term temporal consistency를 증가시킨다.

short tubelet + long tubelet을 결합하여 최종 detection 결과를 내게된다.
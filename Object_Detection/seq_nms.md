# SEQ-NMS FOR VIDEO OBJECT DETECTION

### Abstract

비디오에서의 object detection은 특정 프레임에서 쉽게 검출되는 object가 같은 클립안의 다른 프레임에서 검출되기 어려울 수 있어 어려윰을 겪습니다. 최근에는 still-image object detection에 많은 발전이 있었습니다. 이러한 방법들은 일반적으로 3가지 단계를 거칩니다. 1) obejct proposal generation 2) object classfication 3) post-prcessing 우리는 같은 클립내의 weak detection의 score를 향상시키기 위한 방법을 제안합니다. 이는 근처의 high-scoring object detection을 이용하고 일종의 post-processing 방법입니다. 우리가 제안한 방법이 still-image object detection 에서 우수한 결과를 보였고 VID task 에서 3위를 달성했습니다.

### 1 INTRODUCTION

single-image object detection은 몇년간 큰 성능향상이 있었습니다. 반면에 비디오에서의 object detection은 여전히 어려운 문제로 남아있습니다. 이는 같은 비디오 클립내에서 한 프레임에서 쉽게 검출 되는 객체가 다른 프레임에서는 검출 되기 어렵기 때문이며 이러한 현상이 발생하는 이유로 몇가지가 있습니다. 1) drastic scale change, 2) occlusion 3) motion blure. 이 논문에서는 single-image object detection 의 확장을 제안하여 이러한 어려움을 극복하려고 합니다. 이 논문의 주요한 공헌은 다음과 같습니다

1. Seq-NMS 라는 것을 제안하고 이는 비디오에서의 object detection pipeline을 향상시킵니다. 특히 우리는 post-processing 단계를 수정하여 high-scroing object detection을 이용하여 인접 프레임의 weak detecion score를 향상시킵니다.

2. Seq-NMS를 VID 데이터셋에서 평가했으며 최신의 single image-based method의 성능을 뛰어 넘었습니다. 우리의 방법이 어려운 scale, occlude, blurred 된 object를 detection 하는데 도움이 된다는 것을 보여줍니다. 특정 클래스에 대해 seq-nms가 성능 향상이 있는 것을 보여줍니다.

3. VID 2015 챌린지에서 3위를 하였습니다.

### 2 OUR APPROACH

우리는 'tracking by detecion' 에서 아이디어를 얻었고 이는 개별적인 detection을 sequence로 결합한 다음 이 sequence를 개별 bounding box를 re-scoring 하는데 이용합니다. 피규어 1에 예제가 나와있습니다. 한쌍의 인접한 프레임에 대해 첫번째 프레임의 상자와 두번째 프레임의 상자의 IOU가 임계값보다 높다면 이들은 link 됩니다. 우리는 전체 비디오 클립에 대해 이러한 연결을 찾습니다. 그 후 전체 클립에 대해 maximum score chain 을 찾으려고 합니다. 이는 sequence 안의 모든 bounding box들의 score의 합이 최대가 되도록 하는 연결을 찾음으로 수행됩니다. 이는 매우 효율적이고 간단한 dynamic programming 알고리즘을 통해 찾을 수 있습니다. 이 알고리즘은 연결된 bounding box의 집합을 출력하며 연결된 bounding box들은 다음에 연결할 box에서 제외 됩니다. 또한 프레임 안에서 suppression을 수행하는데 만약 chained boxes 중 하나의 bounding box를 b_t 라고 했을때 b_t와의 IOU가 임계값을 넘는 box들을 candidate box에서 제거됩니다. 이 알고리즘은 길이 1의 chain이 출력될 때 까지 반복됩니다.

chain이 추출된후 chain을 구성하는 bounding box를 re-scoring 할 수 있습니다. 가장 간단한 방법은 각 bounding box의 object class를 chain 전체에 걸쳐 평균을 내는 것입니다. 다른 방법은 chain에 걸친 maximum score로 할당합니다. 우리는 두가지 접근 방법 모두 실험을 하였습니다.

### 3 EXPERIMENTS

우리는 VID 데이터셋의 validation, test 셋을 통해 실험을 수행하였습니다. 우리의 시스템은 RPN, Faster R-CNN의 classfier를 사용합니다. RPN은 Zeiler & Fergus의 구조를 사용하였고 Simonyan & Zisserman의 VGG16 구조를 classfier로 사용합니다. post-processing 단계 동안 우리는 3가지 기술에 대해 실험합니다. 1) single image nms, 2) seq-nms(avg) 3) seq-nms(max), seq-nms을 통해 각각 평균, 최대값을 사용하여 sequence를 re-scoring 하였습니다. 테이블 1은 VID validation, test 셋에 대한 성능을 보여줍니다. 우리의 seq-nms(avg) 방법은 VID 2015 에서 3위를 차지 했으며. 피규어 2는 seq-nms(avg) 를 사용할때 어떤 클래스의 성능이 증가하는지를 보여줍니다. 피규어 3에서는 VID 데이터에서 seq-nms가 성능을 향상시킨 클립들을 보여줍니다.

#### 정리 

proposal box들이 주어져 있을때.. 1) IOU가 0.5 이상인 박스들을 모두 연결 하려고 한다 2) 연결되는 chine의 score의 합이 최대가 되도록 한다. 3) 이를 dynamic programming으로 해결. 4) 연결된 chain의 box를 중심으로 nms를 수행 5) chain의 scroe는 average or max 값으로 할당




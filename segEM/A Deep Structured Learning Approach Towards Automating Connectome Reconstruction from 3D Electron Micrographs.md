# A Deep Structured Learning Approach Towards Automating Connectome Reconstruction from 3D Electron Micrographs

### Abstract

우리는 3D EM 으로 부터의 뉴런의 세그멘테이션을 위한 딥러닝 방법을 제시한다. 이는 성능과 안정성 면에서 다른 기술보다 큰 향상을 보였다. 우리의 방법은 3D U-NET으로 구성되어져 있고 voxels에서의 affinity graph를 예측하도록 학습시켜졌다. 그 후 간단하고 효율적인 region agglomeration이 따라온다. 우리는 U-NET을 새로운 Malis loss를 통해 학습시켯고 이는 topological correctness를 장려한다. 우리의 Malis loss의 확장은 두가지로 구성되어져 있다. 첫째로 loss gradient를 계산하기 위한 O(nlogn) 방법을 보였고 이는 원래의 O(n^2) 보다 효율적이다. 두번째로 우리는 훈련단계에서 이상한 gradient를 피하기 위해 grdient의 계산을 두단계로 나눕니다. Our affinity predictions are accurate enough that simple learning-free
percentile-based agglomeration outperforms more involved methods
used earlier on inferior predictions. 우리는 3가지 다른 이미지 기술과 동물에 대한 EM(CREMI, FIB-25, SEGEM) 에 대해 결과를 보였고 이전의 결과에 비해 27%, 15%, 250%의 성능향상을 보였다. 우리의 발견은 하나의 3D 세그멘테이션 전략이 isotropic block-face EM 데이터와 anisotropic serial sectioned EM 데이터에 모두 적용가능 하다. 우리의 알고리즘의 실행시간은 O(n) 이며 megavoxel당 2.6초의 성과를 보였다. 

### Intorduction

신경 연결의 정확한 reconstruction은 생물학적 신경의 기능을 이해하는데 매우 중요하다. 3D EM은 시각화를 위한 충분한 해상도를 얻고 모호함없이 dense neural을 얻기 위한 유일한 방법이다. 그러나 이러한 해상도에서는 적당히 산경 신경회로라도 이미지 크기가 너무 크기 대문에 수동 으로 reconstruction 하기 힘들다. 그러므로 인간의 분석을 돕기 위해서는 자동화된 방법이 필요하다.

우리는 3D EM으로 뉴런을 재구성하기 위한 딥러닝 기반의 이미지 세그멘테이션 방법을 제안한다. 이는 확장성과 정확도 면에서 다른 방법들을 뛰어넘었다. 피규어1을 참고하면 우리의 방법에 대한 구성요소를 볼 수 있다. (1) 3D U-NET 을 사용하여 affinity graph를 예측한다. (2) topological error를 줄이기 위한 Malis loss를 사용하여 학습시킨다. (3) 효율적인 O(N)의 agglomeration 방법을 사용한다. 이는 quantile of predicted affinites에 기반한다.

voxel affinites를 예측하기 위해 3D U-NET을 선택한 동기는 2가지 이다. 첫째로 U-NET은 2D와 생물학적 3D 이미지 데이터에서 매우 높은 성능을 보이고 있다. 이들의 특징중 하나는 Multi-scale 구조인데 이는 계산적, statistical 효율이 있다. 두번째로 U-net은 효율적으로 큰 region을 예측하다. 우리는 region에 대한 affinity prediction이 필요하기 때문에 Malis loss와 결합하여 학습을 시켰다.

우리는 Malis loss의 확장을 사용하여 3D U-NET이 affinites를 예측하도록 학습시켰다. 원래의 <span style="color:RED">Malis loss와 같이 우리는 hypothetical thresholding 위에서의 topological error를 최소화하고 예측된 affinites에서 connected component 분석을 한다. </span> (뭘한다는 건지 모르겠다 Malis 페이퍼를 봐야 알듯) 우리는 원래의 수식을 확장하여 모든 예측된 affinities에 관한 gradient를 계산하여 dense한 예측을 하고 (원래는 sparse sampling 해서 backprop 했나봄?) 빠른 gradient 계산을 하도록 한다. 게다가 우리는 Malis loss를 두가지 흐름으로 계산한다. positive pass에서는 우리는 예측된 모든 affinities 사이의 공간이 0이되고 모든 gt 의 바깥의 공간이 0이 되도록 한다. negative pass에서는 affinities 안의 공간이 1이 되도록 하고 이는 이상한 gradient를 피한다. <span style="color:RED">(라벨의 할당을 어떻게..했다는거 같은데.. 잘 이해는 안간다.. 대충 이해해보면.. affinities랑 랑 background의 교집합을 0 으로 할당하는 것이 positive pass negative pass에서는 affinites 안의 region을 1로 할당하고 backprop 했다는 것 같음)</span>

후속적인 thresholding을 가정하고 학습이 진행되었지만 우리는 fragment(supervoxel 세그멘테이션의 가장 작은 조각) 을 반복적인 agglomeration을 하는 것이 더 robust 하다는 것을 발견했다. (네트워크의 출력은 0~1의 값을 가지는 voxel일 것이다. 이를 단순히 thresholding 할 수도 있지만.. watershed를 하니까 더 효율적이더라). 이 후 우리는 watershed 알고리즘을 사용하여 예측된 affinites에서 fragment를 추루한다. 이 fragment는 그다음 region adjancency graph로 표현된다. RAG는 edge로써 예측된 affinites와 adjacent fragment사이의 score를 반영한다. (??뭔소린지 모르겠음) 작은 score를 가지는 edge는 high score를 가지는 edge와 합쳐집니다. 우리는 edge score를 k개의 bin으로 나누고 이를 sorting을 위한 priority queue에 들어갈 수 있도록 합니다. (watershed를 쓰기 위한 priority queue에 넣었다는 소리인듯) 이러한 방법으로 agglomeration은 최악의 경우에도 선형시간안에 수행가능하다.

affinites, watershed, agglomeration 은 볼륨의 크기 n과 O(n)의 관계를 가진다. 볼륨이 보통 tarabyte 까지도 할수 있기 때문에 현재의 최신의 방법론들은 다음과 같은 비슷한 방법을 취한다. 

1. 딥네트워크를 이용하여 voxel-wise 예측을 한다. 
2. 네트워크의 예측을 이용하여 greedy(CELIS, GALA) 혹은 global optimal objective(MULTICUT) 등을 사용하여 fragment를 얻는다. 

현재의 노력들은 fragment의 병합에 초점을 두고 있다. CELIS, GALA 와같은 것들은 classfier를 학습시켜 계층적 agglomeration을 위한 score를 예측합니다. 이는 inference 동안의 computation 과 복잡도를 증가시킵니다. 비슷하게 MULTICUT 의 변종은 classfier를 훈련시켜 fragment의 연결성을 예측합니다. 그 다음에 계산적으로 매우 비싼 combinatorial optimization을 품으로써 clustering 합니다. 우리가 제안한 fragment agglomeration 방법은 이러한 계산적 복잡도를 감소시키고 agglomeration에 별도의 학습이 필요하지 않습니다.

우리는 3가지의 다른 3D 현미경으로 찍힌 이미지를 통해 우리의 방법론의 효율을 보여줍니다. 우리의 방법은 각 데이터셋에서 최고의 성능을 보이는 방법을 뛰어넘었습니다. 우리는 traning, agglomeration의 소스코드를 공개하였고 우리의 Cremi 결과를 재현하돌고 했습니다.

## Method
### 2.1 Deep multi-scale convolutional network for predicting 3D voxel affinities

우리는 3D U-NET 구조를 통해 3D 볼륨에서의 voxel affinities를 예측합니다. 모든 데이터셋에 동일한 구조를 사용합니다. 특히 우리의 3D U-NET은 4가지의 해상도 레벨로 구성되어져 있습니다. 우리는 적어도 하나의 convolution pass와 두개의 convolution으로 구성되어져 있고 Relu를 사용합니다. 이들 레이어 사이에서 우리는 max pooling을 사용하고 이들 커널은 dataset의 해상도에 따라 달라집니다.(그림에서 노란색 화살표), 같은 크기로 transposed conv를 수행합니다.(갈색 화살표). 업샘플링의 결과는 같은 레벨의 downsampling pass와 concat 되어집니다. 개별 패스의 세부사항은 피규어6에 나와있습니다. 각 데이터셋에서의 구조에 대한 자세한 사항은 그림 5에서 볼 수 있습니다.

<span style="color:RED">우리는 foreground/background 에대한 예측대신 edge위의 affinites를 예측하기로 했습니다. 이는 우리의 방법이 low-spatial resolution을 다룰 수 있도록 합니다.</span> 그림 1b에서 볼 수 있듯이 낮은 z resolution(보통 serial section EM에서 나타난다?) 에서 foreground/background 라벨링이 불가능합니다. 반면에 affinites는 효과적으로 우리의 모델의 표현력을 높이고 올바른 세그멘테이션을 얻게 합니다. 또한 affinites는 임의의 neighborhoods로 일반화 될 수 있으므로 longer range 연결성을 가능하게 합니다.

그림1.)
여기서 표시된 예제는 voxel은 foreground/background로 올바르게 라벨될 수 없습니다. 만약 A가 전경으로 라벨되면 이는 다음 섹션의 영역과 반드시 합쳐집니다. A가 배경으로 라벨되면 이것은 split을 유발할 것입니다. affinites에 대해 라벨링 하는 것은 B,C를 A에서 분리하는 것이 가능하며 region 내에서의 connectivity를 유지할 수 있습니다. 예측된 affinites로 부터 우리는 oversegmentation을 얻을 수 있고 그다음 percentile-based agglomeration 알고리즘을 사용하며 마지막 segmentation을 얻을 수 있습니다. (watershed를 쓰기 때문에 그런것인가..?? watershed를 안쓴다면 상관이 없나?)

### 2.2 Training using constrained MALIS

우리는 Malis loss의 확장을 사용하여 네트워크를 훈련합니다. constrained Malis 라는 loss는 얻어진 세그멘테이션에서의 topological error와 connected component 분석을 최소화 하도록 합니다. 
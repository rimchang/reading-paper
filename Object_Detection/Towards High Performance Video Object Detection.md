# Towards High Performance Video Object Detect

### Abstract

최근 몇년 동안 still-image object detection 에서의 큰 발전이 있었습니다. 하지만 video object detection이 더 어렵고 실제 시나리오에 알맞지만 적은 주목을 받고 있습니다. [flow-guided feature aggregation(36), deep feature flow(37)]와 같은 최근의 연구는 multi-frame 에서의 feature, cross-frame motion을 end-to-end로 학습하는 통합된 접근방법을 제안합니다. 우리의 접근은 세가지 새로운 기술을 통해 이전의 연구를 확장하고 speed-accuracy tradeoff를 탐색하고 더 좋은 video object detection 성능을 개선하도록 나아갑니다.

### 1. Introduction

최근 몇년동안 still-image detection의 성능이 매우 발전했습니다. 하지만 이를 video 도메인으로 바로 적용하는 것은 문제가 있습니다. 첫번째로 모든 비디오 프레임에 딥 네트워크를 적용하면 계산량이 엄청나게 증가하게 됩니다. 둘째로 비디오에서는 still-image에서 볼 수 없는 motion-blur, video defocus, rare pose 등이 존재해 성능이 저하 됩니다. 

지금까지 video object detection에 관한 연구는 거의 없었습니다. [flow-guided feature aggregation(36), deep feature flow(37)]과 같은 최근의 연구는 multi-frame end-to-end learning 을 제안하여 위에서 언급한 문제를 효과적으로 다룹니다. 특히 [37]에서는 연속적인 프레임간의 data redundancy를 활용해 대부분의 프레임에서의 feature를 추출하기 위한 연산량을 줄이고 속도를 향상시켰습니다. Temporal feature aggregation은 accuracy와 feature의 질을 높이기 위해 수행됩니다. 이 논문은 VID 2017 챌린지에서 우승을 하였습니다.

두 작업은 서로 다른 측면에 집중하며 그들의 단점을 제시합니다. [37] 에서는 Sparse feature propagation 이 사용되어 대부분의 프레임에서의 feature를 추출하기 위한 연산을 절약했습니다. 프레임의 feature들은 sparse key frame 으로 부터 전달받아 연산량이 매우 적습니다. 그러나 전달된 feature는 근사된 feature이기 때문에 error가 생길 수 있고 accuracy를 떨어트릴 수 있습니다. [36] 에서는 Multi-frame dense feature aggregation 을 수행하여 feature의 질을 높이고 detection accuracy를 높였습니다. 그럼에도 이는 반복적인 motion estimation, feature propagation, aggregation 때문에 훨씬 느린 속도를 보입니다.

이 두 연구는 본질적으로 상호 보완적입니다. 이들은 동일한 원칙를 가지고 있습니다. motion estimation module이 네트워크 안에 내장되어져 있고 여러 프레임에 걸쳐 모든 모듈은 end-to-end로 학습됩니다.

이러한 발전과 원칙을 바탕으로 우리의 연구는 이들을 통합하여 더 빠르고 정확하며 유연한 방식을 제시합니다. 특히 세가지 새로운 기술이 제안됩니다. 1) feature aggregation의 질을 높이기 위해 sparsely recursive feature aggregation이 사용됩니다. 이는 sparse key frame 에만 연산이 수행되어져 연산량을 줄입니다. 이 기술은 [37, 36] 의 장점을 결합하고 두 모델보다 나은 성능을 보입니다.

두번째로 spatially-adaptive parital feature updating 이라는 기술을 도입합니다. 이는 non-key frame으로 전파된 warped feature의 질이 안좋은 경우 feature를 다시 계산합니다. feature의 질은 새로운 형식을 통해 학습되며 end-to-end 학습이 가능합니다. 이 기술은 성능을 더 향상시킵니다.

마지막으로 temporally-adaptive key frame scheduling 을 사용하여 고정된 key-frame scheduling을 대체합니다. 이는 예측된 feature quality에 따라 사용해야 할 key-frame을 예측합니다. 이는 더 효과적인 key-frame을 사용하게 만듭니다.

제안된 기술들은 앞선 [37, 36] 의 연구들과 통합되어 집니다. 다양한 실험을 통해 이 세가지 기술이 더 나은 성능을 야기하는 것을 보였습니다. 예를들어 우리는 77.8% map를 15.22 fps 의 속도로 얻었습니다. 

### 2. From Image to Video Object Detection

static image 에서의 object detection은 최근 CNN의 사용을 통해 큰 발전을 얻었습니다. 최신의 detector는 유사한 방법론과 두 단계로 구성된 네트워크 구조를 사용합니다.

첫번째 단계는 conv feature map F 를 추출하는 것입니다. 전체 입력 이미지 I에 대해 fully convolutional backbone 네트워크를 사용해 feature map을 추출합니다. backbone은 일반적으로 ImageNet classfication task에 의해 pre-train 되고 이후 fine-tune 됩니다. 이 논문에서는 이러한 bakcbone을 feature network라고 부르며 N_feat(I) = F 로 나타냅니다. feature network는 매우 깊고 느려서 모든 프레임에 대해 feature를 추출하는 것은 불가능합니다.

두번째 단계는 feature map F에 대해 detection result y를 생성합니다. 이는 region classfication, bounding box regression 을 통해 수행되며 sparse object proposal 이나 dense sliding window 방식을 사용합니다. 이 논문에서는 이를 detection network 라고 부르며 N_det(F) = y 로 나타냅니다. 이는 랜덤하게 초기화 되며 N_feat 와 jointly 학습이 가능합니다. 이는 일반적으로 shallow 하며 빠릅니다.

### 2.1. Revisiting Two Baseline Methods on Video

<b>Sparse Feature Propagation </b> 이 논문은 video object detection을 위한 key frame에 대한 개념을 도입합니다. 이의 동기는 일반적으로 인접한 프레임은 유사한 픽셀값을 가지고 있어 유사한 feature를 얻을 것입니다. 그러므로 모든 프레임에 대해 feature를 계산할 필요가 없습니다.

inference 동안 feature network N_feat는 10번째 마다의 프레임인 sparse key frame에만 적용됩니다. non-key frame의 feature map은 key frame k 에서 전파되어 지며 bilinear interpolation, feature warping을 통해 수행됩니다. 프레임 간의 pixel-wise motion은 2차원의 motion field M_ik 에 저장되게 됩니다. key frame으로 부터 전달 받는 i 번째 frame의 feature는 다음과 같이 정의됩니다.

<img src="https://latex.codecogs.com/gif.latex?F_%7Bk%5Crightarrow%20i%7D%20%3D%20W%28F_k%2C%20M_%7Bi%5Crightarrow%20k%7D%29%20%5C%2C%5C%2C%5C%2C%5C%2C%20%281%29" />

여기서 W는 feature warping function을 나타냅니다. detection network N_det 는 warped feature F_Ki 를 사용하게 되며 이는 real feature F_i의 근사 입니다.

motion field는 경량화된 flow network를 통해 추정됩니다. N_flow(I_k, I_i) = M_ik 이는 두개의 프레임 I_k, I_i 를 인풋으로 받아 motion field를 출력합니다. 모든 모듈은 end-to-end로 학습이 가능하며 detection 성능을 크게 향상시키고 feature approxiamtion으로 인한 에러를 보완해줍니다. single frame detector와 비교할때 N_flow가 N_feat 보다 훨씬 연산량이 낮습니다. [37] 에서 사용된 방법은 낮은 성능의 저하에도 10배 정도 빠른 성능을 보여줬습니다.

<b>Dense Feature Aggregation</b> 이 논문에서는 처음으로 temporal feature aggregation 이라는 개념을 소개합니다. 이의 동기는 특정 프레임에서 motion blur, occlusion 등으로 인해 appreance가 손상될 수 있고 이는 feature의 질을 낮춥니다. 이 논문에서는 인접한 프레임의 feature를 aggragation 함으로써 개선합니다.

inference 동안 feature 네트워크인 N_feat은 모든 프레임에 대해 evaluation 됩ㄴ디ㅏ. 임의의 프레임 i 에 대해 r 이라는 temporal window 만큼의 인접 프레임을 i 프레임에 대해 warping 시킵니다. [sparse feature propagation(37)] 과 다르게 propagation은 모든 프레임에 대해 평가됩니다. 다른 말로 하면 모든 프레임이 key frame이 됩니다.

i번째 프레임의 aggregated feature map 은 weighted avreage를 통해 구해집니다. 

weight 는 feature간의 유사도를 구하기 위한 embedding feature 를 통해 adaptive 하게 계산됩니다. feature aggregation, feature warping이 position-wise 하게 구해지며 location p 에 대한 인접 프레임에 대한 weight의 합이 1이 되도록 합니다.

[37] 과 비슷하게 모든 모듈은 flow network, aggregation weight를 가지고 있으며 jointly train이 가능합ㄴ디ㅏ. single frame detector와 비교해 보자면 feature aggregation이 detection 성능을 3% 가량 높여주며 특히 빠르게 움직이는 object의 경우 더 높은 성능 개선이 보여집니다. 그러나 모든 프레임에 대한 flow estimation, feature aggreagation 때문에 single detector보다 3배정도 느린 속도를 가집니다.

### 3. High Performance Video Object Detection

위의 두가지 방법의 차이점은 분명합니다. [sparse feature propagation(37)]  는 sparse feature propagation을 통해 feature를 근사해 연산량을 줄이지만 성능을 낮추게 됩니다. [Dense Feature Aggregation(36)] 은 adaptive aggregation을 통해 feature의 질을 높이지만 연산량을 증가시키게 됩니다. 이들은 본질적으로 상호 보완적입니다.

이들은 두가지 원칙을 기반으로 하고 있습니다. 1) 프레임 간의 propagation을 위해 motion estimation이 필요합니다. 2) end-to-end 로 학습이 가능하며 모든 구성된 모듈이 detecion 성능에 필수적입니다. [37, 36] 에서의 절제실험으로 확인 할 수 있습니다.

이러한 기본원리를 기반으로 이 논문은 video object detection의 높은 성능을 위한 통합된 프레임워크를 제안하며 섹션 3.4에 요약되어져 있습니다. 세가지의 새로운 기술들을 제안합니다. 1) [37, 36] 에서의 방법을 통합하고 상호 보완적인 특징들을 이용합니다. 이는 정확하고 속도가 빠릅니다. 2) temporal 도메인에서의 adaptive feature 연산에 대한 개념을 spatial 도메인으로 확장시킵니다. spatially adaptive feature computation이 더 효과적인 것을 보입니다. 3) feature computation의 성능을 향상시키기 위해 adaptive key frame scheduling을 제안합니다.

이 기술들은 매우 간단하며 직관적입니다. 이들은 본질적으로 이전의 연구를 확장한 것이며 섹션 5의 실험을 통해 performanc envelope를 증가시킵니다. 2개의 베이스라인과 새로운 3개의 기술은 피규어 1에 나와있습니다.


### 3.1. Sparsely Recursive Feature Aggregation

[Dense Feature Aggregation(36)] 은 detection 성능이 크게 증가했지만 매우 느립니다. 이 모델은 feature network N_feat 를 모든 프레임에 대해 적용합니다. 하지만 인접한 프레임의 유사한 apperance 때문에 이는 불필요한 일입니다. 또한 feature aggregation은 여러개의 feature map에 대해 수행되며 대응하는 프레임에 대한 flow를 추정해야할 필요가 있습니다. 이는 detector의 속도를 크게 낮추게 됩니다.

이 논문에서는 Sparsely Recursive Feature Aggregation 이라는 기술을 제안합니다. 이는 sparse key frame에만 recursive feature aggregation, feature network N_feat을 적용합니다. 연속적인 key frame k, k\`가 주어지면 k\` 프레임의 aggregated feature 는 다음과 같이 계산됩니다.

<img src="https://latex.codecogs.com/gif.latex?%5Cbar%7BF%7D_%7B%7Bk%7D%27%7D%20%3D%20W_%7Bk%5Crightarrow%20%7Bk%7D%27%7D%20%5Codot%20%5Cbar%7BF%7D_%7Bk%5Crightarrow%20%7Bk%7D%27%7D%20&plus;%20W_%7B%7Bk%7D%27%5Crightarrow%20%7Bk%7D%27%7D%20%5Codot%20%5Cbar%7BF%7D_%7B%7Bk%7D%27%7D%20%5C%2C%5C%2C%5C%2C%5C%2C%20%284%29" />

여기서 bar_F 는 warped feature를 나타내며 동그라미는 element-wise multiplication을 나타냅니다. weight W_kk'(p) + W_k'k'(p) =1 이 되도록 모든 location p 에서 normalize 됩니다.  

이전 수식2의 recursive 한 버젼으로써 sparse key frame에 대해서만 aggregation을 하게 됩니다. 본질적으로 aggregated key frame feature는 key frame의 모든 히스토리를 가진 풍부한 feature가 되게 됩니다. 이 feature는 다음 key frame k'에 전파되어 k'의 원래 feature와 aggregation 되어지는데 사용됩니다.

### 3.2. Spatially-adaptive Partial Feature Updating

Sparse Feature propagation을 통해 real feature F_i 를 근사하여 매우 빠르게 feature를 구할 수 있지만 propagated feature map F_ki 는 프레임간의 pixel값의 변화 때문에 오류가 생길 수 있습니다.

non-key frame에 대해서 우리는 feature propagation의 아이디어를 사용해 효율적으로 구할 수 있도록 합니다. 수식1은 propagation quality를 나타냅니다. propagted feature F_ki quality를 양적으로 나타내기 위해 feature temporal consistency Q_ki 를 도입합니다. 우리는 flow network N_flow에 새로운 branch를 추가하여 Q_ki를 예측합니다. motion field M_ik, feature temporal consistency Q_Ki는 다음과 같이 구해집니다.

<img src="https://latex.codecogs.com/gif.latex?%5Cbegin%7BBmatrix%7D%20M_%7Bi%5Crightarrow%20k%7D%2C%20Q_%7Bk%5Crightarrow%20i%7D%20%5Cend%7BBmatrix%7D%20%3D%20N_%7Bflow%7D%28I_k%2CI_i%29%20%5C%2C%5C%2C%5C%2C%5C%2C%20%285%29" />

Q_ki 는 k(key) frame에서 i frame로의 temporal consistency를 나타내며 , M_ik 는 i frame에서 k(key) frame 으로의 motion field를 추정한다.

만약 Q_ki(p) < t , 즉 임계값 보다 temporal consistency가 낮다면 propagted feature F_ki(p) 가 inconsistent 하다는 것을 나타냅니다. 즉 F_ki(p) 가 좋지 않은 근사값이라는 의미이며 이는 real feature를 통해 보정해줘야 함을 의미합니다.

우리는 non-key frame을 위한 partial feature updating을 제안하며 feature frame i 는 다음과 같이 업데이트 됩니다.


<img src="https://latex.codecogs.com/gif.latex?%5Chat%7BF_i%7D%20%3D%20U_%7Bk%5Crightarrow%20i%7D%20%5Codot%20N_feat%28I_i%29%20&plus;%20%281-U_%7Bk%5Crightarrow%20i%7D%29%20%5Codot%20F_%7Bk%5Crightarrow%20i%7D%20%5C%2C%5C%2C%5C%2C%5C%2C%20%286%29" />

왜 partial 하게 업데이트를 할까?? key frame의 feature는 이미 aggregation 되어 있는 feature이기 때문에... 최대한 aggregated feature를 보존하고 싶어서인듯

U는 updating을 위한 mask 로써 Q_Ki(p) 각 임계값 보다 낮으면 U_ki(p) = 1 이며 아니면 0이 됩니다. 우리의 구현에서는 더 경제적인 방법을 사용합니다. layer n 에서의 recompute feature F_i(n) 을 계산하기 위해 이전 레이어에서 partially update feature를 계산하고 이를 N_feat(n) 의 인풋으로 넣어 계산합니다. 즉 partial feature updating은 layer-by-layer로 계산 될 수 있습니다. 다른 레이어 간의 feature map resolution 때문에 우리는 nearest neighbor interpolation을 사용하여 updating mask를 계산합니다.

[3]에 따라 어쩌구..저쩌구를 하였고 하여튼 이는 미분가능합니다. non-key frame에 대한 feature quality를 향상시키기 위해 위에서와 비슷한 feature aggregation을 사용합니다.


<img src="https://latex.codecogs.com/gif.latex?%5Cbar%7BF%7D_%7B%7Bi%7D%7D%20%3D%20W_%7Bk%5Crightarrow%20%7Bi%7D%7D%20%5Codot%20%5Cbar%7BF%7D_%7Bk%5Crightarrow%20%7Bk%7D%27%7D%20&plus;%20W_%7B%7Bi%7D%5Crightarrow%20%7Bi%7D%7D%20%5Codot%20%5Chat%7BF%7D_%7B%7Bi%7D%7D%20%5C%2C%5C%2C%5C%2C%5C%2C%20%284%29" />

잘살펴보면.. F_i(즉 non-key frame에 대한 aggregated feature) 는 F_ki(ket frame feature에서 motion field에 따라 warped feature) 와 hat F_i ()partially updated feature) 와의 가중치를 곱한 것을 더해서 구해진다!!

여기서도 weight W_ki(p) + W_ii(p) = 1 로 normalization 됩니다.

### 3.3. Temporally-adaptive Key Frame Scheduling

sparse key frame 에만 feature network N_feat을 적용하는 것은 빠른 속도를 위해 중요합니다. [37] 에서 사용된 naive key frame scheduling policy는 key frame을 l frame 마다 선택하는 pre-fixed rate policy 입니다. temporal domain의 변화하는 dynamic에 adaptive 한 policy는 더 나은 key frame scheduling 이 될 것입니다. 이는 consistency indicator Q_ki를 기반으로 설계될 수 있습니다.

key = is_key(Q_ki)  

간단한 휴리스틱을 이용하여 is_key function을 다음과 같이 설정했습니다.

<img src="https://latex.codecogs.com/gif.latex?is%5C_key%28Q_%7Bk%20%5Crightarrow%20i%7D%29%20%3D%20%5B%5Cfrac%7B1%7D%7BN_P%7D%5Csum_p%201%28Q_%7Bk%20%5Crightarrow%20i%7D%28p%29%20%5Cleq%20%5Ctau%5D%29%5D%20%3E%20%5Cgamma%20%5C%2C%5C%2C%5C%2C%5C%2C%20%289%29" />

모든 position p에 대해서 temporal inconsistency 한 것들이 gamma 보다 높으면 이를 key frame으로 삼는다.

1() 은 indicator function이며 N_p 는 all location p의 갯수입니다. Q_ki(p) 가 임계값 tau 보다 낮은 것은 apperance가 변화되었거나 나쁜 feature propagation quality를 야기하는 large motion 이 있다는 것을 나타내며 recompute 해야할 feature 가 됩니다. 만약 어떤 프레임에서 recompute를 해야하는 area가 gamma보다 크다면 그 프레임을 key로 사용합니다. 피규어 2는 시간에 따라 Q_ki 값을 보여줍니다. 그림에서 세개의 주황색 점들이 우리의 is_key function을 통해 선택된 key frame이 됩니다. 그림을 보면 key frame으로 선택된 프레임은 appearance는 급격한 변화가 있는 것을 볼 수 있습니다. 파란색 점은 non-key frame들을 보여줍니다. 이들의 appearance는 약간의 변화만 있는 것을 볼 수 있습니다.

key frame scheduling의 potential, upper bound를 찾기 위해 우리는 ground-truth information을 사용하는 oracle scheduling policy를 설계했습니다. 실험은 key frame scheduling policy를 제외한 위에서 제안한 방법으로 진행되었습니다. 임의의 i 번째 프레임이 주어지면 이 i번째 프레임을 non-key frame, key frame으로 삼고 2개의 detection result를 계산한뒤 map score를 ground truth를 사용하여 계산합니다. key frame으로 삼을때 map가 높아진다면 이 i 번째 프레임을 key 로 선택합니다.

이 oracle scheduling은 훨씬 좋은 성능을 보였습니다. 22.8 fps 속도로 80.9% map을 얻을 수 있었습니다. 이는 key frame scheduling이 중요하다는 것을 나타내며 미래의 연구 방향을 제시합니다.

### 3.4. A Unified Viewpoint

feature map을 효율적으로 계산하기 위해 Spatially-adaptive partial feature updating(bad feature를 location 별로 partial updating ) 이 사용됩니다. 수식 6은 non-key frame 에만 정의되어져 있지만 이는 모든 프레임에 대해 적용될 수 있습니다. frame i와 이전의 key frame k가 주어졌을때 수식6은 다음과 같이 요약 될 수 있습니다.

<img src="https://latex.codecogs.com/gif.latex?%5Chat%20F_i%20%3D%20PartialUpdate%28I_i%2C%20F_k%2C%20M_%7Bi%5Crightarrow%20k%7D%2C%20Q_%7Bk%20%5Crightarrow%20i%7D%29%20%5C%2C%5C%2C%5C%2C%5C%2C%20%2810%29" />

key frame 에서 Q_ki = -무한대 , 이면 propagated feature F_fi 가 항상 안좋은 근사가 된다는 것을 의미하며 우리는 feature를 다시 계산해야만 합니다. non-key frame 에서 Q_ki = + 무한대 라면 F_ki가 real feature의 좋은 근사라는 것을 의미하며 우리는 key frame에서 propagated feature를 바로 사용하게 됩니다. (뭔소리지 하여튼 Q_Ki에 따라서.. 다시 계산하거나 바로 feature를 사용한다)

partially updated feature map의 질을 향상시키기 위해서 feature aggregation이 사용됩니다. 수식 4는 key frame 에만 sparsely recursive feature aggreagation이 정의되어있고 수식 7은 partially updated non-key frame에 대해서만 정의되어져 있습니다. 수식 4 는 수식 7 특별한 형태로 볼 수 있습니다. i=k', hat F_i = F_k' 인 경우의 퇴화된 버젼으로 볼 수 있습니다. 그래서 feature aggregation은 수식 7 처럼 수행됩니다. 


<img src="https://latex.codecogs.com/gif.latex?%5Cbar%20F_i%20%3D%20G%28%5Cbar%20F_k%2C%20%5Chat%20F_i%2C%20M%7Bi%5Crightarrow%20k%7D%29%20%5C%2C%5C%2C%5C%2C%5C%2C%20%2811%29" />

feature computation을 더 효율적으로 향상시키기 위해 temporally-adaptive key frame scheduling이 사용됩니다.
 
# Deep Sparse Rectifier Neural Networks

## Abstract

tanh보다 sigmoid보다 더 생물학적으로 그럴듯 하지만 tanh가 multi-layer 뉴럴네트워크를 학습하는데 더 좋다는 것이 밝혀져 있습니다.. 이 논문에서는 relu가 생물학적 뉴런을 모델링하는데 더 그럴듯한 함수라는 것을 보이고 hard non-linearity, non-differentiable 함에도 sigmoid, tanh보다 더 좋거나 비슷한 성능을 보이는 것을 실험적으로 보입니다. relu는 sparse representation을 가능하게 하며 이는 자연에서의 sparse data에 적합한 것처럼 보입니다. relu를 사용한 네트워크도 semi-supervised의 이점을 얻을 수 있지만 deep relut network는 이러한 unsupervised pre-traning 세팅이 없이도 좋은 성능을 보입니다. 따라서 이러한 결과는 deep 뉴럴네트워크에서의 purely supervised 러닝에서의 어려움을 이해하는 것으로 볼 수있으며 unsupervised pre-training 네트워크와의 성능 격차를 줄였습니다.

## 1 Introduction 

머신 러닝 연구자에서의 뉴럴네트워크와 계산 신경망학자들간에서 사용되는 뉴럴네트워크는 많은 차이가 존재합니다. 머신러닝 연구자들의 목적은 computationally efficeint learner를 모델링하여 새로운 example에서도 일반화를 잘할 수 있도록 하는 것입니다. 반면 계산 신경망 학자들의 목적은 신경과학적인 데이터를 추상화하여 그들이 포함하고 있는 원리들을 설명하는 것입니다. 이를 통해 미래의 생물학적 실험을 위한 가이드라인과 예측을 제공합니다. 

따라서 두가지 연구분야에서 일치되는 영역은 특히 인공지능 연구를 향상시킬 수 있는 부분이라고 할 수 있으며 연구할 가치가 있습니다. 이 논문에서는 머신러닝에서의 뉴럴네트워크 모델과 계산 신경망 분야에서의 모델의 차이가 relu activation을 통해 줄여들 수 있다는 것을 보입니다. 실험적인 결과가 relu activation이 특히 deep한 구조에서 더 잘 학습되도록 한다는 것을 보였습니다.

최근의 통계적 머신러닝에서의 이론적, 실험적 연구들은 deep learning에서의 learning 알고리즘의 중요성을 보였습니다. 이러한 학습 알고리즘은 포유류의 visual cortex에 대한 관찰에서 동기를 얻었고 이 visual cortex는 수많은 processing element들의 chain으로 구성되어져 있고 각각의 unit들은 visual input에 대한 각기 다른 representation과 연합되어져 있습니다. 영장류에서의 visual system은 보다 명확합니다. 이들의 visual ssystem은 edge detector, shape detector과 같은 processing unit들의 연속으로 되어져 있습니다. 흥미롭게도 딥러닝에서 배우는 feature들 또한 낮은 레벨의 feature들이 이러한 edge detector, 등으로 이루어져 있다는 것이 밝혀졌습니다. 또한 높은 레벨의 feature들은 camera movement와 같은 것에 대해 더 invariant 해진다는 것이 밝혀졌습니다.

딥러닝 구조를 학습시키는 것에 대해서는 몇가지 주요한 연구들이 수행되어졌습니다. 2006년에 DBN은 unsupervised learning 을 통해 각 레이어를 초기화 시켰습니다. 몇몇 연구자들은 이러한 unsupervised 세팅이 왜 학습에 도움이 되느지를 이해하려고 하였고 몇몇의 연구자들은 pure-supervised learning이 왜 딥러닝에서는 실패하는지를 이해하려고 했습니다. 머신러닝의 관점에서 이 논문은 이러한 이해에 대한 결과를 제공합니다.

우리는 딥러닝에서의 tanh, sigmoid 대신 relu 를 사용하는 것에 대해 제안합니다. 또한 L1 정규화를 activation value에 적용하여 sparsity를 높이고 unbounded activation으로 인한 수치해적적 문제를 방지할 수 있습니다. Nair, Hinton은 relu의 영향을 RBM에서 연구했으며 우리의 연구는 relu를 denosing autoencdoer 에 대해서 확장하고 image calssfication benchmark에서의 relu와 tanh간의 실험적 결과를 제공합니다. 

image, text data에 대한 우리의 실험은 뉴런이 0으로 활성화 되어져 있지 않거나 대부분 선형인 경우에 training 과정이 더 잘 진행됨을 보여줍니다. relu activation은 놀랍게도 unsupervised pre-train이 없이도 가장 좋은 성능을 달성합니다. 또한 이 논문은 unsupervised pre-train과 pure-supervised 네트워크간의 성능 격차에 대한 이해를 제공합니다. relu netwrok는 또한 unsupervised pre-train을 통해 성능의 이점을 얻을 수 있습니다. relu unit은 본질적으로 sparse network를 야기하며 보다 생물학적 뉴런의 활성화 반응과 가깝습니다. 이 논문은 머신러닝과 신경망 연구에서의 activation function, sparsity에서의 연결점을 제공합니다.

## 2 Background

## 2.1 Neuroscience Observations

생물학적 뉴런을 모델링 하기 위해서 activation function은 일종의 현재의 입력에서 기대되는 한 synapse의 firing rate 에 대한 함수입니다. activation function은 강하게 흥분되는 입력의 반대의 경우일때 강하게 억제되거나 흥분되는 경우에 각각 대칭
(linear), 혹은 반대칭(piece wise linear)이라고 불리며 반대되는 입력의 경우에 zero response이면 one-side 라고 불립니다. 머신러닝과 신경과학의 차이로 주로 고려될 것은 다음과 같습니다.

- brain energy expnese에 대한 연구는 뉴런이 정보를 sparse, distributed 방법으로 인코딩 한다는 것을 제안합니다. 같은 시간에 반응하는 뉴런의 비율은 대략 1~4% 정도로 추정됩니다. 이는 dense representation과 sparse representation간의 trade-off에 해당합니다. L1 penaly와 같은 추가적인 방법 없이는 평범한 feedforward netwrok는 이러한 특징을 가지지 못합니다. 예를 들어 sigmoid activation을 사용하면 1/2 을 중심으로 안정적인 상태를 유지하며 small weight로 초기화한 네트워크의 모든 뉴런은 1/2 보다 큰 지점에서만 firing 합니다.(?? 잘 모르겠다) 이는 생물학적으로 부정확하며 gradinet-based optimization에서 안좋은 영향을 끼칩니다.

- 신경과학과 머신러닝에서의 중요한 차이는 non-liner activation function에서 있습니다. 신경과학에서 보통 사용되는 것은 leaky intergrate-and-fire 를 사용합니다. 이는 다음과 같은 수식을 통해 현재 인풋과 firing rate간의 관계를 모델링 합니다. 머신러닝에서 사용되는 activation 모델은 보통 sigmoid, tanh를 사용합니다. tanh는 zero에서 steady state를 가지며 이는 최적화 관점에서 더 선호됩니다. 이는 zero 이점에서 비대칭적이며 이는 생물학적 뉴런에서 보이지 않는 특성입니다. 

## 2.2 Advantages of Sparsity

sparsity는 주요한 연구로 떠올고 있습니다. 이는 신경과학, 머신러닝 뿐만 아니라 통계학, 신호처리에서도 자주 언급됩니다. 신경과학에서는 visual system에서의 sparse coding의 관점에서 처음 소개되었습니다. 이는 또한 cnn을 이용하는 auto-encoder 변종 중 sparse distributed representation을 사용하는 모델의 핵심 요소입니다. 또한 DBN에서의 핵심 요소입니다. sparsity penalty는 신경과학과 머신러닝 모델에서 많이 사용되어져 왔습니다. 그러나 머신러닝에서 많이 사용된 sparse 뉴런은 결국에 작지만 nonzero인 activation 혹은 firing 확률을 가지게 됩니다.(진짜 zero firing이 아닌 매우 작은 firing을 가지게 된다는 소리인듯.. 어떻게 이전에 했는지는 모르겠지만) 우리는 relu를 사용하는 것이 real zero activation을 갖게 하고 ture sparse representation이 가능함을 보입니다. computational의 관점에서는 이러한 sparse representation이 다음과 같은 이유때문에 선호됩니다.  

- Information disentangling : 딥러닝 의 목적중 하나로 주장되는 것은 데이터에서의 변동을 설명하는 요인들을 dis-entangle 하는 것입니다. dense representation은 highly entangle 되어져 있습니다. 인풋의 작은 변화는 representation vector의 대부분을 변화시킵니다. 반면 representation이 <b>sparse, robust wrt small change of input 이라면 non-zero element 들은 입력의 작은 변화에 대해 대략적으로 보존되게 됩니다.</b>

- Efficient variable-size representation : 다른(variable?) 인풋은 아마 다른 정도의 정보를 포함하고 있을 것이며 보통 variable-size data-structure를 통해 표현됩니다. 이는 infromation을 컴퓨터를 통해 representation할때 보통 사용되는 방법입니다. <b>activation되는 뉴런의 갯수를 variable 하게 만드는 것은 모델이 representation의 효과적인 차원을 조절할 수 있게 합니다.</b>

- Linearly separability : sparse representation은 보다 linear separable하게 하거나 보다 적은 non-linearity를 통해 linear seperable되도록 합니다. 왜냐면 information을 보다 더 높은 차원에서 representation을 가능하게 하기 때문입니다. 게다가 sparsity는 인풋의 혀애를 반영할 수 잇습니다. text 데이터의 경우는 원본 데이터가 이미 sparse한 형태입니다.

- Dstributed but sparse : dense distributed representation은 매우 풍부한 representation입니다. purely local(왜 갑자기 local한 것이 나오지?)한 것보다 지수적으로 더 효율적입니다.sparse representation의 효율성은 non-zero feature의 갯수에 대해 지수적입니다.

너무 많은 sparsity를 강제하는 것은 동일한 뉴런을 가진 모델의 경우 예측 성능이 저하될 수 있습니다. 이는 effective capacity를 감소시킬 수 있습니다.

## 3 Deep Rectifier Networks

## 3.1 Rectifier Neurons

신경과학에서는 대뇌 피질 뉴런이 maximum saturation regime에는 거의 존재하지 않고 이들의 activation function이 rectifier에 의해 근사될 수 있음을 시사합니다. 뉴럴네트워크에서의 rectifier activation을 포함한 이전의 연구들은 대부분 recurrenct network의 형태에서 다룹니다.

relue는 one-sided 함수이며 이들은 대칭적 혹은 반대칭적 sign을 강제하지 않습니다. 대신 흥분성 입력 패턴에 반대되는 입력에 대한 반응이 0이 되도록 합니다. 그러나 두개의 shared relu unit을 통해 비대칭적, 반대칭적 activation을 얻을 수 있습니다.

<b>Advantage</b> : relu는 네트워크가 sparse representation을 가능하도록 허락합니다. 예를 들어 uniform init wight를 하게 되면 50% 정도의 hidden unit의 value는 0에 가깝도록 됩니다. 이러한 zero에 가까운 activation의 비율은 L1 regularization과 같은 것들로 증가시킬 수 있습니다. 생물학적으로 더 가깝게 만들 뿐만 아니라 sparsity는 수학적으로 장점을 가집니다. 

figure 2에서 볼 수 있듯이 네트워크의 non-linearity는 오직 개별 뉴런이 활성화 되었는지에 대한 여부와 관련된 경로 선택에서만 나타나게 됩니다. 주어진 입력에 대해 어떤 뉴런의 부분집합만이 활성화 되었을때 non-zero activation의 부분집합에서는 linear 한 계산을 하게 됩니다. 즉 뉴런의 부분집합이 주어진다면 output은 인풋에 대해 단지 선형함수일 뿐입니다. (인풋의 큰 변화는 non-zero activation set을 변화시킬 수 있습니다.) 각각의 뉴런으로 부터 계산된 함수 혹은 네트워크의 출력은 linear by part가 됩니다. 우리는 이러한 모델을 일종의 <b>exponential number of linear models that share parameter</b> 로 볼 수 있습니다. 이러한 linearity 덕분에 활성화된 뉴런의 gradient flow가 잘 작동합니다.(sigmoid혹은 tanh로 부터 나타나는 gradinet vanishing 효과가 없게 됩니다.) 또한 mathematical investigation 또한 매우 쉽습니다. 연산량또한 매우 적어지게 됩니다. sigmoid, tanh에서와 같은 exponential 에 대한 계산이 필요가 없습니다. sparsity 또한 연산을 줄여주게 됩니다.


<b>Potential Problems</b> : 0에서 hard saturation 되는 것에 대한 하나의 가설은 back-prop을 막음으로써 optimization이 잘 안되게 할 수 있습니다. 이러한 잠재적 문제를 평가하기 위해서 soft-plus activation에 대해서도 실험을 진행하였습니다. soft-plus는 relu의 soft version입니다. 이를 통해 true sparsity에 대한 것을 잃지만 training과정이 더 쉬운 것을 기대할 수 있습니다. 그러나 실험적인 결과는 이러한 가설과 모순되며 hard zero가 supervised training에서 실질적으로 도움이 된다는 것을 제안합니다. 우리는 <b>hard non-linearity가 optimization에 어려움을 주지 않는데 이는 gradient가 각 레이어의 non-zero가 아닌 unit을 통해 다른 path를 통해 전달될 수 있을 것이라고 생각합니다.</b> 즉 보다 고르게 distributed 되어 있지 않고 단지 On unit이 optimization을 더 쉽게 할 것이라는 가설을 세웠습니다. 다른 문제는 activation의 unbounded 때문에 발생할 수 있습니다. 이는 잠재적인 수치해석적 문제를 방지하는 regularzier를 사용 할 수 있습니다. 그러므로 우리는 activation value에 L1 penalty를 적용했습니다. 이는 또한 추가적인 sparsity를 강제합니다. 또 하나의 문제는 data에서의 효과적인 대칭,반대칭적 representatoin을 하기 위해서는 relu는 2배의 hidden unit이 필요합니다.

마지막으로 relu network는 ill-conditioning of parametrization에 당하기 쉽습니다. bias, weight는 같은 netwrok ouput을 유지하면서 다르게 scaling 될 수 있습니다. 다른 weight, bias인데도 같은 output을 뱉는.. 즉 ill-condition 된다. 

## 4.2 Sentiment Analysis

Nair and Hinton은... relu가 특히 image-related task에 효과적이라고 한다 한번 찾아보자.


# Learning Deep Architectures for AI

## 13.2 Why Sparse Representations and Not Dimensionality Reduction

우리는 이 논문에서 뇌와 마찬가지로 고정된 크기의 representation을 가지려고 한다면, sparse representation이 example마다 variable size representation을 가지도록 하는 것보다 더 효율적이라고 주장합니다. learning theory에 따르면 좋은 일반화를 얻기 위해선 전체 train set을 encode하는데 필요한 총 bits 수가 train set 크기에 비해 더 작아야 한다고 말합니다. 많은 도메인에서 서로 다른 example들은 서로 다른 information 정보를 가지고 있습니다. 이것이 image compression 알고리즘이 보통 다른 image에 대해서 다른 갯수의 bits를 통해 encode하는 이유입니다. (비록 그들이 보통 같은 차원을 가질 지라도?? )

반면에 PCA, ICA와 같은 linear 차원축소, LLE, Isomap과 같은 non-linear 차원축소 알고리즘은 각기 다른 example을 같은 크기의 low-dimensional space로 매핑합니다. 위와 같은 논의에 따라서는 각각의 example을 variable-length representation으로 매핑하는 것이 더 효율적입니다. 주장을 간단히 하기 위해, binary vector representation이라고 가정해 봅시다. 만약 우리가 각각의 example을 고정된 길이의 representation으로 매핑한다고 생각해봅시다. 이러한 representation에서의 가장 좋은 솔루션은 대부분의 example을 표현할 수 있는 자유도를 representation이 가지는 것입니다. 반면 고정된 bits vector를 작은 사이즈의 variable-code로 압축하는 것으로 보다 많은 exmaple을 표현할 수 있습니다. 이제 우리는 두가지 종류의 representation을 가졌습니다. 하나는 fixed-length를 가지며 하나는 보다 smaller하며 variabel-size length를 가집니다. 이 variable size vecotr는 fixed-length에서의 압축 단계를 거쳐 얻어진 것입니다. 예를 들어 만약 우리의 fixed-length repressentation이 매우 높은 확률로 0 이된다면 (sparsity condition) 이들을 fixed-length vector로 압축하는 것은 매우 쉬운 일일 것입니다.(average by amount of sparsity)

sparsity가 선호되는 또 다른 이유는 fixed-length representation이 soft-max linear unit과 같은 또다른 추가 처리를 위한 입력으로 사용되므로 이는 해석이 쉬워야 한다는 것입니다. 매우 높게 압축된 encoding은 보통 완벽하게 얽혀 있으므로 모든 bits를 살펴보지 않는 이상 bits의 부분집합에 대한 해석을 할 수 없습니다. 대신 fixed-length sparse representation이 각각의 bits or bits의 부분집합이 해석가능하길 원합니다. 즉 각각의 bits가 입력의 의미있는 부분에 해당하고 데이터의 변동을 포착할 수 있기를 원합니다. 음성 신호를 예로 들면 몇몇 bits가 발화자의 특징을 encode하고 다른 bits가 발음에 대한 일반적인 feature를 encode한다고 생각해 봅시다. 이러한 bits를 가지고는 데이터에 대한 변동에 대한 factor를 풀어헤칠 수 있으며 factor의 부분집합들을 가지고도 충분히 어떠한 예측과 관련된 task를 수행할 수 있습니다.

spare representation에 대한 다른 정당화는 Ranzator가 제안하였습니다. 이러한 관점에서는 학습된 representation이 sparsity와 같은 다른 제약조건이 있는 경우에, partition functtion이 명시적으로 최대화 되지 않거나 근사적으로 최대화 되는 경우에도 우수한 모델을 얻을 수 있는 방법을 설명합니다.auto-associator(auto-encoder like한건가?) 가 배운 representation이 sparse하다면 이러한 것은 모든 인풋 패턴들을 잘 reconstruction할 수 없습니다. train set에서의 reconstruction error를 줄이기 위해 이러한 auto-associator는 데이터 분포에 대한 통계적 일반성을 포착해야만 합니다. 첫번째로 Ranzato는 free energy를 reconstruction error의 형태로 연결합니다.(reconstruction error가 sum의 형태가 아닌 maximizing 형태일때?) 이러한 경우 reconstruction error를 줄이는 것은 free energy의 양을 줄이는 것이 됩니다. 분모(partition function) 이 모든 가능한 인풋에서의 분자의 합이므로 우리는 대부분의 인풋 구성에 대해 reconstruction error를 높게 만들것입니다.(?뭔소리야?). 이는 encoder(인풋을 representation으로 매핑하는 사상) 이 입력 패턴의 모든 가능성을 나타낼 수 없도록 제약을 주면 됩니다. (대부분의 가능한 인풋 패턴에 대해 reconstruction error가 높은 경우?)

이를 위한 하나의 접근 방법은 Ranzato에 의해서 이루어진 representation에 sparsity penalty를 주는 것입니다. 이는 training criterion에서 이뤄질 수 있습니다. 이러한 방법에서는 log-likelihood의 수식에서 pratition function과 관련된 그라디언트는 완변히 무시되고 이는 hidden unit code에 대한 sparsity penalty로 대체됩니다. 흥미롭게도 이러한 아이디어는 RBM의 학습과정에 도움을 줄 수 있고 이는 log of partition function에 대한 gradient의 estimator의 근사로만 사용됩니다. 만약 우리가 hidden representation에 sparsity 패널티를 추가하게 되면 우리는 근사에 대한 약점을 보완할 수 있습니다. 이는 가장 가능할만한 input configuration의 free energey를 증가시킴으로써 가능하며 input example의 Contrastive divergence of negative phase로 부터 얻어진reconstructed neighbors의 free engergy를 증가시킵니다.


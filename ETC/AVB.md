# Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks

VAE는 학습데이터로 부터 복잡한 확률분포를 학습 할 수 있는 표현력이 뛰어난 latent variable model 입니다. 그러나 모델의 결과는 inference model(p(z|x)) 의 성능에 매우 의존적입니다. 우리는 variational autoencoder를 학습시키기 위한 AVB라는 기술을 소개합니다. 우리는 보조의 disriminative 네트워크를 소개하고 이는 two-player 게임을 통해 maximum-likelihood 문제를 해결합니다. 또한 VAE와 GAN간의 이론적인 연결을 소개합니다. 우리의 방법이 nonparametric 제한 안에서 생성 모델의 파라미터에 대한 정확한 maximum-likelihood 를 통해 파라미터를 할당 할 수 있음 을 보이고 이는 observation이 주어졌을때의 정확한 posterior variable인 것을 보입니다. VAE, GAN을 결합한 다른 방식과는 다르게 우리의 접근 방식은 이론적으로 명확하고 표준적인 VAE의 장점을 가지며 구현하기 쉽습니다.
 
### Introduction

머신러닝에서의 생성모델은 unlabeled 데이터셋에서 학습이 가능하며 학습이 끝난후에 새로운 데이터를 생성 할 수 있는 모델입니다. 새로운 데이터를 생성하는 것은 가지고 있는 학습 데이터에 대해 모델이 이해 할 수 있어야 하며 이러한 모델들은 unsupervised learning의 핵심 요소로 간주됩니다.

최근 몇년 동안 생성 모델은 점점 더 강력해 지고 있습니다. PixelRNN, PixelCNN, real NVP, Plug & Play generative network등의 연구들이 수행되어져 왔고 가장 중요한 두가지는 VAE, GAN 입니다.

VAE, GAN은 각각의 장단점을 가지고 있습니다. natural image의 representation을 배우느것에 적용될때 GAN은 일반적으로 더 선명한 결과를 얻습니다. VAE는 generative, inference 모델을 모두 가지고 있기 때문에 매력적입니다. (GAN은 P(z|x) 를 알 수가 없고 단지 생성만 해낼 수 있다.). 또한 VAE는 종종 더 나은 log-likelihood를 보인다는 연구 결과가 있습니다 최근 소개된 BiGAN은 GAN에 inference 모델을 추가했지만 reconstruction 결과가 종종 인풋과 너무 비슷하며 의미적으로만 비슷할 뿐 픽셀 값의 관점에서는 그렇지 않은 경우가 많습니다. (전체적으로는 그럴듯해 보이지만 세부적으로는 그렇지 않다는 뜻인듯)

VAE가 선명한 이미지를 생성하지 못하는 것은 학습 동안의 inference model이 true posterior를 포착할 만큼 충분히 expressive 하지 않다는 것입니다. 최근의 연구에 따르면 더 expressive 한 모델이 시각적으로나 log-likelihood 측면에서 더 나은 결과를 얻을 수 있습니다. Chen은 expressive inference model이 decoder가 latent space를 모두 사용할 수 있도록 해준다고 제안합니다

이 논문에서는 AVB라는 것을 제안하며 이는 variataional autoencoder를 학습 할 수 있는 기술을 제안합니다. 우리는 non-parametric 제한 하에서 maximum-likelihood를 달성할 수 있음을 보였고 생성 모델이 정확한 posterior를 포착하도록 합니다.

VAE, GAN을 결합하려는 시도가 있었지만 대부분의 시도들은 maximum-likelihood의 관점이 아니였고 이는 maximum-likelihood 의 파라미터를 찾지 못합니다. 예를들어 AAE에서는 VAE에서 나타나는 KL-divergence 를 adversarial loss를 대체합니다. AAE가 maximum-likelihood objective의 하한을 최대화 하지 않음에도 AAE는 우리의 접근 방법의 일종의 approxiation으로 볼 수 있습니다. 이는 섹션 6.2에 기술하였고 AEE와 maximum-likelihood 학습과의 관계를 보여줍니다.

생성모델의 관점을 벗어나서, AVB는 뉴럴 네트워크를 이용하는 새로운 variational bayes를 수행하는 한 방법입니다. 이는 피규어 1에 나타나있고 AVB를 non-trivial unnormalized 확률 분포를 샘플링 하기 위해 학습시켰습니다. 이는 확률 모델의 posterior 를 정확하게 근사하는 것을 허용합니다. 이러한 variational method를 다룰 수 있는 유일한 방법은 Stein Discrepancy 방법입니다. 그러나 이 방법은 revese KL-divergence를 목표로 하지 않음으로 latent variable을 학습하기 위한 lower bound로 사용할 수없습니다. 우리의 작업의 공헌은 다음과 같습니다.

- 우리는 adversarial 학습방법을 이용하여 variational autoencoder를 임의의 복잡한 inference model로 사용할 수 있게 합니다.

- 우리의 방법에 이론적인 통찰을 제공하며 non-parametric 제한 하에서 우리의 방법이 true posteriro distribution을 배울 수 있고 이는 파라미터에 대한 true maximum-likelihood를 기반으로 합니다.

- 우리의 모델이 rich posteriro distribution을 학습 할 수 있고 복잡한 데이터 셋에서 데이터를 생성할 수 있음을 경험적으로 보여줍니다.

### Background

우리의 모델은 VAE의 확장이므로 VAE의 간단한 리뷰부터 시작합니다.

VAE는 latent variable이 주어졌을때 visible variable을 생성하는 parametric generative model(p(x|z)) 를 명시하며, latent variable에 대한 사전분포 p(z) 와 approximate inference model q(z|x) 를 모델링합니다. 이는 다음과 같이 나타낼 수 있습니다.


<img src="https://latex.codecogs.com/gif.latex?logp_%5CTheta%20%28x%29%20%5Cgeq%20-KL%28q_%5CPhi%20%28z%7Cx%29%29%20&plus;%20E_%7Bq_%5CPhi%20%28z%7Cx%29%7D%5Blogp_%5CTheta%28x%7Cz%29%5D" />

우변은 variational lower bound 혹은 ELBO 라고 불리며 q_φ(z | x) = p_θ(z | x) 인  φ가 존재한다면 밑의 수식이 성립합니다.

<img src="https://latex.codecogs.com/gif.latex?logp_%5CTheta%20%28x%29%20%3D%20max_%5CPhi%20-KL%28q_%5CPhi%20%28z%7Cx%29%29%20&plus;%20E_%7Bq_%5CPhi%20%28z%7Cx%29%7D%5Blogp_%5CTheta%28x%7Cz%29%5D" />

하지만 일반적으로 참이 아니며 우리는 (2.2) 수식에서 등호가 아닌 부등식을 얻게 됩니다.

maximum-likelihood 학습을 수행한다면 우리가 최적화해야 할 식은 marginal log-likelihood 입니다.

<img src="https://latex.codecogs.com/gif.latex?E_%7BpD%28x%29%7D%5Blogp_%5CTheta%28x%29%5D" />

p_D 는 data distribution입니다. 불행하게도 log p_theta(x) 를 계산하기 위해서는 p_theata (x,z) 에서 z를 marginal 해야 하며 이는 intractable 합니다. variational bayes 는 (2.1) 의 수식을 이용하여 intractable proble을 최적화 문제로 풀어 냅니다. 이는 다음과 같습니다.

<img src="https://latex.codecogs.com/gif.latex?max_%5CTheta%20max_%5CPhi%20E_%7Bp_D%28x%29%7D%5B-KL%28q_%5CPhi%20%28z%7Cx%29%2C%20p%28z%29%29%20&plus;%20E_%7Bq_%5CPhi%20%28z%7Cx%29%29%7D%5Blogp_%5CTheta%28x%7Cz%29%5D%5D"/>

(2.1) 수식 때문에 우리는 true maximum-likelihood objective의 lower bound를 최적화 하는 것입니다.

자연스럽게 이 lower bound의 성능은 inference model q(z|x) 에 의해 달려있습니다. q(z|x) 가 diagonal covariance matrix를 갖는 gaussian distribution 으로 모델링 되며 각 mean, variance vector는 X를 인풋으로 받는 뉴럴네트워크에 의해 parameterized 된다. 이 모델은 x에 대한 의존성이 매우 유연하지만 이의 z space가 매우 제한되어져 있어 잠재적으로 generative model의 결과를 제한합니다. 실제로 natural image에 표준적인 variational autoencoder를 적용하면 흐린 이미지를 얻는 것이 관찰됩니다

### Method

이 연구에서는 우리는 black-box inference model qφ(z | x) 를 사용하는 것을 보여주고 adversarial training을 활용하여  maximum likelihood assignment θ∗ 를 얻고 true posteriro pθ∗(z | x) 와 가까운 qφ∗(z | x) 을 얻어냅니다. 피규어 2의 왼쪽에는 일반적인 VAE가 보여집니다. 오른쪽은 우리의 flexible black-box inference model을 보여줍니다. Gaussian inference model인 표준적인 VAE와는 다르게 우리는 inference model의 추가적인 인풋으로 noise epsilon_1 을 포함합니다. 이를 통해 inference network가 복잡한 확률 분포를 학습 할 수 있게 합니다.

### 3.1 Derivation

우리의 방법을 도출하기 위해서 위에서 봤던 최적화 문제를 밑과 같이 재구성 합니다.


<img src="https://latex.codecogs.com/gif.latex?max_%5CTheta%20max_%5CPhi%20E_%7Bp_D%28x%29%7DE_%7Bq_%5CPhi%20%28z%7Cx%29%29%7D%5Blogp%28z%29%20-%20logq_%5CPhi%28z%7Cx%29%20&plus;%20logp_%5CTheta%28x%7Cz%29%5D"/>

qφ(z|x) 를 뉴럴네트워크에 이해 parameterized된 gaussian 같은 것으로 명시적으로 표현 할때 우리는 reparameterization 트릭을 사용해 sgd로 최적화 할 수 있습니다. 불행히도 피규어 2b와 같이 qφ(z|x)를 black-box 로 정의할 때는 불가능 합니다.

우리의 접근방법의 아이디어는 implicit representation을 사용하여 이러한 문제를 우회하는 것입니다.

<img src="https://latex.codecogs.com/gif.latex?logp%28z%29%20-%20logq_%5CPhi%28z%20%7Cx%29" /> (3.2)

추가적인 real-valued discriminative network T(x|z)의 optimal value를 사용하여 이러한 문제를 다룹니다. 더 구체적으로는 보기 위해 given qφ(x|z) 일때의 disciriminator T(x|z) 의 objective 를 살펴봅시다.

<img src="https://latex.codecogs.com/gif.latex?max_T%20E_%7Bp_%7BD%28x%29%7D%7DE_%7Bq_%7B%5Cphi%28z%7Cx%29%7D%7Dlog%5Csigma%20%28T%28x%2Cz%29%29%20&plus;%20E_%7Bp_%7BD%28x%29%7D%7DE_%7Bp%28z%29%7Dlog%281-%5Csigma%28T%28x%2Cz%29%29%29" /> (3.3)

여기서 σ(t) 는 sigmoid-function을 나타냅니다. 직관적으로 T(x,z) 는 p(x)p(z), p(x)q(z|x) 로 부터 독립적으로 샘플링된 샘플들을 구별합니다.

이론적 분석을 간단히 하기 위해서 우리는 T(x|z) 가 x,z의 변수를 가지는 어떠한 함수를 나타낼 수 있다고 가정합니다. 이러한 가정은 종종 non-parametric limit라고 불리며 이는 딥뉴럴 네트워크가 univesal function approxmiator인 것에 의해 정당화 됩니다.

(3,3) 에서의 optimal discriminator T*(x,z)은 (3.2) 에 음수를 취한 것과 동일합니다. 

<b> Propostion 1.</b>

<img src="https://latex.codecogs.com/gif.latex?For%20%5C%2C%20p_%7B%5Ctheta%7D%28x%20%7C%20z%29%20%5C%2C%20and%20%5C%2C%20q_%5Cphi%28z%20%7C%20x%29%20fixed%2C%20%5C%5C%5B12pt%5D%20the%20%5C%2C%20optimal%20%5C%2C%20discriminator%20%5C%2C%20T*%20%5C%2C%20according%20%5C%2C%20to%20%5C%2Cthe%5C%2C%20objective%20%5C%2Cin%20%5C%2C%283.3%29%5C%2C%20is%20%5C%2Cgiven%20%5C%2C%20by%20%5C%2C%20%5C%5C%5B12pt%5D%20T*%28x%7C%20z%29%20%3D%20log%20q_%5Cphi%28z%20%7C%20x%29%20-%20log%20p%28z%29%3A%20%283.4%29" /> 

결국에..discriminator를 최대화 하는 것은 KL-divergence 를 구하게 되네.

proof. 이 증명은 보충자료에 나와있습니다. proposition 1은 (2,4) 를 다음과 같이 쓸 수 있게 해줍니다.

<img src="https://latex.codecogs.com/gif.latex?max_%5Ctheta%20max_%5Cphi%20E_%7Bp_D%28x%29%7DE_%7Bq_%5Cphi%20%28z%7Cx%29%29%7D%5B-T%5E*%28x%2Cz%29%20&plus;%20logp_%5Ctheta%28x%7Cz%29%5D" />  (3.5)

여기서 T*(x,z) 는 (3,3) 을 최대화 한 결과입니다.

(3.5) 를 최적화 하기 위해서 우리는 (3.5) 에서의 theta, phi에 대한 그라디언트를 구해야 합니다. theta에 대한 그라디언트를 구하는 것은 간단하지만 phi에 대한 그라디언트를 구하는 것은 매우 복잡한데 T*(x,z) 또한 phi 그 자체에 의존한 optimization problem의 솔루션이기 때문입니다. 그러나 다음의 proposition은 T*(x,z) 에 대한 그라디언트를 취할 필요가 없다는 것을 보여줍니다.

<b>Propostion 2.</b> 


<img src="https://latex.codecogs.com/gif.latex?Eq_%7B%5Cphi%28z%7Cx%29%7D%28%5Cbigtriangledown_%7B%5Cphi%7DT%5E*%28x%2Cz%29%29%20%3D%200%20%3A%20%283.6%29" />  

proof. 이는 보충 자료에서 찾을 수 있습니다.

vae 논문의 reparmeterization 트릭을 사용하면 다음과 같이 나타낼 수있습니다.
<img src="https://latex.codecogs.com/gif.latex?max_%7B%5Ctheta%5Cphi%7D%20E_%7Bp_D%28x%29%7DE_%7Bq_%5Cepsilon%7D%5B-T%5E*%28x%2Cz_%7B%5Cphi%7D%28z%2C%5Cepsilon%29%29%20&plus;%20logp_%5Ctheta%28x%7Cz_%7B%5Cphi%7D%28z%2C%5Cepsilon%29%29%5D" />  

적절한 함수 z(x,e)와 proposition 1은 theta, phi에 관한 unbiased estimate of gradient를 찾을 수 있게 합니다.

### 3.2 Algorithm

이론적으로 proposition 1,2 는 (2.4) 를 SGD로 바로 구할 수 있게 합니다. 그러나 T*(x,z) 를 항상 최적으로 유지하는 것은 매우 비싼 계산을 요구합니다. 우리는 (3.3), (3.7) 의 optimization problem을 two-player 게임으로 생각합니다. proposition 1,2 은 (2.4) 의 내쉬 균형이 obejective의 stationary point를 생성한다는 것을 보여줍니다.

실무적으로 step size h_i 를 (3.3), (3.7) 에 모두 적용한 SGD를 통해 내쉬-균형을 찾으려고 합니다. 여기서 우리는 뉴럴네트워크 T의 파라미터를 vector psi로 나타냅니다. 이 알고리즘이 수렴한다는 보장은 없더라도 이 알고리즘의 fix point는 (2.4)의 stationary point를 생성합니다.

θ , T 를 고정해놓고 (3.5) 를 최적화 하는 것은 encoder 네트워크가 deterministic function이 됩니다. 이것은 보통의 GAN 에서의 일반적 문제이기도 합니다. 이 때문에 discriminative T 를 계속적으로 Optimal 하게 만드느 것이 중요합니다. 따라서 알고리즘1의 변형은 adversary 네트워크에는 여러번의 SGD 업데이트를 수행하고 generative model에는 한번의 업데이트를 수행합니다. 그러나 다른 언급이 없는 이상 우리는 AVB의 간단한 알고리즘 1 버전을 사용합니다.

### 3.3 Theoretical result

섹션 3.1 에서 우리는 (2.4) 에 대한 variational lower bound를 sgd를 통해 최적화 하는 방법으로 AVB를 도출했습니다. 이 섹션에서는 게임이론의 관점에서 알고리즘 1의 특징에 대해 분석합니다. 다음의 propostion 이 보여주듯이 알고리즘 1의 내쉬-균형은 (2.4) 의 global optima를 생성합니다.

<b>proposition 3.</b>의 증명은 추가자료에 나와 있습니다.

우리의 parameterized q(z|x) 는 q(z|x) 가 latent space에 대한 any probability density가 되도록 합니다.

<b>Corollary 4</b> T를 z,x를 인풋으로 받는 any function 이라고 가정하고 q(z|x) 를 any probability density 라고 하면. (3.3), (3.7) 의 내쉬 균형은 다음과 같다.

1. θ* 가 maximum-likelihood assignment
2. qφ*(z|x) 가 pθ*(z|x) 와 같아질때.
3. T*가 x,z 에 대해 pointwise mutual information 일때.

### 4. Adpative contrast 

non-parametric limit(T(x,z) 가 any function이 될 수 있다.)일때 우리의 방법이 정확한 결과를 산출합니다. 실무적으로 T(x,z) 는 학습 과정에서 optimal T*(x,z) 에 충분히 근접하지 못할 수 있습니다. 이 이유는 AVB가 pD(x)qφ(z|x) 를 계산하지만 true pD(x)p(z)와는 근본적으로 다릅니다. 그러나 로지스틱 회귀는 유사한 두 분포를 비교할 때 likelihood-ratio estimation으로 잘 작동하는 것으로 알려져 있습니다.

따라서 estimate의 성능을 향상시키기 위해 우리는 auxiliary conditional probability distribution  rα(z|x) 를 도입합니다. 예를들어 rα(z|x)는 qφ(z |x) 의 mean, variance 와 일치하는 diagonal covariance matrix를 가지는 gaussian이 될 수 있습니다. 이러한 auxilary 분포를 사용하여 (2.4) 의 variational lower bound를 다음과 같이 쓸 수 있습니다.

<img src="https://latex.codecogs.com/gif.latex?E_%7Bp_D%28x%29%7D%5B-KL%28q_%7B%5Cphi%28z%7Cx%29%7D%20%2C%20r_%5Calpha%28z%7Cx%29%29&plus;%20E_%7Bq_%5CPhi%20%28z%7Cx%29%29%7D%5B-logr_%5Calpha%28z%7Cx%29&plus;%20logp_%5Cphi%28x%2Cz%29%5D%20%5D%20%3A%20%284.1%29" />  

우리가 이미 rα(z|x)의 분포를 알기 때문에 (4.1) 의 두번째 텀은 theta, phi의 sgd를 따르게 됩니다. 그러나 섹션3에서 기술한 것과 같이 AVB를 이용해 첫번째 텀을 구할 수 있습니다. rα(z|x)가 q(z|x) 를 잘 근사한다면 rα(z|x)를 이용한 KL-divergence 텀이 원래 보다 작아지며 이는 adversary 학습 방법이 정확한 probability ratio를 배울 수 있게 합니다.

p(z) 대신에 adaptive distribution 인rα(z|x)를 근사하기 때문에 이 기술을 Adaptive Constrast(AC) 라고 부릅니다. 이를 사용하여 generative model, inference model은 maximize 하게 최대화 되게 됩니다.

<img src="https://latex.codecogs.com/gif.latex?E_%7Bp_D%28x%29%7DE_%7Bq_%5CPhi%20%28z%7Cx%29%29%7D%5B-T%5E*%28x%2Cz%29%20-logr_%5Calpha%28z%7Cx%29&plus;%20logp_%5Cphi%28x%2Cz%29%20%5D%20%3A%20%284.1%29" /> (4.2)

T*(x,z) 는 optimal discriminator 이며 rα(z|x)과 q(z|x) 를 구별합니다.

이제 rα(z|x)가 q(z|x)와 mean, variance vector가 일치하는 대각 공분산 행렬을 갖는 가우시안 분포로 가정해 보자. KL-divergence가 reparameterization 하에 invariant함으로 (4.1)의 첫번째 텀 은 다음과 같이 쓸 수 있다.

<img src="https://latex.codecogs.com/gif.latex?E_%7BpD%28x%29%7D%5BKL%28%5Ctilde%7Bq%7D_%5Cphi%28%5Ctilde%7Bz%7D%7Cx%29%2C%20r_0%28%5Ctilde%7Bz%7D%29%29%5D
" /> (4.3)

여기서 tilde_z 는 normalized vector를 나타내며 r_0(tilde_z) 는 mean 0, variacne 1인 가우시안을 나타냅니다. 이 방법은 adversary(discriminative) 는 오직 q(z|x) 과 unit gaussian 과의 비교만 하면 됩니다. 

실무적으로 우리는 mean, variance vector를 montecarlo estimate 를 사용하여 추정하며 보충자료에서 계산을 효율적으로 하기 위한 네트워크 구조를 설명합니다.



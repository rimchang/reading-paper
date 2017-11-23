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
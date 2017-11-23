# DENOISING CRITERION FOR VARIATIONAL AUTOENCODING FRAMEWORK

### Abstract

Denosing autoencoder는 input level에 노이즈를 추가함으로써 깨끗한 input을 reconstruct하도록 훈련되어졌습니다. 반면에 VAE는 중간의 stochastic hidden layer에서 노이즈가 추가되고 regularizer가 이 noise를 다루게 됩니다. 이 논문에서는 우리는 input, stochastic layer 모두에서 노이즈를 추가하는 것이 유리할 수 있음을 보이고 개선된 objective function으로의 variational lower bound를 제안합니다. 입력에 노이즈가 추가되면 표준적인 VAE lower bound는 input noise에 대한 encoder conditional distribution( q(z|x) )을 marginalizing 하며 이는 training criterion을 intractable하게 만듭니다. 대신에 우리는 인풋에 노이즈가 추가 되었을대도 tractable한 training criterion을 제안합니다. 우리의 DVAE가 기존의 VAE, IWAE 보다 더 나은 average log-likelihood를 가지는 것을 보여줍니다.

### Introduction

Variational inference는 MCMC와 함께 approximate bayesian inference의 주요한 요소중 하나입니다. posterior를 풀어 MCMC에 비해 많은 연구자들에게 인기가 있습니다. variational inference의 장점은 (1) optimization tool의 이점을 얻을 수 있고 (2) 최적화 문제를 품으로써 MCMC보다 빠른 학습 과정 (3) MCMC는 언제 샘플링을 끝내야 할지 모르지만 variational inference는 criterion이 분명해 언제 끝내야 할지가 분명합니다.

variational inference의 주목할만한 개선은 inference network를 posteriro distribution의 근사로 사용하는 것입니다. 기존의 전통적인 variational inference는 각각의 latent variable에 대해 다른 variational parameter를 필요로 하지만 inference network를 이용하면 각각의 latent variable에 대한 approxiamte posterior distribution은 단지 q(z|x) 일 뿐이고 latent variable 사이의 파라미터를 공유하게 됩니다. reparameterization 트릭과 REINFORCE 와 같은 학습방법과 결합하여 variational inference를 큰 데이터셋에 효율적으로 학습할 수있게 되었습니다.

이러한 발전에도 불구하고 true posterior distribution을 정확히 근사할 만한 유연한 variational distribtution을 얻는 것이 여전히 큰 문제입니다. 예를 들어, VAE 에서는 효율적인 학습을 위해 latent variable의 각 차원이 독립인 것으로 가정합니다. (lambda가 주어졌을때 독립이라고 가정한다) 또한 각각의 latent variable은 unit gaussian으로 모델링합니다. VAE가 MINIST같은 간단한 이미지들을 잘 생성하지만 실세계의 더 복잡한 문제에 적용하기 위해서는 variational distribution에 대한 제약조거들을 완화할 필요가 있습니다. 최근들어 이러한 제약을 완하하기 위한 노력들이 있습니다. Saliman은 MCMC와 variational inference를 통합하여 true distribution에 더 가깝게 하였습니다. 비슷한 아이디어인 latent space에  invertible non-linear transform 을 사용한 방법도 있습니다.


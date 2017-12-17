# Understanding image representations by measuring their equivariance and equivalence

### Abstract

HOG, CNN 에서의 image representation의 중요성에도 불구하고 이들에 대한 이론적인 이해는 제한적입니다. 이러한 이론적 격차를 해소하기 귀해 우리는 세가지 equivaraince, invariance, equivalence라는 수학적 특징들을 연구합니다. equivariance는 인풋 이미지의 transformation이 representation에 어떤 영향을 주는지를 연구합니다. invariance는 trasformation이 representation에 아무런 영향을 주지 않는 특별한 경우입니다. equivalence는 다른 파라미터를 가진 CNN의 두개의 representation이 같은 visual information을 포착하는지 아닌지를 연구합니다. 이러한 특징들을 가지기 위한 몇몇의 경험적인 방법들이 제안되었습니다. 이들은 CNN에 transformation layer, stitching layer를 넣으므로써 이러한 특징들을 가지도록 합니다. 이들은 유명한 구조들에 적용되어서 그들의 구조에 대한 해석을 가능하게 하며 어떠한 geometric invariance 들을 가지도록 하는지 연구되었습니다. 이 논문에서는 보다 이론적인 것에 초점을 맞췄으며 structed-output regression(? 이게 뭐지) 에 대해서도 적용하여 보여줍니다.

### 1. Introduction

image representation은 지난 20년 동안 컴퓨터 비전에서 주요한 부분입니다. 주요한 예들로는 teston, HOG, SIFT, BOW, sparse and local coding, super vector coding, VLAD, Fisher Vector, CNN 등이 있습니다. 그러나 이러한 인기에도 불구하고 이들 representation에 대한 이론적인 이해는 여전히 제한적입니다. 일반적으로 좋은 representation은 invaiance와 discriminablilty 를 모두 가지고 있어야 한다고 여겨집니다. 그러나 이러한 특징을 나태내는 것들은 다소 모호한데 예를들어 representation 안에 어떠한 invariance들이 포함되고 어떻게 이러한 invariance들이 얻어지는지는 종종 불분명 합니다.

이 논문에서는 우리는 image representation을 연구하는 새로운 접근방법을 제안합니다. 우리는 image representation을 어떠한 임의의 mapping function φ 으로 나타내며 이는 인풋 이미지 x를 vector φ(x) (d 차원) 으로 매핑하는 함수로 생각합니다. 그리고 이 함수에 대한 수학적 특징들을 연구하게 됩니다. 우리는 특히 세가지의 특성에 대해 초점을 맞춥니다. 첫번째는 equivariance 이며 이는 인풋 이미지의 transformation에 대해 representation이 어떻게 변하는지를 보여줍니다. 우리는 HOG, CNN의 representation이 인풋 이미지로 부터 어떻게 예측이 쉬운 representation으로 나타내는 지를 보여줍니다. 이러한 equivariant transformation이 데이터로부터 경험적으로 학습된다는 것을 보여주고 더 중요한 것은  이들이 representation output의 단순한 linear transformation으로 이들의 양을 측정 할 수 있다는 것을 보여줍니다. CNN의 경우 우리는 새로운 transformation layer를 도입함으로써 이를 얻을 수 있었고 학습된 equivarant transformation을 분석하여 representation의 invariance에 대한 특징을 찾고 특징을 나타낼수 있었습니다. 또한 이를 통해 invariance의 양을 정량화하고 deep model의 depth에 따라 어떻게 변화하느지를 보여줍니다.

세번째 주목한 특징은 equivalence로서 heterogeneous(서로 다른?) representation 으로 포착된 information이 실제로 같은지를 연구합니다. 특히 CNN은 non-convex optimiazation의 특징 때문에 같은 데이터로 부터 다른 파라미터를 학습 할 수 있으며 이는 수백만개의 redundant 파라미터를 포함 할 수 있습니다. 여기서의 문제는 이러한 학습된 다른 파라미터가 진짜인 것인지 필요없는 것인지에 대한 여부입니다. 이 질문에 답하기 위해서 서로 다른 네트워크에서의 모듈을 swap 할 수 잇는 stitching layer를 학습합니다. Equivalence는 이러한 stitching layer를 도입한 "Franken-CNN" 이 원래와 같은 결과를 얻으면 Equvalence 하다고 주장합니다.

논문의 나머지 부분은 다음과 같이 구성됩니다. 섹션 2에서는 equivariance, invariance, equivalence 를 경험적으로 학습할 수 있는 방법에 대해 기술합니다. 섹션 3.1, 3.2 에서는 
shallow, deep representation의 equivariance에 대해 기술하며 섹션 3.3 에서는 equivalence에 대해 기술합니다. 섹션 3.4 에서는 structured-output regression을 통해 equivariant representation의 실용적 방법에 대해 보여주며 섹션 3.4 에서는 우리의 발견들을 요약합니다.

<b>Related work.</b> invariant, equivariant feature를 설계하는 것은 컴퓨터 비전 에서 광범위 하게 연구되었습니다. 예를들어 유명한 접근 방법은 [12,13,15] 와 같은 equivarant (co-varaint) detector로 부터 invariant local descriptor 를 추출하는 것입니다. [20, 26] 과 같은 논문에서 저자들은 representation에 명시적으로 equivariance를 통합하는 것을 연구했습니다. [10] 과 같은 CNN 구조와 최신의 모델들에서는 레이어를 깊게 쌓음으로써 invariant가 증가한다고 생각합니다. 이것은 [22] 에서의 scattering transform에서 더 명시적으로 연구되었습니다.

이러한 예들에서 invariance는 주어진 구조에서 만족하거나 만족되어지지 않게 설계되어집니다. 이와 반대로 우리의 목적은 invariance를 학습하는 새로운 방법을 제안하는 것이 아니라 representation이 가질 수 있는 invariance, equivaraince와 같은 특징들을 알아내는 것입니다. 우리가 아는 한은 이러한 유형의 분석을 수행하는 것은 매우 제한되어져 있습니다. [8, 33] 에서는 몇몇의 특정한 image transformation에 대한 뉴럴 네트워크의 invariance를 연구했습니다. 우리는 해석적으로 분석하고 이들 특징들을 정량화 하는 첫번째 방법이라고 생각하며 다른 representation에 대한 equivalence를 연구한 첫번째 논문입니다.

### 2. Notable properties of representations

HOG, SIFT, CNN의 image representation은 인풋 이미지 x 를 representation vector φ(x) 로 매핑하는 어떠한 임의의 함수입니다. 이 섹션에서는 representation의 invariance, equivariance, equivalence에 대해 기술하며 이들을 경험적으로 볼 수 있는 알고리즘을 제공합니다.

<b>Equivariance.</b> 어떤 representation φ 가 임의의 인풋 이미지에 대한  transformation g 에 대해 equivaraint 하다는 것은 transformed image를 통해 얻은 representation과 원래의 image를 통해 얻은 representation에 transform 한 것이 같다면 equivariance 하다고 합니다. 수식적으로 equivariance는 다음과 같이 나타냅니다.

<img src="https://latex.codecogs.com/gif.latex?%5Cforall%20x%20%5Cin%20%5Cchi%20%3A%20%5Cphi%28gx%29%20%5Capprox%20M_g%5Cphi%28x%29%20%5C%2C%5C%2C%5C%2C%5C%2C%20%281%29%20%5Cnewline%20where%5C%2C%5C%2C%20exist%20%5C%2C%5C%2C%20M_g%20%3A%20%5Cmathbb%7BR%7D%5Ed%20%5Crightarrow%20%5Cmathbb%7BR%7D%5Ed" />

image transform g 에 해당하는 representation transform M_g 가 있어야만 위의 수식이 성립한다!!

M_g의 존재성에 대한 충분조건은 representation φ 가 invertible 해야 한다는 것입니다. 이 경우에는  <img src="https://latex.codecogs.com/gif.latex?M_g%20%3D%20%5Cphi%20%5Ccirc%20g%20%5Ccirc%20%5Cphi%20%5E%7B-1%7D" />
를 통해 얻을 수 있습니다. HOG 같은 경우에는 approximately invertible한 representation으로 알려져 있습니다. 그러므로 M_g가 존재 할 뿐만 아니라 근사하여 구할 수도 있습니다. 특히 M_g는 linear function과 같이 간단해야 합니다. 이러한 representation이 종종 linear classfier에 사용되거나 CNN의 경우 linear filter에 의해 처리되기 때문에 중요합니다. 게다가 임의의 인풋 이미지에 대해 동일한 mapping M_g 가 적용되는 것이 필요하기 때문에 representation의 본질적인 geometric properties가 포착됩니다. (뭔소리인지,,, 모르겠지만 최대한 간단한 M_g 예를들어 affine transform 같은 것들이 적용되어야 기하학적 변화에 대한 특성들을 볼 수 있다는 소리인가봄)

transformation function g는 임의의 함수이지만. 우리의 논문에서는 affine warp, flip image와 같은 변환에 초점을 맞춥니다.

<b>Invariance.</b> invariance는 equivariance의 특별한 경우로서 얻어진 M_g 가 매우 간단한 transformation인 경우(예를 들어 identity mapping)입니다. 컴퓨터 비전에서 invariant 한 특징은 다양한 task를 위해 필요한 특징으로써 종종 representation의 주요한 특징들로 여겨집니다. 예를들어 이미지안의 object category는 viewpoint change에 대해 invariant 해야 합니다. invariance를 체계적으로 연구함으로서 representation의 invariance가 어떻게, 어디서 달성되는지 명확히 하는 것이 가능합니다.

<b>Equivalence.</b> equi/invariance 는 인풋 이미지의 transformation이 representation에 어덯게 영향을 주는지를 보여줍니다. equivalence는 서로 다른 representation 간의 관계를 연구합니다. 서로 다른 representation φ, φ' 은 다음과 같은 mapping이 존재하면 equivalence 합니다.

<img src="https://latex.codecogs.com/gif.latex?%5Cforall%20x%20%3A%20%5Cphi%27%28x%29%20%5Capprox%20E_%7B%5Cphi%20%5Crightarrow%20%5Cphi%27%7D%5Cphi%28x%29" />

즉 두개의 representation이 인풋 이미지의 variant에 대해 같은 정도의.. 변화를 가진다.

φ 가 invertible 한다면 <img src="https://latex.codecogs.com/gif.latex?E_%7B%5Cphi%20%5Crightarrow%20%5Cphi%27%7D%20%3D%20%5Cphi%27%20%5Ccirc%20%5Cphi%5E%7B-1%7D" /> 는 존재하게 됩니다. 

<b>Example: equivariant HOG transformations</b> φ 를 HOG feature extractor로 가정해 봅시다. 이 경우에는 φ(x) 를 일종의 HxW vector filed의 D-dim feature vector(cell) 로 볼 수 있습니다. g를 일종의 vertical flipping으로 생각한다면 φ(x), φ(gx) 는 feature component의 permutation으로 정의 될 수 있습니다. 이러한 transform은 HOG cell들을 수평축으로 swap 하는 것이 되며 각 HOG cell 안에서는 gradient의 방향의 대각축으로 값들을 swap 하게됩니다. 그래서 mapping M_g는 일종의 permutation matrix의 꼴이 되게 됩니다. horizontal flip과 180 rotation은 동일하며 근사적으로 90 rotation 입니다.? HOG의 구현은 이러한 permutation 연산을 제공합니다.

<b>Example: translation equivariance in convolutional representations. </b> HOG, DSIFT(densely-computed SIFT), CNN은 convolutional representation의 하나의 종류이며 이들은 local, translation invariant operator(convolution?) 으로 얻어집니다. boundary, samplint effect를 제외하고 생각하면 임의의 convolutional representation은 equivariant합니다. 즉 feature에서의 translation과 인풋 이미지에서의 translation이 동일합니다. (컨벌루션 연산은 근본적으로 translation invariance하다.)


### 2.1. Learning properties with structured sparsity

equivariance, equivalence를 연구할때 transformation M_g, E 는 closed form으로 구하는게 불가능하며 data로 부터 추정해야만 합니다. 이 섹션에서는 이를 위한 몇가지 방법을 소개합니다. equirvariant transformation M_g 에 초점을 맞추지만 equivalence transformation E 도 매우 비슷합니다.

representation φ과 g라는 transformation이 주어진다면 목표는 수식 1을 만족하는 mapping M_g를 차즌ㄴ 것입니다. 가장 간단한 케이스는 M_g = (A_g, b_g) 인 affine transformation을 나타냅니다. 이 경우 φ(gx) ~~ A_gφ(x) + b_g 가 됩니다. 이러한 affine transform은 많이 제한적이지 않습니다. 위에서 보았던 M_g가 permutation 이라면 A_g를 permutation matrix로 구현 가능합니다. 

(A_g, b_g) 를 추정하는 것은 emprical rsit를 최소화 하는 문제로 풀 수 있습니다. data x sample이 주어지고 regularization term이 주어진다면 다음과 같이 나타낼 수 있습니다.

<img src="https://latex.codecogs.com/gif.latex?E%28A_g%2C%20b_g%29%20%3D%20%5Clambda%20R%28A_g%29%20&plus;%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20L%28%5Cphi%28gx_i%29%2C%20A_g%5Cphi%28x_i%29%20&plus;%20b_g%29%20%5C%2C%5C%2C%5C%2C%5C%2C%20%282%29" />

그냥 MSE loss + regularization term으로.. M_g 를 추정하겠다는 거네.

여기서 R는 regulariser 이며 L는 regression loss 이며 밑에서 언급되어집니다. 이 objective function은 equivalence 에도 적용 가능하며 φ(gx) 를 φ'(x) 로 대체하면 됩니다.















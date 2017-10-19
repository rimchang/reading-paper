# EXPLAINABLE ARTIFICIAL INTELLIGENCE: UNDERSTANDING, VISUALIZING AND INTERPRETING DEEP LEARNING MODELS

https://arxiv.org/pdf/1708.08296.pdf


### Abstract

큰 database와 최근의 딥러닝 방법론의 발전과 함께 AI 시스템의 성능이 인간의 성능을 넘어서는 일이 매우 많이 증가하고 있습니다. 이러한 발전은 image classfication, sentiment ananlysis, speech understanding, strategic game playing 등의 분야에서 보여지고 있습니다. 그러나 이들의 non-linear한 구조 때문에 이러한 높은 성능을 가진 machine learning, ai 모델들은 black-box 의 방법론으로 적용되고 있습니다. (예측을 위해 어떠한 과정을 거치는지에 대한 정보가 주어지지 않는 모델들). 이러한 투명성의 부재는 매우 큰 단점중에 하나입니다. 딥러닝 모델을 설명하고 해석하려는 분야는 최근들어 큰 주목을 받고 있습니다. 이 논문은 이 분야의 최근의 발전을 요약하고 ai를 더 해석가능하게 해야 하는 이유에 대해 기술합니다. 또한 이 논문은 딥러닝 모델을 설명하기 위한 두가지 접근방법을 설명합니다. 한가지 방법은 input의 변화에 따른 예측 결과의 민감도를 계산하는 것이고 또 다른 한가지 방법은 input variable 의 관점에서 의미 있도록 decision을 분해하는 것입니다. 이들 방법들은 3가지 classfication task로 평가됩니다.

### 1. Introduction

머신러닝, ai 분야는 지난 10년간 발전되어 왔습니다. 이러한 발전의 원동력은 svm의 발전과 최근의 딥러닝 방법론의 발전 덕분입니다. 또한 Imagenet, Sport1M 같은 큰 데이터셋과 GPU 를 통한 빠른 계산능력, Caffe, Tensorflow 같은 프레임워크의 높은 유연성 들이 이러한 성공의 원인입니다. 오늘날의 AI를 기반으로하는 머신러닝은 NLP, object detection, speech signal 등의 다양한 분야에서 탁월한 성능을 보이고 있습니다. 그 외로 최근의 AI 시스템은 바둑, 텍사스 홀덤 포커 같은 어려운 전략 게임에서 인간 플레이어를 능가했습니다. 이러러한 엄청난 AI 시스템의 성공, 특히 딥러닝 모델을 활용한. 들은 이러한 기술들의 혁명적인 성격을 보여주고 학계를 넘어서 큰 영향을 끼칠 것입니다. 또한 산업과 사회에 파괴적인 변화를 가져올 것입니다.

그러나, 이들 모델이 매우 높은 예측 accuracy를 도달했더라도 그들의 non-linear한 구조는 이들을 해석 불가능 하도록 만듭니다. (이들은 input data의 어떤 정보가 모델의 결정을 도달하게 만드는지에 대한 정보가 불분명합니다.) 그러므로 이들의 모델은 보통 일종의 black boxes로 언급이 됩니다. 알파고와 이세돌간의 두번째 경기에서 이러한 AI 시스템의 해석 불가능한 부분이 보여집니다. 알파고는 바둑 전문가가 다음과 같이 말할 정도로 기대하지 않았던 방법으로 말을 움직입니다.

“It’s not a human move. I’ve never seen a
human play this move.” (Fan Hui, 2016).

이 경기 동안 AI 시스템이 왜 이러한 움직임을 선택했는지는 불분명합니다. 이는 알파고만 알 수 있는 알파고가 게임을 이기기 위한 결정이였습니다. 이 경우에서 알파고가 black box라는 특징은 상관이 별로 없습니다. 하지만 다양한 분야에서 AI 시스템의 decision process를 이해하기 못하고 타당화 하지 못한다는 것은 매우 큰 단점입니다. 예를들어 medical diagnosis의 경우 black-box 시스템의 예측을 무작정 믿는 것은 무책임 합니다. 대신 매번 decision에 도달할 때마다 전문가에 의해 적절한 타당화를 할 수 있도록 만드는 것이 필요합니다. 또한 self-driving car의 경우 한번의 잘못된 예측은 매우 큰 비용을 초래할 수 있습니다. 올바른 feature에서의 모델의 신뢰성은 보장되어야 합니다. explainable and human interpretable AI 모델의 사용은 이러한 보장을 제공하기 위한 조건입니다. 더 많은 설명가능한 AI에 대한 논의는 섹션 2에 있습니다.

당연하게도 이러한 블랙박스 모델의 "opening" 기술의 발전은 커뮤니티에서 많은 주목을 받고 있습니다. 이들은 그 모델이 배운것이 어떤 것인지를 이해하는 더 나은 방법을 포함하고 있습니다.

[Visualizing higher-layer features of a deep network(12),   
Deep residual learning for image recognition(14),  
  Multifaceted
feature visualization(29)] 는 모델의 representation이 무엇인지를 이해하려고 하며  

[Interpreting individual classifications of hierarchical networks(19),  
 Deep inside convolutional networks: Visualising image classification models and saliency maps(35),   
 Visualizing and understanding convolutional networks(39),  
  On pixel-wise explanations for ¨
non-linear classifier decisions by layer-wise relevance propagation(5),  
 Explaining nonlinear classification decisionswith deep taylor decomposition(26)] 은 개별 예측을 설명하기 위한 방법을 언급합니다.  
 
[Methods for interpreting and understanding deep neural networks(27)] 은 위의 두 카테고리에서의 방법론에 대한 튜토리얼 페이퍼를 제공합니다.

설명가능함은 svm 에서도 매우 중요하며 뉴럴네트워크가 아닌 다른 방법론에서도 매우 중요한 이슈입니다.


이 논문의 가장 중요한 목표는 머신러닝과, ai에서의 설명가능함의 필요성에 대한 인식을 조성하는 것입니다. 이는 섹션2 를 통해 기술했습니다. 섹션3 이후로는 2가지 SA, LRP 라는 두가지 최근 기술들을 소개합니다. 이는 input variable의 관점으로 AI 모데의 개별 예측을 설명합니다. 어떻게 객관적으로 이러한 설명의 질을 측정하는 것은 섹션4에 기술되어 있습니다. 섹션5 에는 이미지, 텍스트, 비디오 classfication 실험에 대한 내용이 기술되어져 있고 섹션6을 통해 미래의 연구와 결론에 대해 기술합니다.

### 2. Why do we need explainable AI

다른 사람의 의사결정에 숨겨진 이유를 설명할 수 잇는 능력은 인간 지능에서의 가장 중요한 양상입니다. 이는 사회적 상호작용에서 중요할 뿐만 아니라 (자신의 의도와 생각을 결코 밝히지 않는 사람은 아마 "이상한 동료"라고 여겨질 것이다.) 교육적 환경에서도 중요합니다 (학생들은 선생님들의 이유를 추론하려고 함?). 또한 어떤 의사결정을 설명하려는 것은 사람들 간의 신뢰 관계를 구성하는데 사전조건이 됩니다. (의사가 그들의 처방의 이유를 환자들에게 설명함)
 
 이러한 사회적 양상들이 AI 시스템에서는 덜 중요함에도 설명가능한 AI에 대한 많은 의견들이 있습니다. 그것들 중 중요한 것들이 기술되엇습니다.
 
 - <b>Verification of the system</b> :  이전에 언급했듯이 많은 응용들은 black-box 시스템의 결과를 바로 신뢰해서는 안됩니다. 예를들어 health care 분야에서는 해석가능하고 의료 전문가들에게 타당화 가능한 모델의 사용이 절대적으로 중요합니다. [Intelligible models for healthcare(7)] 의 저자는 헬스케어 분야에서 폐렴의 위험도를 예측하도록 훈련된 AI 시스템이 완전히 틀린 결론에 도달하는 예를 보여줍니다. black-box 방법으로써의 이 모델의 적용은 폐렴으로 인한 죽음을 줄이지 않고 오히려 늘리게 될 것입니다. 간단하게 설명하면 모델이 심장질환을 앓고 있는 천식환자가 건강한 사람보다 폐렴으로 죽을 가능성이 낮다고 배웠습니다. 의사는 천식과 심장질환이 prognosis for recovery(사후 회복?) 에 부정적인 영향을 끼친다는 것을 인식 할 수 있습니다. 하지만 AI 모델은 천식과 폐렴에 대해 아무것도 알지 못합니다. 단지 데이터로 부터 추론 할 뿐입니다. 이 예제에서 데이터는 bias 되어져 있었고 그 이유는 건강한 사람들과 달리 천식, 심장질환 환자들은 의료 감독 서비스를 받는 경우가 많기 때문이였습니다. 이러한 감독과 이들 환자의 민감도의 증가 때문에 이 그룹은 폐렴으로 인한 죽음의 가능성이 낮았습니다. 그러나 이러한 상관관계는 인과 관계를 가지지 않는 것이며 이는 폐렴 치료를 위한 의사결정의 기저로 사용되어선 안됩니다.
 
  - <b>Improvement of the system </b> : AI 시스템을 향상시키는 첫번째 단계는 그의 약점을 이해하는 것입니다. 분명하게도 해석 가능한 모델보다 black-box model의 약점을 분석하는 것은 더 어렵습니다. 또한 모델 혹은 데이터셋에서의 biases를 검출하는 것은 설명가능한 모델이 더 쉽습니다. 또한 모델의 해석가능성은 다른 모델과 구조들끼리 비교할 때도 유용합니다. 예를들어 [20, 2, 3] 의 저자는 같은 classfication 성능을 보이는 모델들이지만 의사결정에 사용되는 feature들에는 매우 큰 차이가 있다는 것을 보였습니다. 이들의 작업은 적절한 모델은 설명가능함을 욕한 다는 것을 보여줍니다. 어떤 사람들은 우리의 모델이 하는 일을 이해하는것 (때때로 그들이 실패하는 이유) 를 이해하는 것이 모델의 성능을 향상시키는 것이 쉽도록 해줄 것이라고 주장합니다.
  
   - <b>Learning from the system</b> : 오늘날의 AI 시스템이 수억개의 데이터로 학습되기 때문에 인간이 접근 불가능한 데이터의 패턴을 관찰 할 수 있습니다. 인간은 학습 할 수 있는 example의 갯수가 제한되어져 있습니다. 설명 가능한 AI 시스템을 사용한다면 우리는 이러한 인공지능으로 부터 distilled 지식을 뽑아 내고 이를 통해 새로운 인사이트를 얻을 수 잇습니다. 이러한 AI 시스템으로 부터의 지식을 뽑아낸 하나의 예가 위에서 언급한 Fan Hui 가 한 말입니다. AI 시스템은 이전의 프로 바둑 플레이어에서 얻을 수 없었던 바둑을 두는 새로운 전략을 발견합니다. Information extraction의 다른 분야는 바로 과학분야입니다. 간단히 말해서 물리학, 화학, 생명과학 분야는 black-box 모델로부터 단순히 예측을 하는 것 보다는 자연의 숨겨진 법칙을 발견하는 것에 관심을 둡니다. 그래서 제한된 해석 가능한 모델만이 이 분야에 사용이 됩니다.
   
 - <b>Compliance to legislation</b> : AI 시스템은 우리의 생활에 더 많은 영향을 끼치고 있습니다. 이를 통해 시스템이 잘못된 결정을 내릴 때의 법적 측면이 최근에 더 많은 관심을 받았습니다. black-box 모델에 의존하는 경우 이러한 법적 문제에 적절한 답을 찾을 수 없기 때문에 미래의 AI 시스템은 더 설명 가능해져야 합니다. 이러한 규제가 AI가 더 설명가능해져야 한다는 다른 예중 하나는 개인의 권리입니다. 사람은 AI 시스템의 결정으로 부터 즉각적으로 영향을 받을 수 있습니다. 예를 들어 머신러닝 알고리즘에 의해 은행으로 부터 대출을 거절당한 사람은 어떤 이유 때문에 대출이 거절 당했는지를 알고 싶어 할 수 있습니다. 오직 설명 가능한 AI 시스템만이 이러한 정보를 제공 가능합니다. 이들 우려는 EU가 "설명 할 권리" 라는 새로운 규제를 만들게 되었고 사용자는 사니이나 자신에 대해 결정된 알고리즘의 의사결정에 대한 설명을 요청 할 수 있습니다.
 
 
 이러한 예들은 설명가능함이 학문적 목적 뿐만 아니라 미래의 AI 시스템에 중요한 역할을 할 것을 보여줍니다.
 
 ### 3. METHODS FOR VISUALIZING, INTERPRETING AND EXPLAINING DEEP LEARNING MODELS
 
 이 섹션에서는 딥러닝 모델의 예측을 설명하는 두가지 유명한 기술을 소개합니다. 설명의 과정은 피규어 1에 요약되어져 있습니다. 먼저 시스템은 인풋 이미지를 올바르게 "rooster" 라고 예측합니다. 그 다음 input variable의 관점으로 예측을 설명하기 위한 explanation method가 적용됩니다. 이 explanation precess의 결과는 예측을 위한 각 픽셀의 중요도를 시각화한 heatmap 입니다. 이 예제에서는 rooster의 빨간 깃과 피부가 AI 시스템의 의사결정의 기저가 됩니다.
 
 ### 3.1 Sensitivity analysis
 
 SA 라고 알려진 첫번째 방법은 모델의 locally evaluated gradient를 기반으로 예측을 설명합니다. 수학적으로 SA는 input variable i의 중요도를 정량화 합니다. (input image를 backprop으로 업데이트)
 
 이 measure는 출력이 가장 민감한 것들이 가장 중요한 input feature일 것이라는 가정입니다. 다음 subsection에서 설명할 접근방법과 반대로 SA는 f(x) 자체를 설명하지 않고 그것의 변화정도를 설명합니다. 다음의 예제과 왜 함수의 민감도를 측정하는 것이 AI 시스템의 예측을 설명하는 것의 suboptiaml이 될 수 있는지를 설명합니다.
 
 SA를 통해 계산된 heatmap은 예측된 클래스로 이미지가 보이기 위해서 어떤 픽셀이 변해야 할지를 나타냅니다. 예를들어 피규어 1의 이미지를 살펴보면 수탉의 중첩된 부분으로 노란색 꽃이 존재합니다. 이 노란색 꽃의 픽셀을 어떤 특정한 방법으로 재구성 함으로써 classfication 스코어를 올리도록 할 수 있습니다. 주의할 것은 이러한 heatmap이 수탉을 예측하는데 주요한 픽셀이 아닐 수도 있다는 것입니다. (변화량즉 그라디언트를 보기 때문에?) 노란색 꽃의 존재가 이미지 안에서의 수탉의 존재를 확실하게 나타내지 않는 것은 아닙니다.(??? 뭔소리지) 이 이유는 SA가 섹션 5에서 기술된 정량적 평가 방법에서 잘 수행되지 않습니다. 
 
 ### 3.2 Layer-wise relevance propagation
 
 다음은 AI 시스템의 예측을 input variable의 관점에서 분해하기 위한 일반적인 프레임워크를 제공합니다. (feed forward NN, back-word model, LSTM, fisher vectore classfier). SA 방법과 반대로 이들 방법은 state of maximum uncertainty와 관련되어 예측을 설명합니다. (수탉을 예측하는데 가장 주요한 픽셀이 무엇인지를 찾습니다.) [26] 과 같은 최근의 연구는 수학적으로 일반적인 function analysis인 Taylor decompositoin을 소개합니다.
 
 LRP 라는 최근의 기술은 classfier의 의사결정을 decomposition을 통해 설명합니다. 수학적으로 f(x) 의 backward를 재분배 하는데 각 input variable 에 relevance score를 할당합니다. 이 재분배 과정의 특징은 relevance conservation 으로 언급됩니다.
 
 이 특성은 재분배 과정의 각 스탭마다 total amount of relevance(prediction f(x)) 가 보존된다는 것입니다. 재분배 과정동안 relevance가 추가되거나 제거되지 않습니다. relevance score R_i 는 각 input variable 이 prediction에 얼마나 기여하는지를 나타냅니다. 그래서 SA와 반대로 LRP는 f(x) 를 완전히 분해합니다.
 
 다음은 feed-forward neural network 를 위한 LRP 재분배 과정을 설명합니다. [5, 4, 20] 등에서 제안되었습니다. x_j 를 layer l에서 활성화되 뉴런이라고 했을때 R_k 는 l+1 layer의 k 번째 뉴런과 관련된 W_jk 와 관련된 relevance score 입니다. 간단한 LRP 재분배 방법은 다음과 같습니다.
 
 epsilon term은 zero division을 방지하기 위한 텀입니다. 직관적으로 이 relevance 재분배 규칙은 layer l+1 에서 layer l의 각 뉴런으로 재분배하는데 두가지 기준이 있습니다. (1) x_j의 뉴런의 활성화 (더 활성화된 뉴런이 더 많은 relevance를 공유받는다) (2) W_jk 의 연결정도의 강도 (더 relevance flow가 더 주요한 연결이다?) . relevance 보존이 epsilon=0 일때 성립한다는 것에 주목해라
 
 alpha-beta 규칙은 다른 대안적인 재분배 규칙입니다.
 
 여기서 ()+, ()- 은 postive, negative parts를 나타냅니다. relevance conservation은 alpha-beta = 1 의 조건하에 성립됩니다. alpha=1인 특별한 경우 [26] 의 저자가 이러한 재분배 규칙이 deep tayloe decompostion과 일치 한다는 것을 보엿습니다.(Relu 뉴런을 사용할때!)
 
 
 ### 4. Evaluating the quality of explanations
 
 SA, LRP 같은 다른 방법에서 생성된 Heatmap을 비교하기 위해 설명에 대한 질을 측정하는 objective measure가 필요합니다. [31]의 저자는 perturbation analysis에 기반한 measure를 제안합니다. 이 방법은 다음과 같은 3가지 아이디어에 기반합니다.
 
 - prediction에 매우 중요한 input variable의 perturbation은 중요하지 않은 input variable보다 prediction socre의 더 많은 감소를 야기합니다.
 
 - SA, LRP 같은 방법들은 모든 input variable에 대한 score를 제공합니다. 그래서 input variable은 이러한 relevance score에 따라 정렬을 할 수 있습니다.
 
 - 가장 관련있는 것들부터 시작하여 반복적으로 input variable을 혼란스럽게 할 수 있습니다. 그리고 매번 교란 단계후에 예측 score를 추적합니다. prediction socre의 평균적 하락 혹은 prediction accuracy의 하락이 설명의 질의 object measure로 사용할 수 있습니다. 이 이뉴는 큰 감소는 explanation method가 매우 진짜 중요한 input variable을 찾았다는 것이 성공적이라는 것을 나타냅니다. 
 
 즉.. method에 의해 가장 중요하다고 판단된 input variable부터 noise를 줘가면서.. accuracy가 얼마나 떨어지는지를 봄!! 이때 많이 떨어지면... 진짜 중요한 varible을 찾앗다고 생각!
 
 다음의 평가는 모델에 상관없이 똑같은 교란을 주었습니다. 이는 biases를 막기 위해서입니다. (input value를 유니폼 분포에서 샘플링해서 대체함)
 
 ### 5. Experimental evaluation
 
 ### 5.1 Image classfication
 
 처음은 구글르넷 모델에 대해 실험을 했습니다. 피규어2의 (A) 는 데이터셋에서의 두가지 이미지를 나타냅니다. 이는 각각 "volcano", "coffe cup" 으로 올바르게 예측되었습니다. heatmap 시각화는 SA, LRP에 얻어진 것을 나타냅니다. "coffee cup" 에 대한 LRP heatmap 이미지는 모델이 컵의 타원형 모양을 "coffee cup"과 관련된 feature로 찾았다는 것을 보여줍니다. 다른 예제로 산의 특정 모양이 이미지안에 화산이 존재하다는 증거로 여겨진다는 것을 볼 수 있습니다. 
 
 SA heatmap은 LRP 에 의해 계산된 값보다 noiser 한 것을 볼 수 있습니다. 그리고 R_i 가 백그라운드에 할당된 것을 볼 수 있습니다. ("vocano" 이미지에서 백그라운드는 카테고리와 상관이 없는데 빨간 점이 있는걸 볼 수 있다.) LRP와는 반대로 SA는 각 픽셀이 prediction에 얼마나 기여하는지를 나타내지 않습니다. 대신 classfier가 input의 변화에 얼마나 민감한 정도를 나타냅니다. 그러므로 LRP가 SA보다 더 주관적으로 나은 설명을 제공합니다. (사람의 눈으로 보고 평가했잖아)
 
 피규어2의 밑의 부분에서 (A) 는 섹션 4에서 소개된 perturbation analysis 의 결과를 보여줍니다. y축은 ILSVRC2012 데이터셋의 5040 이미지의 평균 상대적인 prediction score를 보여줍니다. (0.8이라는 값의 의미는 원래 score가 20%가 떨어졌다는 뜻) 매번 perturbation 단계마다 이미지의 9x9 패치(SA, LRP 에 의해 선택된) 이 유니폼 분포에서 샘플링된 값으로 대체됩니다. LRP가 SA보다 더 빠르게 감소하므로 LRP가 SA보다 객관적으로 나은 설명을 제공한다고 할 수 있습니다. (evaluation 방법에 의해 평가함.)
 
 
 ### 5.2 Text document classfication
 
 이 실험해서는 CNN에 기반한 word-embedding이 text document를 판별하기 위해 학습되었습니다. 피규어 2에서의 (B) 는 SA, LRP heatmap을 보여줍니다 (각 word에 할당된 relevance score R_i). document위에 써있듯이 이는 sci.med 로 판별되었습니다(text가 medical topic으로 분류됨) SA, LRP 두 방법 모두 "sickness", 'body', "discomfort" 같은 단어를 classfication decsion의 기저로 찾았습니다. SA와는 반대로 LRP는 단어를 positive(red), negative(blue) 로 구별합니다. (sci.med 로 결정하는데 도움이 되는 워드와 오히려 역효과를 내는 word). LRP heatmap의 경우 몇가지 contradicts word가 있음에도 불구하고 올바르게 sci.med로 판별 한 것을 볼 수 있습니다. SA 방법은 postive, negative 를 구별하지 않습니다. 피규어의 밑의 부분은 정량적 평가를 나타냅니다. y축은 상대적인 accuracy의 감소를 나타내고 4154개의 document를 평균 낸 것입니다. 각 perturbation step 마다 가장 중요한 단어가 0으로 제거 되었습니다. 또한 이 결과는 LRP가 정량적으로 더 informative heatmap을 제공한다는 것을 보여줍니다. 
 
 
  ### 5.2 Human action recognition in videos
  
  마지막 예제는 Fisher vector/ SVM classfier에 대한 것입니다. 이들은 압축된 비디오로 부터 human action을 예측하도록 학습되어졌습니다. 계산적 비용을 줄이기 위해 classfier는 block-wise motion vectore 로 학습이 되어졌고 evaluation은 HMDB51 데이터셋에 수행되었습니다. 피규어2는 LRP heatmap이 비디오 샘플의 프레임의 다섯개 샘플에 대한 계산을 보여줍니다. 이 비디오는 "sit-up" 으로 올바르게 분류되었습니다. 하나 볼 수 있는 것은 모델이 사람의 몸의 위쪽의 block에 대해 집중하는 것입니다. 이는 매우 말이 되는데 이 비디오 프레임은 "sit-up" 이라는 행동의 모션을 보여주기 때문이고 이는 몸이 upward and downward 움직임이 나타내기 때문입니다.
  
  피규어2에서의 밑의 (C) 이미지는 프레임별로 relevance의 분포를 나타냅니다. 여기서 볼 수 있는 것은 사람이 upward and downward 움직임을 수행할대 relevance score가 커진다는 것입니다. 그래서 LRP heatmap이 비디오 프레임에서의 행동과 관려된 location을 보여줄 뿐만 아니라 비디오 시퀀스에서 가장 관련된 time point를 찾 을 수 있다는 것을 보여줍니다.
  
  
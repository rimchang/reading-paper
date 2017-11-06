# Going Deeper into Action Recognition: A Survey

### Abstract

visual data에서 인간의 행동을 이해하는 것은 object recognition, human dynamics, domain adaption, semantic segmentation 을 포함한 연구 영역의 발전과 연관된다. 지난 10년간 인간의 행동을 분석하는 것은 제한이 많은 이전의 구조에서 부터 수백만개의 비디오및 거의 모든 일상 활동에 적용 될 수 있는 요즘의 고급 솔루션에 까지 진화되어져 왔다. video surveillance(비디오 보안 감시) 에서부터 human-computer interaction 까지 광범위한 어플리케이션에서 action recognition의 성취가 빠르게 달성되었다. 결국 한때 좋은 성능을 가지던 모델들이 사라져 버렸다. 이러한 것은 우리에게 동기를 부여했고 action recognition의 포괄적 리뷰를 제공하게 되었다. 우리는 손으로 디자인한 representation을 사용하는 이전의 방법에서 시작하여 딥러닝 기반의 방식으로 이동한다. 우리는 이 페이퍼에서 객관적인 괁머을 유지하려고 하며 개선할 수 있는 사항 뿐만 아니라 새로운 질문을 제기하고 독자들을 위한 새로운 연구방향을 제시하기를 바란다.

### Introduction

너의 스마트기기와 로봇 어시스턴스가 어떠한 개입 없이 무엇인가 도와줄 수 있는 수준으로 너의 행동을 이해하고 인식 할 수 있는것을 상상해보라. 우리는 아직 그정도의 기술을 가지지 못하지만 우리의 기술적 진보는 명백히 그러한 것을 향하고 있다. 이 논문에서는 우리는 현재 존재하는 action recognition에 대한 연구와 우리의 행동을 인식하는 스마트한 알고리즘을 개발하기 위해 우리에게 무엇이 필요한지를 말할 것이다.

### But first, what is an action?

인간의 motion은 팔다리의 간단한 움직임에서부터 복잡한 관절의 그룹, 몸의 움직임으로 확장된다. 예를들어 축구에서의 다리의 움직인은 매우 간단하다. 헤드슛을 날리기위한 점프는 다리, 팔, 머리 , 전체 몸에 대한 움직임의 집합이 될 것이다. 이러한 직관적이고 단순한 개념에서 불구하고 action이라는 용어는 정의하기 매우 어려운 것처럼 보인다. 아래에는 몇가지 문헌에서 발췌한 예제가 나와있습니다.

- <b>Moeslund and Granum(2006); Poppe(2010) </b>: 초기적인 action 정의는 "limb 수준에서 기술될 수 있는 움직임" 으로 정의됩니다. 따라서 이러한 action의 정의는 다양한 움직임, 간단하고 원시적인 것에서 부터 주기적인 신체의 움직임 까지를 정의할 수 있습니다. "activity" 라는 용어는 "a number of subsequent actions" 로 사용되며 복잡한 움직임을 나타냅니다. 예를들어, 왼쪽 다리의 앞으로가는 것은 달리기라는 행동의 기본 요소입니다. 허들을 넘는 것은 달리기와, 점프로 시작되는 하나의 activity 입니다.

    - action : limb(사지) 수준에서 기술될 수 있는 움직임 ex) 왼쪽 다리의 앞으로가는 움직임
    - activity : a number of subsequent actions(몇몇개의 action의 sequnece??) ex) 허들을 넘는 것은 달리기, 점프로 구성되어져 있다. 

- <b>Turaga et al.(2008) </b>: action을 "일반적으로 매우 짧은 시간 동안 지속되는 한 사람에 의해 수행되는 간단한 동작의 패턴" 으로 정의합니다. 이들은 activity 를 "서로 다른 방식으로 상호작용하는 몇몇 사람들에 의해 수행되는 복잡한 action의 sequence" 라고 정의합니다. 예를들어 actions은 걷기, 수영하기와 같은 것이며 activities는 두 사람이 악수를 하거나 축구를 하고 있는 팀들입니다.

    - action : 일반적으로 매우 짧은 시간 동안 지속되는 한 사람에 의해 수행되는 간단한 동작의 패턴 ex) 걷기, 수영하기
    - activity : 서로 다른 방식으로 상호작용하는 몇몇 사람들에 의해 수행되는 복잡한 action의 sequence ex) 악수를 하는 두사람, 축구하는 팀
    
- <b>Chaaraoui et al (2012)</b> : 이들은 인간의 행동의 맥락에서의 행동의 계층적 구조를 제안한다. 이러한 나눔은 의미적 정보와 시간적 granularity(세분성?) 을 기반으로 하고 motion과 activity 의 중간수준에서의 action을 고려한다. action은 몇분 동안 지속될 수 있는 원시적인 동작으로 정의됩니다. 

    - action : 몇분 동안 지속될 수 있는 원시적 동작 ex) sitting, walking
    - motion < action < activity 의 계층적 구조를 제안.
    
- <b>Wang et al. (2016)</b> : action이 환경에 가져올 수 있는 변화, 변화에 행동의 진정한 의미가 있다고 제안합니다. 
 
    - action : action으로 부터 일어나는 환경의 변화에 진정한 의미가 있다고 생각
    
#### Action define in survey

옥스포드 사전에서는 action이란 "통상적으로 어떠한 목표를 성취하기 위해 수행하는 어떠한 process, fact"로 정의합니다. activity는 "사람이나 그룹이 하는 일" 을 뜻합니다. 우리는 action에 대한 통합적인 정의를 제공합니다. "Action이란 어떠한 의미를 가지는 가장 기초적인 surrounding interaction 입니다."

상호작용과 관련된 의미를 category of action 이라고 합니다. 일반적으로 인간의 행동은 다양한 물리적 형태가 가능합니다. 우리의 정의에서 상호작용이라는 용어는 환경에 영향을 미칠 수도 있고 미치지 않을 수도 있는 주변에 대한 상대적인 motion으로 이해 될 수 있습니다. 어떠한 상황에서는 의미 있는 해석(털을 닦는 행동)을 얻기 위해 특정 대상을 "surrounding" 과 연관 시켜야 할 필요가 있습니다. 이는 Wang et al.의 정의와 일치하고 여기서 action은 환경에 가져오는 변화로 정의됩니다.

예를 들어 그림1의 motion의 sequence를 봅시다. 먼저 한 선수가 달림으로써 원시적인 다리의 motion이 수행됩니다. 비록 이러한 움직임이 주의와 상대적인 움직임이라고 해도 우리는 그것에 의미를 부여하기 힘듭니다. (상대적인 움직임이라는 말을 매우 강조하는 듯 하다.) 반면에 달리기의 결과로 얻은 limbs motion의 집합은 의미를 가집니다. 이것은 가장 기초적이고 의미있는 motion이고 우리는 "달리기" 라는 행동으로 고려합니다. 비슷하게 선수가 차는 행동, 골키퍼가 점프하는 행동은 구별가능한 "kicking", "jumping" 이라는 행동입니다. (잘 이해가 안간다.. 달리기로 인해 발생하는 상대적인 motion으로는 달리기라고 말하기 힘들지만.. limbs의 집합으로는 달리기라고 정의할 수 있다는 뜻인가?)

- 이 논문에서의 정의
    - action : 어떠한 의미를 가지는 가장 기초적인 인간과 주변과의 상호작용 이 상호작용이 환경에 변화를 일으킬수도, 아닐 수도 있다.
    
    
### Every survey paper has a taxonomy

일반적인 taxonomy는 우리를 갈라놓습니다.(다른 정의는 우리를 갈라놓는다는 뜻인가?) 대신에 우리는 솔루션들을 기반으로 그룹화 할 것이며 독자들은 끝에 이를 이해할 것입니다. 우리는 다양한 아키텍쳐와 교육방법들을 논의하고 딥러닝을 위해 별도의 세션을 구분합니다. 동시에 우리는 비디오 representation 방법을 나열합니다. 우리는 taxonomy의 이중성을 고려하고 이는 두 카테고리(딥러닝과 전통적 방법?)의 구성요소를 강조하는데 유용합니다.

### Why shoud we learn more about action recognition?

motion, action을 분석하는 것은 오랜 역사를 가지고 있으며 심리학, 생물학, 컴퓨터과학을 포함한 다양한 분야에서 매력적입니다. 이러한 매력을 볼 수있는 하나의 예는 500BC의 Zeno's dichotomy paradox 로 돌아갑니다. 엔지니어링 관점에서 볼때 action recognition은 video surveillance에서 부터 human-computer interaction, retail analytics, user interface desing, learning for robot, web-video search 등의 어플리케이션으로 확장됩니다. 많은 논문들은 "메뉴얼화된 action, motion 데이터의 분석은 불가능하다" 라고 강조합니다.

### 1. Wherer to start from?

80년대 초의 엄청난 생각을 인용하며 시작합니다. "첫째로, 뇌에서의 shape information을 나타낼 수 있는 symbolic system이 필요합니다. 두번째로 뇌는 이미지로 부터 이러한 정보를 이끌어 낼 수 있는 processor를 포함하고 있습니다". action recognition의 맥락에서는 좋은 representation은 반드시 "연산하기 쉽다", "충분히 큰 action class에 대한 설명을 제공해야 한다", "두 행동간의 유사성을 반영해야 한다", "다양한 변화에 robust 해야 한다."
 
<b> good representation in action recognition</b>

- be easy to compute
- provide description
for a sufficiently large class of actions
- reflect the similarity between two like actions
- be
robust to various variations (e.g., view-point, illumination)

action recognition을 위한 처음의 작업은 action을 기술하기 위해 3D 모델을 사용하는 것입니다. 하나의 예제는 인간의 행동을 이해하고 설명하기 위해 (Hogg) 에의해 소개된 계층적 WALKER 모델입니다. 다른 예제는 connected cylinder를 사용해 limb connection을 모델링한 보행자 recognition입니다. <span style="color:RED">일반적으로 비디오에서 정확한 3D 모델을 구축하는 것은 매우 어렵고 비용이 많이듭니다. 그래서 많은 솔루션들이 3D 모델을 피하고 Holistic, local representation을 취합니다.</span>

- <b>Holistic representations</b> : 인간의 몸구조, 모양, 움직임으로 부터의 global representation에 기반한 action recognition
 
- <b> Local representation</b> : local feature에 기반한 action recognition

### 전통적인 방법은 일단 패스.

### 1.1 Holistic Representation

### 2. Local Representation based approaches

### 3. Deep Architecture for Action Recognition

딥러닝 구조 덕분에 많은 task가 획기적으로 발전하는 것을 볼 수 있습니다. CNN과 같은 딥러닝은 이미지 content를 학습하는데 있어 하나의 방법이 되었습니다. 일반적으로 학습이란 데이터로 부터 복잡한 decision function을 만드는 것입니다. 딥러닝 구조에서는 이를 여러 layer의 non-linear 연산을 겹침으로써 작업을 수행합니다. 딥러닝의 파라미터를 찾는 것은 decision surface의 non-convexcity를 고려할때 매우 어려운 일입니다. 그라디언트 디센트 방법에 기반한 학습 알고리즘은 매우 많은 양의 데이터와 새로운 하드웨어를 통해 가능해 졌습니다.

이 섹션의 의도는 비디오로 부터 action을 배우는 문제를 다루는 다양한 딥러닝 모델을 논의하는 것입니다. taxonomical 관점으로 볼때 4가지 범주로 나눌 수 있습니다.

- Spatiotemporal networks
- Multiple stream networks
- Deep generative networks
- Temporal coherency networks

아래에서 각 카테고리에 대해 자세히 기술하고 연구 문제와 가능한 향상에 대한 정보를 제공합니다.

### 3.1 Spatiotemporal Network

CNN 구조는 네트워크의 search space를 줄이기 위해 이미지의 구조를 효과적으로 사용합니다. 이는 pooling, weight sharing을 통해 수행됩니다. pooling, weight sharing은 scale, spatial variations의 robust 하게 만들어 줍니다. CNN 구조에서 배운 첫번재 필터를 살펴보면 매우 낮은 레벨의 feature를 볼 수 있습니다. 위의 레이어는 보다 높은 의미적 레벨의 필터가 나타납니다. 이것은 CNN을 일반적인 feature extractor로 사용하는 것으로 확장합니다.

딥러닝을 사용한 action recognition의 직접적인 접근은 시간적 정보를 가진 conv 연산을 수행하는 것입니다. 이를 하기 위해 3d-conv net이 소개되었습니다. 3D-conv 네트워크는 3d 커널을 사용하며 (시간축으로 확장된 커널) spatial, temporal 차원으로 부터 feature를 추루합니다. 그래서 3d-conv는 인접한 프레임의 motion과 spatio-temporal 정보를 가져올 수 있다고 기대됩니다. 실무적으로는 학습을 용이하게 하기위해 opical-flow와 같은 추가적 정보를 제공하는 것이 중요합니다. 경험적으로 3D-conv 네트워크는 2D-conv 네트워크와 큰 성능차이가 있습니다.

일반적으로, 3D-conv 네트워크는 매우 엄격한 시간적 구조를 가집니다. 네트워크는 미리 정의된 수의 frame을 인풋(7 frame 등의)으로 받습니다. spatial 차원을 고정되게 쓰는 것은 방어의 여지가 있습니다.(spatial pooling은 scale, location의 robust를 증가시켜 준다) <span style="color:RED">하지만 시간 domain에서 고정된 수를 받는 것의 이유는 불분명 합니다. 다른 action들간에 다른 speed, span을 가지기 때문에 고정된 temporal span을 고르는 것은 불분명 합니다.</span>

시공간적 정보가 cnn에 어떻게 넣어져야 하는 것에 답으로 다양한 융합 방법이 제안되어져 왔습니다. Ng et al. 은 temporal pooling을 연구했고 max-pooling이 바람직하다는 결론을 내렸습니다. Karpathy 는 slow fusion의 개념을 제안했고 이는 cnn의 시간적 정보의 반영을 증가시킵니다. 

  <img src="https://www.frontiersin.org/files/Articles/160288/frobt-02-00028-HTML/image_m/frobt-02-00028-g012.jpg" />

slow fushion 방법에서는 cnn은 비디오의 몇몇의 연속적인 부분을 받아들이고 동일한 레이어를 통해 처리해 시간적 정보를 반영한 결과를 출력합니다. 이러한 반응은 fc-layer를 통해 video-descriptor를 생성합니다.

다른 형태는 early fusion 입니다 (ex> 3d-conv) 이는 인접한 프레임을 인풋으로 받습니다. late fusion은 프레임-wise feature들이 마지막 레이어에서 합쳐집니다. 


  <img src="http://slideplayer.com/6193197/18/images/63/Next+Milestone+in+Action+Recognition.jpg" />
  
카파시는 또한 두개의 분리된 네트워크를 사용하여 multiresolution 접근을 하였습니다. accuracy를 높일 수 있을 뿐만 아니라 학습할 파라미터 수를 줄였습니다. 이는 (fovea, context stream) 이 더 작은 인풋을 받도록 합니다. fovea stream은 프레임의 중앙 지역을 받아 많은 비디오에 존재하는 camera bias(관심이 되는 대상이 주로 중앙 지역에 존재함)를 활용합니다. 

 <img src="https://image.slidesharecdn.com/13temporalactionlocalizationinuntrimmedvideosviamulti-stagecnns1-160614103500/95/temporal-action-localization-in-untrimmed-videos-via-multi-stage-cnns-10-638.jpg?cb=1465900715" />

일반적인 이미지 descriptor 로의 VGG, Decaf 네트워크의 사용과 비슷하게 Tran et al은 3D conv 네트워크를 기반으로한 일반적 비디오 descriptor를 찾는 것을 시도합니다. 이러한 feature 추출 네트워크는 Sport-1m에서 학습되었고 경험적으로 저자들은 3x3x3 필터를 모든 레이어에서 사용하는 것은 필터의 시간축 차원을 바꾸는 것보다 효과적이였습니다. 시간적 범위에 대한 유연성은 3D-pooling을 통해 얻을 수 있습니다. C3D라고 불리는 이 모델은 C3D 네트워크의 첫번째 fc-layer를 average 함으로써 얻어집니다.

Varol et al은 인풋 레이어에서 더 긴 시간동안(더 긴 시간의 인풋을 받을 수 있도록 하고자하는듯?) 3d conv를 수행할 수 있도록 하는 효과를 찾고자 합니다. 개선 사항은 입력의 시간적 깊이를 다른 temporal awareness를 인풋으로 가지는 네트워크와 결합하는 것입니다. ({16, 20, 40, 60, 80, 100}의 시간적 span을 인풋으로 가지는 네트워크 5개를 결합 모델)

 <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/4fcd19b0cc386215b8bd0c466e42934e5baaa4b7/2-Figure1-1.png" />

spatial filter를 3d로 확장하면 필연적으로 파라미터의 수가 증가합니다. 3D 필터의 단점을 개선하기 위해 Sun 은 3D 필터를 2D와 1D필터로 분해하는 것을 제안합니다. 이러한 파라미터의 감소로 이들은 학습 동안의 트랜스퍼러닝이 없이 Zisserman과 비교할만한 성능을 얻었습니다. 

 <img src="https://csdl-images.computer.org/trans/tp/2017/04/figures/donah1-2599174.gif" />


시간적 정보를 이용하기 위해 몇몇 연구는 recurrent 구조를 사용합니다. Baccouche, Donahue는 CNN다음 RNN(LSTM)의 구조로 action recognition 문제를 다룹니다. recurrent라는 단어가 암시하듯이 RNN은 feedback loop을 사용해 dynamics를 모델링합니다. 전형적인 RNN 블록은 x_t 를 받아 z_t를 생성하며 이는 h_t에 기반을 둡니다. 분명하게 RNN은 Linear Dynamical System의 일종이며 비디오를 모델링하기에 충분합니다.

일반적으로 RNN을 학습시키는 것은 그라디언트 vanishing, exploding 때문에 쉽지 않습니다. 이러한 논의를 위해서 하나의 RNN cell의 recursive expression은 h_t = w_h\*h_(t-1) 을 가집니다. 이러한 재귀적 관계는 h_t = w_h\*h_t-1 = ...... = w_h^t\*h_0 까지 펼쳐집니다. 이와 같은 네트워크는 short term dependency 혹은 very long dependency를 배우게 되고 이는 이상적이지 않습니다. LSTM 은 control gate를 토해 이러한 이슈를 해결합니다.


 <img src="https://image.slidesharecdn.com/multi-modalretrievalandgenerationwithdeepdistributedmodels-160503181152/95/multi-modal-retrieval-and-generation-with-deep-distributed-models-57-638.jpg?cb=1462299229" />


action을 분류하기 위해 Baccouch는 3D-conv 네트워크로 추출한 feature를 LSTM에 집어넣습니다. 3D-conv와 LSTM은 각각 독립적으로 학습됩니다. 즉 3D-conv 네트워크가 action label을 통해 학습되고, 3D-conv 네트워크의 학습이 끝나면 conv feature가 LSTM 네트워크의 학습에 사용됩니다.


 <img src="https://csdl-images.computer.org/trans/tp/2017/04/figures/donah1-2599174.gif" />

LSTM에 기반한 다른 구조는 Donahue에 의해 제안되었고 end-to-end 학습이 가능합니다. LRCN이라는 이름의 구조는 action recognition 뿐만 아니라 이미지, 비디오 캡셔닝에도 성공적입니다. CNN-LSTM의 end-to-end 학습과정을 통해 spatiotemporal filter가 데이터를 기반으로 학습됩니다. 


### 3.2 Multiple Stream Network

시지각 에서 우리의 시각 피질의 Vental Stream? 은 appearance, color, identity를 처리합니다. 객체의 motion, location은 분리되어 처리됩니다. action recognition을 위해 외형적 정보와 motion 정보를 분리하는 딥러닝 네트워크가 고안되었습니다. 

 <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHR6yaME5Y-TXAA0Ct4RX9Anumi-wXtQpubZ7Lh5t3pOaotuv4rw" />


Simonyan, Zisserman 은 처음으로 multiple stream CNN 구조를 소개했습니다. 이는 action recognition을 위해 2개의 네트워크가 병렬적으로 사용됩니다. spatial stream 네트워크라고 불리는 것이 인풋으로 optical flow를 얻는 동안 spatial stream 네트워크는 raw 비디오 프레임을 인풋으로 받습니다. 

- <b>Pretraning for the spatial stream network</b> : scratch부터 spatial stream 네트워크를 학습하는 것은 실무적으로 좋지 않습니다. 경험적으로 imagenet을 통한 pretrained 모델을 fine-tuning 하는 것이 높은 accuracy를 얻었습니다.
- <b>Early fusion for the temporal stream network</b> : temporal stream의 인풋인 optical flow를 early fusion등으로 쌓는것이 효과가 있습니다.
- <b>Multi-task learning for the temporal stream network</b> : temporal stream 네트워크는 비디오 데이터로부터 순수하게 학습되어야 하니다. 이는 매우 깊은 구조에서 중간 혹은 작은 데이터로는 매우 힘든 것으로 관찰 되었다. 이러한 어려움을 피하기 위해 temporal stream 네트워크는 하나이상의 classfication 레이어를 갖도록 수정된다. 각 classfication 레이어는 특정 데이터셋에서 따로 연산되며(하나는 HMDB-51 하나는 UCF-101) 각 데이터세트에서 오는 비디오에만 응답을 합니다. 이 구조는 multi-task learning을 통해 representation을 배우고 하나의 task뿐만아니라 다른 task에도 적용됩니다.

 <img src="https://ilguyi.github.io/assets/2017-06-15/Two-stream.png" />

(a) : LRCN, (b) : two-strean , (c) : fused-two-stream

이 두개의 stream은 softmax를 통해 합쳐집니다. Feichtenhofer는 중간층에서의 합침이 성능을 향상시키고 파라미터의 수를 상당히 줄여줍니다. 마지막 conv 레이어 이후에 fusion이 일어날때 가장 높은 성능을 보이며 흥미롭게도 마지막 conv layer 뒤에서 fusion을 하면 매우 파라미터가 많은 두개의 stream의 fc-layer를 하나로 합칠 수 있음을 보였습니다. 원래의 네트워크와 비교하면 fused-네트워크는 파라미터의 갯수가 절반으로 줄었지만 여전히 잘 수행합니다.


 <img src="http://wanglimin.github.io/tdd/TDD.PNG" />


Two-stream 네트워크의 확장은 Wang의 연구도 포함합니다. 여기서 dense trajectories는 two-stream네트워크의 feature맵을 추적하고 이를 fisher vector를 통해 합칩니다. 여기서는 세번째 stream으로 audio signal을 사용하는 네트워크가 추가됩니다.


 <img src="http://feichtenhofer.github.io/pubs/cvpr16_architecture.jpg" />


optical-flow 프레임은 두개의 네트워크에서 사용되는 유일한 motion과 관련된 정보입니다. 이러한 것은 two-stream 네트워크가 long-term motion dynamics 를 포착할 수 있는지에 대한 의문지 들 수 있습니다. <span style="color:RED">(optical-flow로는 long-term motion을 모델링 할 수 없다.)<span> 딥러닝 구조와 handcrafted 솔루션을 효과적으로 합친 것은 이 페이퍼의 범위를 벗어납니다.

### 3.3 Deep Generative Model

거의 혹은 전혀 supervision을 필요로 하지 않은 딥러닝 구조를 통해 얻을 수 있는 보상은 웹에서 제공되는 방대하고 지속적으로 증가하는 비디오를 고려할때 상상을 초월합니다. 좋은 생성 모델은 underlying 데이터의 분포를 정확하게 배우는 것입니다. sequence 분석을 위한 생성모델은 미래의 sequence를 예측하는데 사용됩니다. x1,x2,,,xn sequence가 주어졌을때 미래인 x_t+1을 예측하는 것으로 볼 수 있습니다. 이러한 task는 위에서 언급된 것과 다르며 라벨링이 필요가 없습니다. 그러나 모델에 의해 sequence의 contents, dynamic이 잘 모델링되어야 정확한 예측이 가능합니다. Vincent, Goodfellow, Hochreiter의 구조는 이러한 목표를 달성하고자 합니다. unsupervised 방법으로 temporal data를 학습하고자 합니다. 비디오 분석같은 라벨링의 비용이 큰 것에서 unsupervised 방법은 선호됩니다. 이들의 잠재력을 느끼면서 depp generative 구조를 리뷰합니다.

### 3.3.1 DYNENCODER

LDS 모델링에서 영감을 얻은 Doretto, Yan의 Dynencoder 를 소개합니다. 이는 deep auto-encoder의 일종으로써 video dynamics를 포착합니다. Dynencoder의 가장 기본적인 구조로써 3개의 레이어가 구성됩니다. 첫번째 계층은 인풋 x_t를 hidden state h_t에 매핑합니다. 두번째 레이어는 현재 h_t를 사용하여 h_t+1을 예측합니다. 마지막 레이어는 예측된 h_t+1 을 통해 x_t+1 프레임을 생성합니다. 훈련의 복잡성을 줄이기 위해 네트워크의 파라미터는 2단계로 학스브됩니다. pretraning 단계에서, 각 레이어는 독립적으로 학습됩니다. pretraining이 완료되면 end-to-end fine-tuning이 수행됩니다.


 <img src="https://i.ytimg.com/vi/PqESDEqeMJI/maxresdefault.jpg" />


Dynencoder는 dynamic texture를 생성하는 것으로 보여집니다. Dynencoder를 생각하는 하나의 방법은 spatiotemporal information을 represent하는 compact way입니다. Dynencoder가 주어진 비디오의 reconstruction error는 classfication을 위한 평균으로 사용될 수 있습니다.(Dynencoder로 학습된 네트워크를 classfication으로 사용할 수 있다는 말인듯)


### 3.3.2 LSTM Autoencoder model

action recognition을 위한 생성모델은 long-term cue를 발견하도록 기대되기에 LSTM을 사용하는 것은 바람직해 보입니다. Srivastava는 LSTM autoencdoer 모델을 소개합니다. LSTM autoencoder는 두개의 Encdoer, Decoder LSTM으로 구성되어져 있습니다. encoder LSTM은 인풋 sequence를 받아 해당하는 compact representation을 학습합니다. encoder LSTM의 state는 sequence의 apperance, dynamics를 포함합니다. sequnece의 compact represenation은 encoder LSTM의 state가 되도록 합니다. decoder LSTM은 학습된 representation을 받아 input sequence를 재구성 하도록 합니다. 자세한 그림은 15를 참조하세요

 <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/025720574ef67672c44ba9e7065a83a5d6075c36/3-Figure4-1.png" />

- LSTM 인코더의 internal state는 인풋 sequence의 compressed version을 포착합니다. 이 state는 두개의 decoder 모델에 카피되어집니다. reconstruction decoder는 original 프레임으로 복원을하려고 하며 predictive 모델은 미래의 프레임을 예측하려고 합니다.

LSTM autoencoder는 미래의 sequence를 예측하는데 사용될 수 있습니다. 실제로 합쳐진 모델은 input sequence를 재구성하고 미래를 예측하는 모델이고 정확한 응답을 가져옵니다.


### 3.3.3 ADVERSARIAL Model


 <img src="https://image.slidesharecdn.com/creativeaimeetup-170901140449/95/emily-denton-unsupervised-learning-of-disentangled-representations-from-video-creative-ai-meetup-37-638.jpg?cb=1504276009" />

생성모델을 학습할때 여러가지 어려움을 피하기 위해 Goodfellow는 GAN이라는 generative model과 disriminative 모델이 적대적으로 경쟁하는 것을 소개합니다. discriminative 모델은 데이터 혹은 generative model로 부터 어떤 sample이 올지를 결정합니다. 학습동안 generative 모델은 원본 데이터 유사한 분포를 가지는 샘플을 생성하는것을 배웁니다. 반면에 adversary mdoel은 주어진 샘플이 진짜인지 아닌지를 배우도록 합니다. Mathieu는 비디오 예측을 위한 multi-scale conv 네트워크를 학습하기 위한 방법으로 이러한 적대적 방법을 적용합니다. 그들은 적대적 학습방법을 pooling layer를 쓰지 않기 위해 사용합니다. 그들은 또한 생성모델에서의 pooling의 이점에 대한 토론을 제공합니다. 


### 3.4 Temporal Coherency Network

이 부분을 결론짓기 전에 우리는 temporal coherency에 대한 갠며을 가져오고 싶습니다. temporal coherency은 weak supervision의 한 형태이며 연속적인 비디오 프레임의 semantically, dynamically 상관된 state입니다.(갑작스러운 motion은 거의 일어나지 않는다) action의 경우 spatial, temporal cue간의 강한 상관이 존재한다. coherent한 sequence는 이들 프레임이 올바른 temporal 순서일때를 말한다. temporal coherency은 각각 올바른 순서의 sequence를 positive로 올바르지 않은 순서의 sequence를 negative로 받음올써 학습이 가능하다. 이는 Goroshin, Wang이 unlabeled 비디오로 부터 robust visual representation을 배우려는 연구에서 사용되었습니다. Misra은 temporal coherency가 action recognition, pose estimation을 위한 딥러닝 모델을 학습시킬때 어떻게 사용될지에 대해 연구합니다. 특히 Siamese 네트워크는 주어진 sequence가 coherent or not인지 학습합니다. 

 <img src="data:image/jpeg;base64,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" />


- 다른 supervised pretrained method와 비교하여 튜플에 의한 학습은 인간의 pose에 더 많은 관심을 쏟는다.
- motion이 풍부한 프레임에서의 tuple을 선택하는 것은 positive, negative tuple을 뽑는것에 대한 모호성을 피할 수 있습니다.
- scratch부터 학습된 네트워크와 비교하여 temporal coherency를 기반으로한 pretrained 네트워크는 성능을 향상 시킬 수 있습니다. 우리는 temporal coherency가 항상 강한 가정은 아니라는 점에 주목합니다. 예를들어 스포츠 경기 (Sport 1m data)동안 방영되는 광고의 급격한 장면변화는 이러한 temporal coherency를 쉽게 위반 할 수 있습니다.


Wang의 temporal coherency에 관한 또다른 연구는 action을 두단계로 나눕니다. 더 구체적으로 x1,x2,,,,,xn의 비디오 프레임을 precondition set X_P = (x1,x2,,,xp) 와 effect set X_e = (x_e, x_e+1,,,,,xn) 으로 나눕니다. 두 세트 모두 딥러닝 모델로 의해 학습되어집니다. 그런다음 x_p의 feature를 x_e의 feature로 매핑하는 transformation을 action으로 인식합니다. 특히 high-level descriptor, transformation은 siamese 네트워크를 통해 학습되어집니다.


 <img src="http://users.cecs.anu.edu.au/~sgould/images/glyphs/icml16.jpg" />

Rank Pooling은 sequence의 temporal 진화를 포착하기 위한 효과적인 해결책입니다. 랭킹을 통해 video representation을 배우고 action classfication은 별도로 수행됩니다. 이는 max pooling과 같은 것과 달리 rank pooling 연산을 위한 closed form을 쉽게 사용할 수 없습니다. 최근에 Fernando, Gould는 rank-pooling과 classfier를 end-to-end로 학습시키는 방법을 제안했습니다. 딥러닝과 관련되진 않았지만 Fernando는 계층적 rank pooling을 제안합니다. 이는 반복적인 rank pooling 연산을 적용하여 비디오에서의 multiple level of dynamical granularty를 코딩합니다.

완벽함을 위해 discrete information space에서의 RNN의 랭귀지 모델링의 성공에 대해 이야기를 합니다. 이를 기반으로 그들은 비디오에 대한 discreate 구조를 소개합니다. 이는 비디오 프레임을 이미지 patch의 모음으로 representation 합니다. 놀랍지 않게도, natural 비디오는 dynamics word sequence posess가 부족하기에 language model이 비디오 영역에서 뛰어난 이유에 대해 힌트를 줍니다. 이들의 관찰에 기초하여 Ranzaro는 recurrenct convnet이 비디오 모델링에 더 robust할 수 있다는 것을 제안했습니다.


### 4. A Quantitiative analysis

이 섹션에서는 위에서 언급한 솔루션에 대한 높은 수준의 분석을 제공합니다. 우리는 몇몇의 성능을 강조표시하고 성능에 초점을 맞춰서 미래에 다뤄져야 할 도전과 방향에 대해 논의합니다.

### 4.1 What is measured by action dataset


Dataset|	Source|	No. of Videos|	No. of Classes
---|---|---|--- 
KTH	|Both outdoors and indoors	|600	|6
Weizmann|	Outdoor vid. on still backgrounds	|90|	10
UCF-Sports|	TV sports broadcasts (780x480)|	150|	10
Hollywood2|	69 movie clips|	1707|	12
Olympic Sports|	YouTube| - | 	16
HMDB-51|	YouTube, Movies	|7000|	51
UCF-50|	YouTube|	-|	50
UCF-101|	YouTube	|13320	|101
Sports-1M|	YouTube|	1133158|	487

action recognition 방법의 성능은 일반적으로 공개된 데이터셋으로 비교 가능합니다. 사용가능한 데이터셋에 대한 리스트와 세부정보는 테이블2에 있습니다. 일반적으로 어떤 데이터셋에 든지 적용가능한 universal solution은 없습니다. 가능한 경우 우리는 왜 솔루션이 성공적인지에 대한 요점을 제공합니다.

솔루션이 진보하는 과정에서 action 데이터셋도 그들의 복잡성이 증가하고 있습니다. 데이터셋의 복작도는 보통 얼마나 실제와 닮았는지로 표현됩니다. 예를들어 KTH, Weizmann 데이터셋은 제한된 환경(제한된 카메라 모션, 배경이 거의 없는) 에서의 인간의 행동을 포함합니다. 게다가 action의 범위가 달리기, 뛰기, 점프 같은 기본적인 action으로 제한됩니다. KTH, Weizmann 데이터셋의 솔루션은 구체적인 필요성이 고려되지 않는 한 별로 쓸모가 없습니다. motion pattern을 고려중이라면 두 데이터셋이 유용하다는 것을 인정합니다.(motion pattern을 학습하기 위해선,,, 간단해서 잘 학습된다는 뜻인가?)


복잡도는 증가하고 있습니다. 유투브 비디오로 부터 데이터셋을 얻고 영화, 티비 방송으로 부터 데이터를 얻습니다(HMDB-51, UCF-101). 유투브 비디오는 비전문가와 handycam에 의해 녹화됩니다. 이 결과 유투브 데이터셋은 흔들린 카메라 모션을 포함하고, viewpoint 변화, 해상도 불일치가 발생합니다. 이러한 데이터에서 잘 동작하기 위해선 솔루션이 위와 같은 변화에 충분해야 합니다. HMDB-51, UCF 101 데이터셋에서는 action이 시간 영역에서 crop되는걸 볼 수 있습니다. 그러므로 이러한 데이터셋은 action localization을 측정하기엔 불충분합니다. HMDB-51, UCF-101의 재밌는 특징은 우리가 subtle class를 포함한다는 것입니다. 씹는행동, 말하기, 바이올린치기, 첼로치기 와 같은 subtle class를 구별하는걸 배우는 것은 spatial, temporal cue에 대한 더 깊은 이해가 필요합니다.

영화와 많은 스포츠 방송은 여러 viewpoint에서 촬영된 다음에 하나로 편집됩니다. 이는 비디오 stream에서의 갑작스러운 viewpoint의 변화를 가져옵니다. Hollywood2, SPort1m 데이터셋은 이러한 view-point/editing 복잡성이 포합됩니다. 또한 action은 보통 small portion of the clip 에서 나타납니다. recognition을 더 어렵게 하기 위해 sport-1m 데이터셋은 관중과 배너광고가 포함되어 있습니다. 그러므로 temporal coherency를 사용하는 방법은 Sprot-1m 에서 실패할 수 있습니다. sport-1m , hollywood2 데이터셋은 모두 노이즈가 심하더라도 텍스트, 스크립트 분석을 통한 라벨이 주어집니다.

앞에서 언급했듯이 HMDB-51, UCF-101, Hollywood2, Sport-1m 데이터셋은 motion clue만으로 구분할 수 없습니다. 몇몇 상황에서는 object와 관련된 object가 무엇인지가 더 중요합니다. 이의 좋은 예제로 23가지의 유형으로 구별된 Sports-1m의 당구 카테고리입니다.따라서 알고리즘은 object의 디테일을 학습하는것이 좋은 성능을 낼 것으로 예상됩니다.(당구와 같은 조그만 공이 어떻게 움직이는지를 배워야 하니까?)

딥러닝 구조는 데이터가 많이 필요한 환경으로 유명합니다. 따라서 딥 네트워크를 KTH, Wiezmann과 같은 작거나 중형 사이즈에서 튜닝하는 것은 어렵고 불만족스러운 결과를 종종 냅니다. Sports-1m 데이터셋은 이러한 제한을 완화하기 위해 모여졌습니다. 더 깊은 네트워크의 학습과 튜닝이 가능하도록.

### 4.2 Recognition result

테이블3 에는 7개의 action dataset에 대한 accuract에 따른 31개의 방법론이 나와 있습니다. accuract는 원본 논문에서 보고 된 것입니다. 각각의 사례를 개별적으로 비교하는 대신 우리는 솔루션의 다양한 구조에 대해 높은 수준의 비교를 제공합니다

##### Various classes of soltuoin

accuracy를 살펴보면 딥러닝과 핸드크레프트 representation에 기반한 방법 모두 잘 수행된다는 것을 알 수 있습니다. 이는 이미지 classfication을 생각했다면 기대하지 못한 관찰이였을 것입니다. 예를들어 stacked FV encdoing of trajectory descriptor의 경우 최신의 딥러닝 모델보다 성능이 좋습니다. 반대로 비슷한 솔루션 (SIFT + FV와 CNN의 비교)의 이미지 classfication에서와 모순됩니다. 이의 여러가지 원인중 하나로는, 데이터의 부족은 무시 될수 없습니다. 이러한 데이터 부족의 지배적인 주제는 pretrained 모델로 부터 성능향상을 얻는 것입니다.

##### State of the art solution

<b>Handcrafted soltion</b> hadcrafted solution에 초점을 맞추면 dense trajectory descriptor에 의해 성능향상을 얻을 수 있습니다. 이 descriptor은 Fisher vector, rank pooling과 같은 방법으로 쉽게 통합될 수 있고 이는 HMDB-51, UCF-101에서 경쟁력있는 결과를 이끌었습니다.

<b>Deep-net solution</b> 딥러닝 솔루션에 주의를 돌리면서 우리는 spatiotemporal network, two-stream network가 다른 구조보다 성능이 뛰어난 걸 발견했습니다. 최근의 이 두 구조에는 3D-conv 필터를 사용합니다. Feichtenhofer, varo의 작업에도 3d covn, pooling이 포함되어져 있습니다. wang의 논문또한 더 깊은 모델이 성능을 향상시키는데 도움이 된다고 제안합니다. 그러나 더 깊은 모델을 학습시키기 위해서는 더욱 엄격한 data augmentation 기술이 요구됩니다.


##### FUSION WITH DENSE TRAJECTORIES SEEMS TO ALWAYS HELP.

대부분의 최신 딥러닝 기반 솔루션들은 wang의 관찰 결과 성능향상이 가능합니다. 딥러닝을 통해 만들어진 구조가 handcrafted trajectory descripotr를 통해 보완이 가능합니다. 딥러닝 네트워크(대부분의 경우) 와 trajectory descriptors(RGB, optical flow)가 비슷한 인풋을 받는다는 것을 언급할만 합니다. Simonyan, Zisserman은 spatial gradient에 일부분의 temporal stream filter가 반응한다는 것을 관찰했습니다. 비슷하게 MBH trajectory descriptor는 optical flow frame의 spatial gradient을 사용하여 도출 됩니다.

### 4.3 What algorithmic change to expect in future?

딥러닝 구조로 향하는 다른 컴퓨터 비전에서의 흐름에 따라 action recognition연구에도 딥러닝을 활용하려고 하는 흐름이 주도하고 있습니다. 비디오 데이터로 부터의 네트워크의 학습의 어려움은 새로운 연구의 필요성을 느끼게 합니다. knowledge transfer 이미지 혹은 다른 소스로 부터의 학습된 모델로 성능을 개선 하는 방안에 대한 연구가 필요합니다. 이와 관련되고 덜 연구된 knowledge transfer에 대한 아이디어는 heterogeneous domain adaptation 연구입니다.

action recgonition과 관련된 구조를 고려할때 기억해야 할 키워드는 3D-CONV, temporal pooling, optical flow frame, LSTM 입니다. 앞서 언급된 요소들은 개별적으로 발전했지만 새로운 방법은 성능을 높이기 이해 이들을 합치는 것을 목표로 합니다. 우리는 spatiotemporal learning을 위한 네트워크의 구조를 위한 일반적인 형태가 될 것이라고 고려합니다. 

다른 기억해야 할 것은 성능 향상을 위해 세심한 엔지니어링 접근이 필요합니다. 예를 들어 <span style="color:RED">data augmentation, foveated architecture, distinct frame sampling 전략 등은 필수적으로 나타났습니다.</span>

### 4.4 Bringing action recognition into life

action recognition은 통제된 환경에서의 인식에서 부터 보다 실제에 가까운 환경에서의 솔루션으로 발전해 왔다. 하지만 실제 생활에서 이러한 솔루션을 사용하려면 다음과 같은 깊은 이해가 필요합니다.

- action recognitioh의 실용적 어플리케이션은 joint detection, squence recognition등을 포함하고 있습니다.(detection을 같이하고 ,,, 영상에 대해 적용하는 recognition등을 말하나?) 최근의 연구들은 joint segmentation, action recognition을 같이하고 있습니다.

- 매우 큰 class(많은 label 종류 갯수)에서의 action을 인식하는 대신 실제 어플리케이션에서 유용한 action들로 제약을 두세요. 이를 위해, fine-grained action recognition task는 새로운 문제가 될 것입니다. 그리고 이미 커뮤니티로부터 큰 관심을 받고 있습니다.

### 5. Conclusion

단일 이미지와의 유사성에도 불구하고 비디오 데이터 분석은 훨씬 복잡합니다. 성공적인 비디오 분석 솔루션은 scale, intra-class, noise와 같은 변화를 극복할 필요가 있고 비디오에서의 motion cue를 분석해야 합니다. action recognition은 비디오 분석의 여왕으로 취급될 수 있습니다. action recognition은 다양한 어플리케이션이 존재하여 신체 움직임에 의해 생성되는 복잡한 motion 패턴을 가집니다. 이 논문에서는 우리는 action recognition을 위한 몇몇의 솔루션들을 살펴봅니다. 우리는 먼저 handcrafted representation에 대해 리뷰하고 딥러닝 구조에 집중합니다. 우리는 이 두가지 방법에 대한 비교 분석을 제공합니다.
### Memory Network에 대한 사전지식이 필요

pr12 NMT
https://www.youtube.com/watch?v=2wbDiZCWQtY&index=7&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS

cs224d memory network
https://www.youtube.com/watch?v=Xumy3Yjq4zk&t=149s

LXMLS memory network tutorial
https://www.youtube.com/watch?v=5ekMog_nhaQ


# A Read-Write Memory Network for Movie Story Understanding

### Abstract

우리는 새로운 memory network model인 RWMN 을 제안합니다. 이것은 large-scale, multimodel movie story understanding에서의 QA task를 수행합니다. <span style="color:RED"> RWMN 에서의 중점을 둔 것은 convolutional layer로 구성된 read, write network를 설계하는 것이고 이러한 read, write network 는 유연성과 높은 복잡도를 가지도록 합니다. </span> 보통의 memory augmented network는 각각의 memory slot을 하나의 independent block 으로 취급하는 반면 우리의 multi-layerd CNN은 <span style="color:RED">우리의 모델이 sequential memory를 하나의 chunk로 취급하도록 하며 이것은 adjacent memory block이 strong correlation을 가지기 때문에 sequential story를 나타내는 합리적인 방법입니다.</span> eval을 위해 우리의 모델을 6가지 MovieQA task에 적용을 하였고 몇몇 task에서 최고의 성능을 보였습니다. (특히 visual QA task 에서) 우리의 모델은 story의 내용 뿐만 아닌 더 추상적인 정보를 이해하는 것 처럼 보입니다. (such as relationships between characters and reason for their actions)

### Introduction

video classfication, video captioning, Movie QA 등에서의 성공의 핵심은 모델이 정확한 process, represent 를 수행하고 long sequential information을 저장하는 것입니다. 딥러닝 에서의 우세한 접근 방법은 RNN을 이용하여 sequential input을 모델링 하는 것입니다. 이런 모델은 주어진 정보를 hidden memory(state?) 에 저장하고 매 시간마다 업데이트를 합니다. 그러나 <span style="color:RED">RNN은 input sequence의 길이와 상관없이 고정된 길이의 length memory 에 정보를 축적하게 됩니다.</span> vanishing gradient와 같은 문제들 때문에 먼 거리의 정보들을 잘 활용하지 못합니다.


이러한 문제에 대한 다른 접근 방법으로는 <span style="color:RED">external memory 구조를 사용하는 것입니다.</span> 이는 보통 neural memory network 라고 언급됩니다. external memory 를 사용하는 가장 큰 장점은 neural model이 sequential input을 memory slot에 저장 해 먼 거리의 정보도 사용할 수 있게 합니다. 이러한 능력은 QA task를 풀대 매우 강력합니다. QA task는 종종 매우큰 양의 정보를 기억해야 하며 주어진 question에 대해 가장 연관된 정보에 접근하는 것이 필요합니다. 이러한 이유로 다양한 최신의 모델에 memory network가 적용되었습니다.

MovieQA는 visual QA dataset에서의 어려운 문제중에 하나인데. 2시간 정도의 영화를 이해하고 영화의 content, plot 에 관련된 QA 문제를 풀어야 합니다. MovieQA 는 6개의 source of information에 해당하는 6개의 task로 이루어져 있습니다. 6개의 source는 QA problem, video, subtitile, DVS(디지털 비디오 시스템즈?), scripts, plot sysnopses , open-end information 으로 이루어져 있습니다. 영화를 이해하는 것은 매우 어려운 task 인데 각각의 video frame에 대한 content (각 배우의 행동, 사건이 일어나는 장소등) 을 이해하는 것 뿐만 아니라 더 상위의 지식이며 더 추상적인 것(배우들의 행동의 이유, 그들간의 관계) 를 이해해야 합니다. 예를 들어 해리포터 영화에서 "What does harry trick lucius into doing? : "Freeing dobby" 라는 QA에 답하기 위해서는 Dobby가 lucius의 집요정이라는 것을 알고 dobby가 루시우스로 부터 벗어나고 싶어 하며 해리와 긍정적인 관계를 가졌고, 해리가 도비를 도왔다는 것을 이해 해야 합니다. 이러한 정보는 영화안에서 시각적 혹은 문맥적으로 관찰 가능하지만 배우들간의 관계, 사건간의 상관들은 추론을 통해 알아야 합니다.

우리의 목적은 large-scale, multimodal movie story understanding을 위한 새로운 memory network를 제안하는 것입니다. 모델의 input은 매우매우 길수 있고 (비디오가 2시간 정도의 길이를 가집니다) 또한 multimodal 일 수 있습니다. (text만 주어지던지 video, text가 같이 주어지든지). 우리이 RWMN가 주목한 것은 memory read/write operation을 매우 높은 복잡도, 유연성을 가지도록 정의하는 것입니다. 이 memory network는 여러겹의 convoltional layer를 가지게 됩니다. <span style="color:RED">현재 존재하는 memory network는 각각의 memory slot을 독립적인 블록으로 보지만 인접한 memory block 은 종종 strong correlation을 가지고 이는 sequential story를 representation 합니다.</span> 이러한 것 처럼 전체의 스토리를 sequence of closely interconnected abstact event 들로 인식하는 것과 인간이 스토리를 이해하는 것과 일치합니다.  그래서 read, write memory cells이 일종의 chunk 처럼 동작하기를 원했고 이를 multiple convolutional layer로 구현을 하였습니다. introduction을 마치기 위해서 우리의 논문의 공헌을 요약합니다.

1, RWMN 이라는 새로운 memory network 를 제안하는데 이는 read/write network를 통해 더 복잡하고 추상적인 정보를 memory slot에 저장하도록 합니다. 알고 있기론 이것이 memory network에 multiple layer CNN을 사용한 첫번째 시도입니다.

2, RWMN은 몇가지 MovieQA task에서 최고의 성능을 보였습니다. RWMN 은 5개의 task 중에서 4개에서 최고의 성능을 보였고 (validation set), 6개의 task 중에서 4개에서 최고의 성능을 보였습니다.(test set) 우리의 질적, 양적 evaluation은 read/write network가 higher-level information을 사용하도록 한다는 것을 보였습니다. 특히 visual QA task 에서!!

### Related Work

<b>Neural Meomory Network</b> : 최근 많은 논문들이 explicit memory network를 사용하여 sequential data를 모델링 해왔습니다. memory network의 memory의 access는 content-based addressing, location-based addressing으로 나뉘어 집니다.

content-based addressing은 controller가 key vector를 생성하도록 하고 어떤 memory cell 과의 유사성을 measure 합니다. key vector와 연관되는 어떤 memory cell에 attention을 줘야 할지를 찾는?? Location-based addressing은 간단한 대수적 연산을 통해 정보를 저장하거나 인출해야하는 주소를 찾아냅니다. (key vector와 상관 없이)

NTM, DNC, D-NTM 은 read/write operation의 전체의 프로세스를 배우려고 합니다. 그래서 주어진 문제에서의 모델의 복잡도가 높습니다?? 이들은 sorting, copying, graph traversal 같은 복잡한 task에 적용되었습니다. [15, 23, 27] 같은 memory network 는 NTM 과 비슷한 continuous memory representation을 통해 QA 문제를 다룹니다. 그러나 NTM이 location-based ,content-based 모두에 영향을 받는것과 달리 content-based memory interaction 만을 사용합니다. 이들은 multi-hops? recurrently memory 의 개념을 적용하여 causal reasoning이 필요한 QA task 에서 성능 향상을 보였습니다. [18, 30]은 key-value memory network 를 제안하였고 이는 external knowledge based로 (key, value) 로 구성된 정보를 저장합니다. 이러한 방법들은 content가 중요하거나 facts in a context 가 중요한 wikimovies, babi dataset 에서 좋은 결과를 보였습니다.

[2,20] 은 read/write operation을 매우 큰 memory를 사용할수 있도록 하였고 [2] 는 계층적 memory를 구성하였습니다. [20]은 read/write operation을 sparse 하게 만들어 연산량을 줄이고  scalability(확장성? 큰 시스템에 사용가능하다는 뜻인가?) 를 증가시켰습니다.

이전의 모델과 비교하여 RWNM은  CNN으로 구성된 read/write network를 사용할 수 있게 하였고 이는 더 추상적인 정보(배우들간의 관계, 배우의 행동의 이유, 스토리에서의 사실에 대한 이해) 를 저장할 수 있도록 하였습니다.

<b>Model for MovieQA</b> : MovieQA 에 적용된 모델중 end-to-end memory network는 좋은 성능을 내는 접근방법 입니다. 이는 각각의 영화를 shot 들로 나누고 video, subtitle feature의 memory slots을 구성합니다. 그 후 주어진 question에 관련있는 정보에 attention 하기 위해 content-based addressing을 이용합니다. 최근에 [25] 는 compare-aggregate(비교하고 합치는??) 프레임 워크를 제안하였는데 이는 setence의 유사성을 word-level로 매칭합니다. 그러나 이는 MovieQA의 하나의 task(plot synopse) 에만 사용 되었습니다.

LSMDC, MSRVTT, TGIF-QA 같은 다른 Video QA task 들이 존재합니다. 이들은 짧은 비디오를 이해하는데 중점을 두고 clips 안의 사실들에 대답하는데 중점을 두고 있습니다. [29  End-to-end Concept
Word Detection for Video Captioning 김건희 교수님 페이퍼]는 video captioning, video QA, video retrieval 에서의 주목할 만한 성과를 보였고 이는 end-to-end trainable concept-word-detector along with vision-to-language model 로 구성되어져 있습니다. (vision-to-language embedding하는.. concept word를 찾는 모델??)

### Read-Write Memory Nerwork(RWMN)

그림2 는 RWMN의 전체의 구조를 보여줍니다. RWMN은 moview content를 적절한 representation을 통해 memory에 저장하고, 주어진 query에 관계있는 정보들을 뽑아냅니다. 그리고 5개의 선택지중의 answer를 선택합니다.(QA task 중의 하나인데.. 주어진 5개 혹은 여러개의 answer 중에 옳은 답을 고르는 task가 있다)

MovieQA 데이터 셋의 QA 의 포맷에 따라 모델에 주어지는 인풋은 다음과 같습니다.

1, sequence of video segment and subtitle pairs (대략 2시간의 영화에 대해서 평균 1558개의 pair가 존재합니다)
2, a question "q" for the movie
3, 후보가 되는 5개의 answer

MovieQA의 video+subtile task의 경우를 예로 들자면
1, 각각의 s_i 는 각 장면의 문장입니다.
2, v_i 는 video subshot 인데 s_i에 시간적으로 정렬되어있고 6 fps 로 샘플링 되어졌습니다.
모델의 아웃풋은 5개의 answer에 대한 score vectore를 출력합니다. v_i is a set of frame

다음으로는 write/read network 를 이용해 movie를 answer로 임베딩하는 구조를 설명할 것입니다.

### Movie Embedding

<b>visual embedding</b>

우리는 각각의 subshot v_i 와 sentence s_i 를 다음과 같은 방법으로 representation 했습니다. v_i에 대한 각각의 v_ij 프레임 마다. Resnet-152 를 적용하여 feature를 얻습니다. 그 후 subshot v_i 에 대한 representation 으로 모든 v_ij 에 대해 mean-pool을 적용하여 얻습니다.

<b>word embedding</b>

먼저 sentence를 word로 나누고 pretrained Word2vec 을 적용합니다. postion encdoing(PE) 를 적용한 mean-pool 을 이용해 s_i 를 representation 합니다.

<b>multimodal embedding</b>

마지막으로 v_i, s_i에 대한 multimodal space embedding을 얻기 위해 Compact Bilinear Pooling(CBP) 를 적용합니다.
우리는 이러한 단계를 모든 n pairs 에 적용하였고 2D movie embedding matrix를 얻게 되고 이는 write network의 인풋으로 들어가게 됩니다.

### The Write Network

write network는 movie embedding matrix E를 인풋으로 받아서 memory tensor M을 출력하게 됩니다. write network의 motivation은 <span style="color:RED">인간이 영화를 이해할 때 단순한 speech, visual content의 sequence가 아닌 인접한 대화아 장면들을 묶어 일종의 event, episode의 형태로 묶는다는 것입니다.</span> 이것이 각각의 memory cell이 각각의 n개의 movie embedding을 분리하여 저장하는 것 대신 이웃한 movie embedding을 연합하는 것이 필요합니다. 이러한 adjacent embedding을 하나의 slot으로 저장하는 것을 구현하기 위해 write network를 CNN으로 구현했습니다.

Nx4096 의 movie embedding (전체 영화 하나의 n개 subshot에 대한 모든 feature인듯?) 에 먼저 FC layer 를 적용합니다 (4096 => 4096) 이는 E[i] (i 번째 mobie embedding) 을 d dim의 vector로 projection 합니다. 이 FC lyaer는 E의 차원을 query embedding, answer embedding의 차원으로 줄여줍니다. 이렇게 하는 것의 나중에 convolution operation의 횟수를 줄여주게됩니다. 그 후 eg) 40xdx3의 사이즈의 필터를 swv, swh 스트라이드로 convolution 합니다. 마지막으로. memory를 생성하여 Mxdx3 의 메모리를 생성합니다. m = [(n-1)/stride + 1 ]

write network는 multiple convolutional layer가 될 수 있어 여러겹의 layer를 겹쳐서 M을 얻을 수 있습니다.


### The Read Network

read network는 qeustion을 받아서 q과 관련된 M으로 부터 답을 생성합니다.

<b>Question embedding</b> : Word2Vec 으로 부터 q를 얻은 후 이를 projecttion 하여 u를 얻습니다. y를 dx300의 매트릭스가 됩니다. 그 다음 M과 embedding 된 u를 받아 score vector o를 출력합니다.

<b>Query-dependent memory embedding</b> : memory M을 query-dependent 하도록 transform 합니다. 이것의 직관은 주어진 query에 따라서 관련된 정보가 memory slot으로 부터 얻어져야 합니다. 예를들어 해리포터 영화에서 하나의 memory slot이 해리가 마법을 쓰는 장면에 대한 정보를 포함한다고 할때 이 memory slot은 어떤 질문이냐에 따라 읽을 수 있도록 해야 합니다.

메모리 M을 query-dependent memory 로 transform 하기 위해 memory slot M과 query embedding u에 CBP를 적용하여 얻습니다.

<b>Convolutional memory read</b> : 우리의 직관은 다음과 같습니다 movie understanding에 대한 올바른 답을 얻기 위해서는 관련된 장면들의 series를 하나로 연결해주는 것이 중요합니다. 그래서 우리는 CNN 아키텍쳐를 사용해 sequential memory를 chunk로 만들도록 해씃ㅂ니다.


### 관련 용어 정리

### bilinear pooling

visual represenation vector, sentence representation vector를 하나로 embadding 을 하기 위해서 사용하는 방법이 여러가지가 있다.

1, concate : 단순히 두 vector를 concate 하는 것, 네트워크가 학습 과정에서 알아서 이 두 vector가 각 element사이의 관계를 학습 해 줄 것이라고 생각.  
2, multiply : 두 vector를 elementwise multiply 해서 사용한다.  

근데 1,2 방법의 경우는.. 두 vector의 요소 사이의 상호작용을 모델링 하는 것이 힘들다 그래서 사용하는게 3번 방법.

3, bilinear pooling : 두 vector의 outer product를 해서 이걸 feature로 사용한다.  

이 때의 문제가 bilinear pooling을 사용하기 위해서는 outer product를 구해야 하는데 만약 visual feature가 2048 dim, sentence feature가 300 dim 이라고 하면 2048 * 300 dim을 가진.. 매트릭스가 되버린다 outer matrix를 구하는 것도 힘들며,,,, 이를 fc layer에 넣는것도 매우 연산량이 크고 파라미터가 많아지게된다.

그래서 나온게 Compact bilinear pooling 인데.. bilinear pooling을 compact 하게 해보겠다는 뜻이다. 즉 outer matrix를 근사해서 사용하겠다!!

여기서 사용하는 것이 <b>Count Sketch projection</b> : 이것은 n 차원의 vector를 d 차원의 vector로 근사하겠다는 것이다. visual vector, sentence vector를 count sketch projection에 의해 각각 2개의 d차원 vector로 근사하고, outer matrix를 구한다. Ψ(x ⊗ q; h; s) = Ψ(x; h; s) ∗ Ψ(q; h; s) 이것에 의해서.. outer matrix를 구하고 count sketch 쓰는 것과 각각 count sketch을 적용하고 이에 컨볼루션 하는것이 같은 연산이다. 즉 outer matrix를 근사하여 사용한다.

Ψ(q; h; s) 의 부분도 결국 300dim 정도의 크기를 가지므로 이를 spatial domain에서 처리하는 것보다.. 주파수 도메인에서 처리하는 것이 연산량이 적어서.. 이를 주파수 도메인에서 처리한다.

Count Sketch : Ψ(v; h; s; n)

h는 1~d의 숫자로 , s는 {-1, 1}으로 랜덤하게 초기화가 된다. v는 우리가 project할 vector, n은 결과물의 차원,
y[h[i]] = y[h[i]] + s[i] · v[i],, 즉 대충 생각해보면 어떠한 랜덤성에 의해 다시 표현된 vector는 비슷한 의미를 가지게 된다. 이때 사람들이 이러한 랜덤성 때문에 일반화의 역할을 수행한다고 하기도 한다고 함.

### Position Encoding

<b>position encoding</b> : sentence를 vector로 나타내기 위해서 back-of-word라는 개념을 사용하는데 즉 sentence 안의 단어들의 word vector의 mean pool을 통해 하나의 sentence를 나타낸다. 이때 문제가 mean을 해버리기 때문에 문장의 representation에 단어의 순서가 반영이 되지 않는다 ex) i am a boy 와 i boy am a 가 같은 representation이 되는데.. 이는 분명히 다른 sentence이다.

이걸 나타내기 위해서 position vector라는 것을 사용하는데 이때 이 vector는 lkj = (1 − j=J) − (k=d)(1 − 2j=J) 로 고정하여 사용한다. 그래서 mean pool 전에 word vector에 postion vector를 element multiply 해주고.. 그다음 mean pool을 해서 사용을 한다.

sentence vector를 mean pool 해서 사용해도 되는 이유는,, word vector는 고차원 공간의 벡터이므로 이들을 단순히 sum해도 어떠한.. 정보를 나타내고 있을 것이라는 백그라운드.

memory matrix가 MxDx3 의 차원을 가지는데 이를 40xdx3 필터로, 30, 1 stride를 적용한다. 왜 d차원으로 딱 맞는 width를 가지는데 stride 1로 할까를 물어봤었는데..

### w_conv^w 에 중간에 1의 stride는 뭐지?

distributed representation 이라는 이론이 있다. 딥러닝의 장점중에 하나가 representation vector가 분산 표현되어 있다는 것이고 이 vector의 일부분에도 어떠한 의미가 있다는 것이다. 이 때문에 1 stride로 covolution해도 의미가 있는 정보를 뽑아 낼 수 있다.

qeustion representation 후..tile이라는 것을 거치는데. 이는 bilinear pooling을 하기 위해 d vector를 mx3 만큼 복사 해준 것!
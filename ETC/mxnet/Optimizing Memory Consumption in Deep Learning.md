# Optimizing Memory Consumption in Deep Learning

지난 십년 동안 딥러닝 분야에서의 꾸준한 트렌드는 더 크고 깊은 네트워크를 구성하는 것입니다. 최근의 하드웨어의 발전에도 불구하고 최신의 딥러닝 모델은 더 큰 GPU RAM을 필요로 합니다. 오늘날에는 적은 메모리를 사용하면서 더 큰 모델을 학습시키는 것이 바람직 합니다. 적은 메모리를 사용함으로써 더 큰 배치사이즈를 사용 할 수 있게 되어 더 빠르게 학습 시킬 수 있고 더 높은 GPU 사용률을 달성하게 되었습니다.

이 문서에서는 딥 뉴럴네트워크에서 메모리 할당을 최적화 하기 위한 기술을 살펴봅니다. 우리는 몇가지 해결책에 대해 다루게 됩니다. 우리의 제안은 포괄 적이지는 않지만 이러한 솔루션이 도움이 되며 딥러닝 라이브러리의 디자인 이슈를 해결 할 수 있도록 도와줍니다.

## Computation Graph

먼저 computation graph의 개념을 다시 한번 살펴봅시다. computation graph는 딥 네트워크 에서의 연산 사이의 (data flow)의존성을 기술합니다. graph 에서 실행되는 연산은 작은 연산이거나 보다 큰 연산일 수 있습니다. 다음과 피규어는 2개의 compuation graph의 예를 보여줍니다.

![Comp Graph Example](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/comp_graph_example.png)

compuation graph의 개념은 Theano, CGT 와 같은 라이브러리에 의해 명시적으로 나타낼 수 있습니다. CXXNet, Caffe와 같은 라이브러리에서는 computation graph는 configuration file을 통해 암묵적으로 나타납니다. 이러한 라이브러리의 가장 큰 차이는 gradient를 어떻게 계산하는지 입니다. 이들은 주로 2가지 방법을 통해 gradient를 계산합니다. 1) 같은 그래프에서의 back-prop 수행 2) 필요한 gradient를 계산하기 위해 명시적으로 backward path를 표현 

![Backward Graph](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/back_graph.png)

Caffe, CXXNet, Torch와 같은 라이브러리는 원래의 그래프에서 back-prop을 수행하는 방법을 사용합니다. Theano, CGT와 같은 라이브러리는 backward path를 명시적으로 나타냄으로써 gradient를 계산합니다. 우리는 최적화를 위한 몇가지 장점이 있기 때문에 backward path를 명시적으로 나타내는 방법을 사용합니다.

그러나 backward path를 명시적으로 나타내는 것이 Theano, CGT 처럼 symbolic 라이브러리로 구성을 필요로 하는 것이 아님을 아님을 주목해야 합니다. layer-based (forward, backward를 하나로 묶어서 제공하는 방법) 라이브러리 에서의 gradient 계산을 통해 명시적인 backward path를 사용할 수도 있습니다. 다음의 그림은 layer-based 라이브러리가 어떻게 그래프를 구성하는 지를 보여줍니다. 기본적으로 이러한 방법들은 forward node와 연결된 backward node를 도입하고 ```layer.backward``` 를 호출함으로써 backward 를 수행하게 됩니다.

![Backward Layer](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/explicit_back_layer.png)

이러한 방법은 기존의 모든 딥러닝 라이브러리에 적용됩니다. (라이브러리 간의 몇가지 high-order differentiation과 같은 차이점이 있을 수 있지만 여기서의 주제를 벗어납니다.)

backward path를 명시적으로 나타내는 것이 좋은 이유가 무엇일지 생각해 봅시다. 두가지 예제를 통해 설명 할 수 있습니다. 첫번째 이유는 backward path를 명시적으로 나타내는 것이 computation 간의 의존성을 명확하게 나타내 줍니다. 다음과 같은 그래프에서 우리가 A, B의 gradient를 얻기를 원하다고 생각해 봅시다. 그래프에서 명확히 볼 수 있듯이 ```d(C)``` gradient를 계산하기 위한 computation은 F에 의존적이지 않습니다. 이 의미는 우리는 ```F``` 에 대한 forward computation이 수행된 이후에 할당된 메모리를 해제 할 수 있습니다. 비슷하게 ```c``` 에 대한 메모리도 재사용 될 수 있습니다.

![Backward Prune](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/back_dep_prune.png)

backward path를 명시적으로 나타내는 것의 다른 장점은 forward와 같은 모양이 아닌 다르게 생긴 backward path를 가질 수 있다는 점입니다. 일반적인 예제는 split connection으로써 다음과 같은 그래프를 생각 해볼 수 있습니다.

![Backward Agg](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/back_agg_grad.png)

이 예제에서 B의 출력은 두개의 연산에 의해 참조됩니다. 만약 우리가 같은 네트워크 에서 gradient 를 계산하기를 원한다면 명시적인 split layer를 도입해야만 합니다. 이 의미는 forward pass를 split 하는 것이 필요하다는 것입니다. 이 피규어 에서 foward pass는 split layer를 포함하고 있지 않습니다. 단지 B의 gradient를 계산하기 이전에 자동적으로 gardient aggregation 노드를 추가 해줄 뿐입니다. 이러한 것은 split layer의 출력에 할당해야할 메모리를 아껴주고 foward pass에서 split layer를 위해 데이터를 복사하는 비용을 줄여 주게 됩니다.

명시적인 backward path의 접근 방법을 사용하면 forawd pass, backward pass 사이의 차이가 없습니다. 우리는 단순히 시간 순서대로 computation graph를 실행하고 computation을 수행합니다. 이는 명시적인 backward path 사용하면 분석하기 쉽도록 해줍니다. 우리는 computaton graph의 각 output node에 얼만큼의 메모리를 할당 해야 할지에 대해서만 답을 하면 됩니다.

## What Can Be Optimized

위에서 볼 수 있었듯이 computation graph는 메모리 할당을 최적화 하기 위한 효율적인 방법입니다. 이미 우리는 explicit backward graph를 사용함으로써 약간의 메모리를 절약 할 수 있었습니다. 이제 다른 최적화 방법을 살펴 보고 합리적인 벤치마킹을 위한 베이스라인을 어떻게 결정했는지를 살펴봅시다.

우리가 ```n``` 레이어의 뉴럴 네트워크를 구성한다고 가정해 봅시다. 일반적으로 뉴럴 네트워크를 구현할때 각 레이어의 출력을 저장할 node space를 할당해야만 하며 back-prop 과정에서 사용될 gradient value를 저장 할 공간이 필요합니다. 이는 러프하게 말해서 ```2n``` memory cell이 필요하다는 뜻입니다. explicit backward graph를 사용하더라도 같은 요구사항이 필요한데 backward pass도 forward pass보다는 약간 절약될 수 있지만 거의 비슷한 갯수의 노드가 필요하게 됩니다.

## In-place Operations

우리가 채택 할 수 있는 가장 간단한 기술은 연산들에 걸쳐 in-place memory sharing을 적용하는 것입니다. 뉴럴 네트워크의 경우 activation function과 같은 연산에 이러한 기술을 적용 할 수 있습니다. 다음 그림과 같은 그래프를 생각해 봅시다. 이 그래프에서는 3개의 sigmoid function을 거친 값을 계산하고 싶습니다.

![Inplace op](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_inline.png)

이 그래프에서 sigmoid를 in-place로 계산이 가능 하므로 input, output이 같은 메모리를 공유하게 됩니다. 우리는 상수 메모리의 복잡도로 임의의 길이의 sigmoid function을 계산 가능합니다.

Note : in-place 최적화를 구현할때 실수를 하기 쉽습니다. 다음과 같은 경우를 고려하면, B의 출력이 c에만 사용되는 것이 아닌 F에도 사용되므로 in-place 연산이 불가능 합니다.

![In-place trap](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_inline_trap.png)

이 경우 in-place 최적화를 수행 할 수 없습니다. B의 값은 ```C=sigmoid(B)```가 계산된 이후에도 여전히 F 노드 때문에 필요로 합니다. 모든 sigmoid 연산에 대한 in-place 최적화는 이러한 함정에 쉽게 빠질 수 있기 때문에 사용 할떄 주의 하는 것이 필요합니다.

## Standard Memory Sharing

메모리를 공유하기 위한 방법이 in-place 연산만이 있는 것은 아닙니다. 다음과 같은 예제에서는 E가 연산된 이후에 B의 값이 더이상 필요하지 않습니다. 이 경우 B의 메모리를 E의 결과를 저장하기 위해 사용 가능합니다.

![Normal Sharing](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_normal.png)


Memory sharing은 같은 데이터의 shape를 요구하지 않습니다. 위의 예제를 보면 ```B```, ```E``` 의 shape는 달라 질 수있습니다. (pooling 연산 때문에 출력 값의 shape가 달라질 수가 있다.) 이러한 상황을 다루기 위해 ```B```,```E```에 필요한 메모리중 큰 메모리를 할당 하여 그것을 공유 할 수 있습니다.

## Example of Real Neural Network Allocation

앞에서 살펴본 예들은 간단한 예제일 뿐이며 오직 forward pass의 계산 만을 다루고 있습니다. 하지만 같은 아이디어를 실제 뉴럴 네트워크에도 적용 가능 합니다. 다음 그림은 2-layer perceptron 에 대한 메모리 할당 계획을 보여줍니다.

![Net Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_mlp.png)

이 예제 에서는
- in-place 최적화가 ```act1```, ```d(fc1)```, ```out``` 을 계산 할떄 적용됩니다.
- ```d(act1)``` ,```d(A)``` 간에 메모리가 공유됩니다.

## Memory Allocation Algorithm

지금까지 메모리 할당을 최적화 하기 위한 기술데 대해 살펴 보았습니다. in-place 메모리 최적화와 같은 경우에는 피해야 할 경우가 존재 하느 것을 보았습니다. 그렇다면 어떻게 적절하게 메모리를 할당 할 수 있을까요? 이는 새롭게 직면하는 문제는 아닙니다. 예를 들어 이는 컴파일러에서 register를 할당하는 문제와 매우 비슷합니다. 여기서 우리가 가져올 수 있는 몇가지 기술이 있으며 여기서는 이러한 기술에 대한 포괄적 리뷰를 제공하지는 않습니다. 하지만 이러한 문제를 해결하기 위해 간단하지만 유용한 트릭을 도입합니다.

가장 핵심이 되는 문제는 메모리가 서로 충돌 하지 않도록 리소스들을 할당해야 한다는 것입니다. 좀더 구체적으로는 각각의 변수들은 계산된 순간과 마지막으로 사용되는 순간 까지의 life time을 가지고 있습니다. multi-layer perceptron의 경우에 ```fc1``` 의 lift time은 ```act1``` 이 계산된 이후에 끝나게 됩니다.

![Net Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_mlp.png)

memory sharing을 위한 원칙은 변수들간의 life time이 겹치지 않도록 하는 것입니다. 이를 수행하기 위한 여러 방법이 존재합니다. 먼저 모든 변수를 node 로 삼고 변수들간의 겹쳐지는 lifespan을 edge로 삼는 graph를 구성한 뒤에 graph-coloring 알고리즘을 적용 할 수 있습니다. 이는 O(n^2) 의 복잡도를 가지며 ```n``` 은 그래프 안의 노드의 갯수입니다. 이는 매우 비싼 알고리즘이 됩니다.

다른 매우 간단한 휴리스틱을 고려해 봅시다. 이 아이디어는 graph를 탐색하는 절차를 시뮬레이션 하고 특정 노드에 의존적인 연산이 몇개 있는지를 세는 것입니다.

![Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/alloc_step.png)

- in-place 최적화 는 현재 연산이 source에만 의존적일 때 수행 가능합니다. (count가 1일때)
- count 0 이 되면 오른쪽 위의 박스에 집어 넣고 메모리를 재사용 할 수 있습니다.
- 새로운 메모리를 할당할때 box 안의 메모리를 재사용 하던지 새로운 메모리를 할당합니다.

***Note:*** 시뮬레이션 동안 메모리는 전혀 할당되지 않습니다. 대신 각 노드마다 어느정도의 메모리가 필요할지를 기록하고 마지막 memory plan에서 공유된 메모리의 최대값으로 메모리를 할당합니다.

## Static vs. Dynamic Allocation

위의 전략은 파이썬과 같은 imperative 언어의 동적 메모리 할당 절차를 정확하게 시뮬레이션 합니다. ```count``` 는 각 memory object를 위한 reference counter 입니다. object들은 reference counter가 0 이 되면 garbage collected 됩니다. 이러한 방법을 통해 우리는 동적 메모리 할당을 시뮬레이션 하고 static allocation plan을 생성합니다. 그렇다면 imperative langague를 사용하여 이러한 그래프를 위한 동적 메모리 할당, 해제를 간단하게 할 수 있을까요? (파이썬과 같은 imperative language를 사용하여 메모리 관리를 파이썬 인터프리터에게만 맡기는 경우를 말하는 듯 하다)

가장 큰 차이점은 static allocation은 단지 한번만 수행되므로 더 복잡한 알고리즘을 사용하여 최적화 할 여지가 있다는 것입니다. 예를 들어 필요한 메모리 블록과 비슷한 크기를 가지는 메모리를 검색 할 수 있습니다. Allocation 또한 일종의 graph로 구성 할 수 있습니다. 우리는 이를 다음 섹션에서 논의합니다. 동적 할당은 빠른 메모리 할당과 garbage collection에 더 많은 부담을 안겨줍니다.

dynamic memory allocation에 의존하려는 사용자를 위한 하나의 팁이 있습니다. 불필요한 객체를 참조하지 마십시요 예를 들어 모든 노드를 리스트로 구성하고 Network object에 저장한다고 가정합시다. 이들 노드들은 절대 dereferenced 되지 않을 것이며 공간이 해제 되지 않아 공간 효율이 떨어집니다. 불행하게도 위의 방법이 뉴럴 네트워크 코드를 짜는 보통의 방법입니다.

#### 정리 

딥러닝 라이브러리에서 메모리 할당 방법 : 직접 동적 할당 을 시뮬레이션 하여 정적 할당을 생성 vs 파이썬과 같은 imperative langauge의 동적 할당에 메모리 관리를 맡김

## Memory Allocation for Parallel Operations

이전 섹션에서는 static allocation plan을 얻기 위해 running procedure를 시물레이션 하는 것에 대해 다뤘습니다. 그러나 병렬 연산을 최적화 하는 것은 resource sharing과 병렬화의 균형을 조절하는데 있어서 다른 문제에 직면합니다. 같은 그래프에 대한 두가지의 allocation plan 을 살펴봅시다.

![Parallel Alloc](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/parallel_alloc.png)

왼쪽 그림을 보면 ```A[2]``` ,```A[5]```  가 메모리를 공유하기 때문에 동시에 병렬적으로 실행 될 수 없다.

두 allocation plan 모두 ```A[2]``` 부터 ```A[5]``` 까지 순차적으로 실행 된다면 적절한 plan 이 됩니다. 그러나 왼쪽의 allocation plan은 추가적인 의존성이 존재하게 되고 이는 ```A[2]``` 와 ```A[5]``` 을 병렬적으로 연산하지 못한다는 의미가 됩니다. 반면 오른쪽의 plan은 병렬적으로 연산이 가능합니다. 병렬 연산을 위해서는 더 세심한 주의가 필요합니다.

## Be Correct and Safe First

Being correct는 우리의 첫번째 원칙입니다. 이 의미는 implicit dependency memory sharing을 고려 하며 연산을 실행한다는 의미입니다. 우리는 execution graph에 이러한 implicit dependency edge를 추가함으로써 수행 할 수 있습니다. 더 간단하게는 execution engine이 [our discussion of dependency engine design](http://mxnet.io/architecture/note_engine.html)에서 설명한 대로 mutation aware 했다면 연산을 queue에 push하고 같은 메모리를 가리키는 variable tag를 쓰게 됩니다.

항상 안전한 memory allocation plan을 수행하게 되며 이 의미는 병렬로 처리 될 수 있는 노드들을 같은 메모리에 할당하지 않는 다는 것입니다. 이는 메모리 감소가 더 필요 할때 바람직 하지 않을 수 있고 하나의 GPU 에서 동시에 실행되는 여러 computing stream을 사용할때 그렇게 많은 이득을 얻지 못합니다. (GPU 하나만 사용 할때는 많은 이득을 얻지 못한다는 뜻인거 같은데..?)

## Try to Allow More Parallelization

이제 우리는 몇가지 최적화를 안전하게 수행 할 수 있습니다. 일반적인 아이디어는 병렬화가 불가능한 노드끼리 메모리를 공유하도록 하는 것입니다. ancestor relationship graph를 구성하고 allocation 동안 ancestor graph에 쿼리를 날립니다. ancestor graph를 구성하기 위한 복잡도는 근사적으로 O(n^2) 이 됩니다. 여기에 color the path in the graph를 통해 몇가지 휴리스틱을 적용 할 수 있습니다. (coloring graph 알고리즘의 휴리스틱을 이용한다는 뜻인가?) 다음과 같은 그림에서 볼 수 있듯이 그래프 안에서의 가장 긴 path를 으면서 같은 path를 같은 색으로 색칠하고 반복하여 계속 진행합니다.

![Path Color](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/memory/graph_color.png)

node에 대한 색상을 모두 얻고 난후 같은 색의 노드끼리만 메모리를 공유하도록 장려 할 수있습니다. 이는 ancestor relationship 보다 강한 제약이 있는 경우이지만 처음 k개의 path를 찾는 경우 복잡도가 O(n) 이 됩니다.

이는 절대적인 솔루션이 아니며 더 나은 접근 방법이 존재 할 수 있습니다.

## How Much Can you Save?

딥러닝에서의 메모리 사용량을 줄이기 위한 알고리즘과 기술들에 대해 살펴봤습니다. 이러한 기술들로 얼마 만큼의 메모리를 절약 할 수 있을까요?

이미 최적화가 수행된 큰 연산들로 이루어진 coarse-grained operation graph의 경우 대략 반정도의 메모리를 절약 할 수 있습니다. Theano와 같은 symbolic 라이브러리에서 사용되는 fine-grained computation network를 최적화 하는 경우 더 많은 메모리를 절약 할 수 있습니다.

이 기사의 대부분의 아이디어는 MXNet의 설계에 영감을 주었습니다. 우리는 다른 시나리오에서 필요한 메모리 용량을 확인 하는데 사용 할 수 있는 메모리 비용 추정 스크립트도 추정합니다. 

이 스크립트에서는 ```forward_only``` 라는 옵션을 호출 할 수 있습니다. 이는 forward pass만을 수행하는 결과를 보여 줍니다. 이 옵션을 사용 할때 다른 옵션에 비해 매우 적은 메모리가 사용됩니다. 이는 forward pass를 수행할때는 더 많은 메모리 공유를 수행 할 수 있기 때문입니다.

여기에 두가지 요점 정리가 있습니다:
- 메모리를 할당하기 위해 computation graph를 사용한다
- 딥러닝 모델의 경우 training 보다 prediction에서 훨씬 적은 메모리가 사용됩니다.






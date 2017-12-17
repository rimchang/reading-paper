# Deep Learning Programming Style

우리가 궁극적으로 성능에 대해 신경을 쓰지만 최적화에 대해 생각을 하기 전에 먼저 동작하는 코드가 필요합니다. 명확하고 직관적인 딥러닝 코드를 작성하는 것은 매우 어려운 일이며 딥러닝을 활용하려는 사람들이 접하는 첫번째 문제는 언어의 문법 그 자체입니다. 이러한 문제를 어렵게 만드는 것은 많은 딥러닝 라이브러리들이 그 들만의 프로그래밍 스타일의 접근법을 가진다는 것입니다.

이 문서에서는 우리는 가장 중요한 2가지의 추상적인 디자인 결정에 초점을 맞춥니다.
1. 수학적 연산을 위해 symbolic, imperative 패러다임을 따를 것인가.
2. 네트워크를 구성하기 위해 보다 큰(추상적인) 연산 혹은 덜 추상적인 atomic 연산들을 사용할 것인가.

우리는 전반적으로 프로그래밍 모델 자체에 초점을 맞출 것입니다. 이러한 프로그래밍 스타일을 결정하는 것은 성능에 많은 영향을 끼치며 이에 대해 다룰 것입니다. 그러나 구현에 대한 세부사항에 대해서는 언급하지 않을 것입니다.

## Symbolic vs. Imperative Programs

만약 Python, c++ 프로그래머라면 이미 명령형 프로그램에 대해 익숙할 것입니다. Imperative-style 프로그램은 그들을 실행함으로써 computation을 수행합니다. 파이썬에서의 대부분의 코드는 명령형이며 다음과 같은 Numpy 코드를 위해 필수적입니다.

```python
    import numpy as np
    a = np.ones(10)
    b = np.ones(10) * 2
    c = b * a
    d = c + 1
```

프로그램이 ``c = b * a``` 구문을 실행할때, 이는 실제로 numerical computation을 수행하게 됩니다.

Symbolic 프로그램은 약간 다릅니다. Symbolic-style 프로그램을 사용하기 위해선, 먼저 함수를 추상적으로 정의합니다. 함수를 정의할때 어떠한 numerical computation이 수행되지 않습니다. placeholder value의 관점에서 추상적인 함수를 먼저 정의하고 그 함수를 컴파일 할 수 있습니다. 그 후 실제의 input이 주어진다면 이 함수를 evaluate 할 수 있습니다. 다음과 같은 예제에서 위의 명령형 프로그램을 symbolic-style 프로그램으로 재작성 합니다.

```python
    A = Variable('A')
    B = Variable('B')
    C = B * A
    D = C + Constant(1)
    # compiles the function
    f = compile(D)
    d = f(A=np.ones(10), B=np.ones(10)*2)
```

위에서 볼 수 있듯이 Symbolic 프로그램에서는 ```C = B * A``` 구문이 실행될때 어떠한 computation이 발생하지 않습니다. 대신에 이 구문은 일종의 _computation graph_ (_symbolic graph_) 을 생성하고 이는 어떠한 computation을 나타냅니다. 밑의 그림은 D를 계산하기 위한 computation graph를 나타냅니다.

![Comp Graph](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/prog_model/comp_graph.png)


대부분의 Symbolic-style 프로그램은 명시적, 혹은 암시적으로 compile 단계를 포함하고 있습니다. 이는 computation graph를 추후에 호출 가능한 function으로 변환하는 것입니다. 위의 예제에서는 ```d = f(A=np.ones(10), B=np.ones(10)*2)``` 라는 구문에서만 numerical computation이 수행됩니다. 이러한 symbolic 프로그램의 특징은 computation graph를 정의하는 것과 실행하는 코드가 철저하게 분리되어 있다는 것입니다. 뉴럴 네트워크를 만들기 위해서 전형적으로 전체의 뉴럴 네트워크를 하나의 computation graph로 나타내게 됩니다.

인기있는 딥러닝 라이브러리인 Torch, Chainer, Minerva는 Imperative 스타일을 채택합니다. Symbolic 스타일의 딥러닝 라이브러리는 Theano, CGT, Tensorflow 등이 있습니다. configuration 파일에 의존하는 CXXNet, Caffe 등도 일종의 Symbolic 스타일 라이브러리로 볼 수 있습니다. 이들 라이브러리는 configuration 파일을 computation graph를 정의하는 내용으로 볼 수 있습니다.

두 프로그래밍 모델의 차이를 이해 했으므로 이제 각각의 장점에 대해 비교를 해보겠습니다.

### Imperative Programs Tend to be More Flexible

파이썬에서 Imperative 스타일의 라이브러리를 사용하고 있다면 이는 Python 에서의 코드를 작성하고 있는 것입니다. 파이썬으로 작성되므로 거의 모든 것들은 필요한 적절한 위치에서만 딥러닝 라이브러리를 사용하여 작성될 수 있습니다. 반면 Symbolic 프로그램으로 작성하면 반복문과 같이 익숙한 파이썬 구문을 사용하지 못 할 수 있습니다. 다음과 같은 Imperative 프로그램을 생각해보고 이를 어떻게 Symbolic 프로그램으로 변환 할지를 생각해 봅시다.

```python
    a = 2
    b = a + 1
    d = np.zeros(10)
    for i in range(d):
        d += np.zeros(10)
```

Symbolic API이 파이썬의 For-loop를 지원하지 않는다면 위의 코드를 변환하는 것은 쉽지 않습니다. 파이썬에서 Symbolic 프로그램을 작성할때 이는 파이썬 코드를 작성하는 것이 아닙니다. 대신 Symbolic API를 이용해 DSL(domaion-specific language) 를 작성하고 있는 것입니다. 딥러닝 라이브러리에서 찾을 수 있는 Symbolic API는 뉴럴 네트워크를 위한 호출 가능한 computation graph를 생성하는 일종의 DSL 입니다.

직관적으로 Imperative 프로그램은 Symbolic 프로그램보다 더 _Native_ 하다고 말할 수 있습니다. 이는 native 언어의 특징을 사용하기 쉽다는 뜻입니다. 예를 들어 computation의 중간에 값을 출력하거나 if문같은 제어문, 루프문을 사용하는 것이 매우 간단합니다.

## Symbolic Programs Tend to be More Efficient

앞서 살펴봤듯이 Imperative 프로그램은 보다 유연하고 host 언어의 프로그래밍 흐름에 맞춰져 있습니다. 그렇다면 왜 수많은 딥러닝 라이브러리가 Symbolic 패러다임을 채택하는 것일까요? 가장 큰 이유는 speed, memory의 효율성 때문입니다. 위에서 보았던 예제를 다시 한번 살펴봅시다.

```python
    import numpy as np
    a = np.ones(10)
    b = np.ones(10) * 2
    c = b * a
    d = c + 1
    ...
```

![Comp Graph](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/prog_model/comp_graph.png)

각각의 요소가 8바이트의 메모리를 차지하는 배열이라고 가정해 봅시다. 이 프로그램을 파이썬 콘솔에서 실행하기 위해 얼만큼의 메모리가 필요할까요

Imperative 프로그램에서는 각 라인마다 메모리를 할당하는 것이 필요합니다. 우리는 위의 연산을 수행하기 위해서 크기가 10인 4개의 배열을 할당해야 만 하며 이는 `4 * 10 * 8 = 320` 바이트가 필요합니다. (a,b,c,d의 4개의 배열이 필요하며 그 사이즈는 모두 10, 각 엘리먼트의 크기의 8 바이트이다.) 반면에 우리가 computation graph를 작성한다면 우리는 사전에 d 배열만을 필요로 한다는 것을 알 수 있으며 d 이전에 할당된 배열을 재사용 할 수가 있습니다. 예를 들어 in-place computation을 수행함으로써 b를 위해 할당된 메모리를 c를 저장하기 위해 사용할 수 있습니다. 또한 c를 위해 할당된 메모리를 d를 저장하기 위해 재사용 가능합니다. (b,c 는 이후에 다시 사용되지 않을 것을 알 수 있음으로 in-place 연산을 수행 가능하다.) 즉 b,c 를 위한 메모리를 할당하지 않아도 되어 메모리 필요량이 절반으로 줄어들게 됩니다. 단지 `2 * 10 * 8 = 160` 바이트만이 필요하게 됩니다.

Symbolic 프로그램은 더 제한되어져 있습니다. 우리가 D 를 컴파일 하라고 요청할때 시스템에게 d의 값만이 필요하다는 정보를 주게 됩니다. 이 경우 계산의 중간값인 c 는 사용자에게 보여지지 않습니다.

Symbolic 프로그램은 in-place 연산을 통해 메모리를 절약할 수 있는 장점을 가지게 됩니다. 하지만 만약 나중에 c에 대한 접근을 필요로 하게 된다면 운이 나쁘게도 실패하게 됩니다. Imperative 프로그램은 computation의 중간에 가능한 모든 요구사항에 대해 준비를 해야 한다면 더 나은 방법입니다. 만약 파이썬 콘솔에서 Imperative 코드를 실행한다면 중간 변수를 언제든지 사용 할 수 있게 됩니다.

_물론 imperative 프로그램에서 garbage collection을 통해 메모리가 재사용 될 수가 있습니다. 하지만 imperative 프로그램은 가능한 모든 요구사항을 준비해야 하며 이는 수행할 수 있는 최적화에 제한을 주게 됩니다. 이는 gradient 계산과 같은 것들의 최적화를 제한시키며 이는 다음 섹션에서 다루게 됩니다._

Symbolic 프로그램은 operation folding 이라는 다른 종류의 최적화를 수행 할 수 있습니다. 위의 예제로 돌아와 다음과 같은 곱셈, 덧셈의 연산을 하나의 연산으로 folding 이 가능합니다. 만약 folding graph가 GPU 프로세서에서 연산이 수행된다면 두개가 아닌 하나의 GPU kernel 만이 실행 될 것입니다. CXXNet, Caffe 같은 라이브러리에서는 이러한 folding을 수행하여 라이브러리를 최적화 합니다. operation folding은 연산의 효율성을 향상 시키게 됩니다.

![Comp Graph Folded](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/prog_model/comp_graph_fold.png)

imperative 프로그램에서는 중간 값들이 추후에 사용될 수 있음으로 이러한 operation folding을 수행 할 수 없습니다. symbolic 프로그램에서는 전체 computation graph가 주어진다면 어떤 값이 필요하고 필요하지 않은지가 분명함으로 이러한 operation folding이 수행 가능합니다.

## Case Study: Backprop and AutoDiff

 이 섹션에서는 auto differenctiation, backpropagation의 관점에서 두가지 프로그래밍 모델을 비교합니다. 미분은 모델을 훈련시키는 메커니즘이기 때문에 딥러닝 에서 매우 중요합니다. 모든 딥러닝 라이브러리에서 loss function을 정의합니다. loss function은 우리가 원하는 출력과 모델의 출력이 얼마나 먼지를 측정하는 함수입니다. 전형적으로 training example을 통과 시키며 각 스탭마다 loss function을 최소화 하도록 모델의 파라미터를 업데이트 합니다. 파라미터를 업데이트 하기 위한 방향을 결정하기 위해 파라미터와 관련된 loss function의 미분을 계산해야 합니다.
 
 과거에는 누군가 새로운 모델을 정의 할때 마다 수작업으로 미분을 계산 했어야 했습니다. 이러한 수학은 합리적으로 타당하지만 매우 복잡한 모델의 경우에는 수많은 시간이 소모되며 지루한 작업이 됩니다. 현대의 딥러닝 라이브러리는 연구자, 실무자 의 작업을 auto differentiation을 통해 훨씬 쉽게 만들어 줍니다.
 
 imperative, symbolic 프로그램 모두 gradient 계산을 수행 할 수 있습니다. 이제 각각이 어떻게 자동 미분을 수행하는지를 알아봅시다.
 
 먼저 imperative 프로그램 부터 시작합니다. 다음과 같은 코드는 자동 미분을 수행하는 파이썬 코드입니다.
 
 ```python
    class array(object) :
        """Simple Array object that support autodiff."""
        def __init__(self, value, name=None):
            self.value = value
            if name:
                self.grad = lambda g : {name : g}

        def __add__(self, other):
            assert isinstance(other, int)
            ret = array(self.value + other)
            ret.grad = lambda g : self.grad(g)
            return ret

        def __mul__(self, other):
            assert isinstance(other, array)
            ret = array(self.value * other.value)
            def grad(g):
                x = self.grad(g * other.value)
                x.update(other.grad(g * self.value))
                return x
            ret.grad = grad
            return ret

    # some examples
    a = array(1, 'a')
    b = array(2, 'b')
    c = b * a
    d = c + 1
    print d.value
    print d.grad(1)
    # Results
    # 3
    # {'a': 2, 'b': 1}
```

이 코드에서 각 array object는 grad function을 클로저 형태로 포함하고 있습니다. ```d.grad``` 구문을 실행할때 입력의 grad function을 재귀적으로 호출하고 gradient value를 backprop 하여 각 인풋에 대한 gradient를 리턴 하게 됩니다.

이는 약간 복잡해 보일 수 있습니다. Symbolic 프로그램에서의 gradient 계산을 고려해 봅시다. 다음과 같은 프로그램은 위의 코드를 symblic 하게 수행합니다.

```python
    A = Variable('A')
    B = Variable('B')
    C = B * A
    D = C + Constant(1)
    # get gradient node.
    gA, gB = D.grad(wrt=[A, B])
    # compiles the gradient function.
    f = compile([gA, gB])
    grad_a, grad_b = f(A=np.ones(10), B=np.ones(10)*2)
```

D의 grad function은 backward computation graph를 생성하여 gradient node ```gA, gB``` 를 반환합니다. 이는 다음 그림에서의 빨간색 노드에 대응합니다.

![Comp Graph Folded](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/prog_model/comp_graph_backward.png)


Imperative 프로그램과 symbolic 프로그램은 동일한 일을 하게 됩니다. imperative 프로그램은 암묵적으로 grad 클로저를 활용하여 backward computation을 저장하게 됩니다. ```d.grad``` 를 호출하게 되면 gradient를 계산하기 위해 ```d(D)``` 에서 시작하여 모든 그래프를 backtrack 하여 결과를 반환하게 됩니다.

symbolic, imperative 프로그래밍에서의 gradient 계산은 같은 패턴을 따릅니다. 그렇다면 무엇이 다른 것일까요? imperative 프로그램에서는 모든 가능한 요구사항을 준비해야 한다는 것을 기억해봅시다. 만약 자동 미분을 지원하는 어떤 배열 라이브러리를 만들었다고 하면 computation을 수행할때 모든 클로저를 메모리에 저장해야 합니다. 이는 ```d``` 변수에 의해 클로저의 형태로 모든 변수가 참조되기 때문에 어떠한 변수도 garbage-collected 될 수 없습니다. 

만약 ```d``` 변수의 값만을 계산하고 gradient 값을 계산할 필요가 없다고 해봅시다. symbolic 프로그래밍에서는 `f=compiled([D])` 를 통해 선언 합니다. 이는 computation의 경계를 선언해 주며 시스템에게 오직 forward pass만이 필요하다는 것을 알려 주게 됩니다. 이 결과 시스템은 이전 결과의 메모리를 해제해주고 인풋과 출력 사이의 메모리를 공유하도록 합니다.

n 레이어의 딥 뉴럴 네트워크를 실행한다고 상상해 봅시다. 만약 backward pass가 필요 없이 forward pass 만을 실행 하는 경우 중간 레이어의 값을 저장하기 위해 n개의 할당이 아닌 오직 2개의 복사본 만 할당하면 됩니다. (이전 레이어와 다음 레이어 인가?) imperative 프로그램에서는 gradient를 계산할지도 모르는 요구사항을 준비하기 위해 중간 레이어의 값을 모두 저장해야만 하고 n개의 복사본에 대한 할당을 수행해야 합니다. 

최적화의 수준은 무엇을 할 수 있는지에 따라 제한되어져 있습니다. Symbolic 프로그램은 graph를 컴파일 할 때 이러한 제한을 명확히 명시합니다. imperative 프로그램은 광범위한 요구사항들을 준비해야만 합니다. Symbolic 프로그램은 컴파일 단계에서 사용자가 무엇을 원하고 원하지 않는지에 대해 앎으로써 본질적인 장점을 가지고 있습니다.

이러한 비슷한 제한을 사용하기 위해 imperative 프로그램을 활용하는 방안이 있습니다. 앞의 예제에 대한 하나의 해결책은 context variable을 도입하는 것입니다. no-gradient context 변수를 도입함으로써 gradient 계산을 하지 않도록 할 수 있습니다.

```python
    with context.NoGradient():
        a = array(1, 'a')
        b = array(2, 'b')
        c = b * a
        d = c + 1
```

이러한 context 변수의 도입은 imperative 프로그램에 제약을 강제 할 수 있지만 효율성을 줄이게 됩니다. 

그러나 위의 예제에서도 여전히 가능한 모든 요구사항에 대해 준비를 해야 하며 이는 forward pass 에서 메모리를 재사용하기 위한 in-place 연산을 수행할 수 없습니다. (이는 GPU 메모리 사용을 줄이기 위해 보통 사용되는 트릭입니다.) 앞에서 논의했던 기술은 명시적인 backward pass를 생성합니다. Caffe, CXXNet 과 같은 라이브러리는 같은 그래프에 대해 암묵적인 backprop을 수행합니다. 이러한 암묵적인 backprop의 경우도 이 섹션에서 논의한 context 변수의 도입을 적용 가능합니다.

Caffe, CXXNet과 같은 configuration-file-based 라이브러리는 두가지의 요구사항에 맞춰서 설계되었습니다. 이는 각 레이어의 activation map을 얻거나 파라미터에 대한 gradient를 얻을 수 있도록 설계되었습니다. 이들 라이브러리는 같은 문제를 가지고 있습니다. 라이브러리가 지원해야 할 일반적인 연산이 많을 수록 최적화(memory sharing) 할 수 있는 것이 줄어들게 됩니다.

이러한 restriction vs flexibility의 trade-off는 대부분의 경우 나타나게 됩니다.

## Model Checkpoint

모델을 저장하고 나중에 다시 로드할 수 있는 것은 매우 중요합니다. 작업한 결과를 저장하기 위한 여러가지 방법들이 존재합니다. 일반적으로 뉴럴 네트워크를 저장하기 위해서는 뉴럴네트워크의 구조에 대한 configuration과 뉴럴 네트워크의 파라미터를 2가지를 저장해야만 합니다.

저장된 configuration을 확인 할 수 있는 것은 symbolic 프로그램의 장점입니다. symbolic graph를 구성하는 단계에서 어떠한 computation을 수행하지 않기 때문에 computation graph를 직접적으로 serialize 하는 것이 가능하며 다시 불러 오는 것도 가능합니다. 이는 추가적인 연산 없이 configuration을 저장하는 것을 가능하게 합니다.

```python
    A = Variable('A')
    B = Variable('B')
    C = B * A
    D = C + Constant(1)
    D.save('mygraph')
    ...
    D2 = load('mygraph')
    f = compile([D2])
    # more operations
    ...
```

imperative 프로그램은 실행에 의해 computation이 정의되기 때문에 작성한 코드 그 자체를 configuration으로 저장해야만 하며 imperaitve language를 위한 다른 configuration layer를 추가해야만 합니다. 

## Parameter Updates

대부분의 symbolic 프로그램은 data flow(computation) graph 입니다. Data flow graph이 수행될 computation을 기술하게 됩니다. 이는 파라미터 업데이트를 기술하기 위해 graph를 사용하는 방법은 분명하지 않습니다. 이는 파라미터 업데이트가 data flow의 개념이 아닌 다른 형태로 수행되기 때문입니다. 대부분의 symbolic 프로그램은 persistent state를 업데이트하기 위한 특별한 update statement를 도입합니다.

일반적으로 imperative 프로그램에서 파라미터 업데이트를 작성하는 것이 더 쉽습니다. 특히 서로가 관련되어 있는 여러번의 파라미터 업데이트를 수행해야 할때 쉽습니다. symbolic 프로그램에서는 update statement는 사용자가 호출할때 실행 됩니다. 이런 관점에서 대부분의 symbolic 딥러닝 라이브러리는 gradient 를 계산하는데는 symbolic 접근 방법을 통해 수행하지만 파라미터 업데이트를 하기 위해 다시 imperative 접근 방식을 통해 수행합니다.

## There Is No Strict Boundary

두가지 프로그래밍 스타일을 비교할때 우리가 언급한 것들이 엄격하게 참은 아닙니다. imperative 프로그램을 더 symblic 프로그램처럼 만들거나 그 반대가 충분히 가능합니다. 그러나 두 구조는 딥러닝 라이브러리의 차이를 이해하기 위한 좋은 추상화 입니다. 우리는 프로그래밍 스타일 간에 명확한 경계가 없다고 결론을 내릴 수 있습니다. 예를 들어 파이썬의 imperaitve 프로그램을 컴파일 하기 위해 just-in-time(JIT) 컴파일러를 사용할 수 있으며 이는 symbolic 프로그램이 가지는 gloval information의 장점들을 제공합니다.

## Big vs. Small Operations

딥러닝 라이브러리를 설계할때 중요한 것 중 하나는 어떤 연산 까지를 지원할지에 대한 것입니다. 일반적으로 대부분의 딥러닝 라이브러리가 지원하는 2종류의 연산들이 있습니다.

- Big operation - FullyConnected, BatchNormalize 와 같은 뉴럴네트워크 레이어에 대한 연산
- Small operation - matrix multiply, element-wise addition과 같은 수학적 함수

CXXNet, Caffe 는 레이어-레벨의 연산들을 지원하며 Theano, Minerva 는 보다 하위 레벨의 연산들을 지원합니다.

## Smaller Operations Can Be More Flexible

보다 작은 연산들을 활용하여 큰 연산들을 작성하는 것은 매우 자연스럽습니다. 예를 들어 sigmoid unit은 division, addition, exponentiation 으로 구성되어져 있습니다..


```python
    sigmoid(x) = 1.0 / (1.0 + exp(-x))
```

하나의 블록을 작성하기 위해 작은 연산들을 사용함으로써 원하는 거의 모든것을 표현 할 수 있습니다. 만약 CXXNet, Caffe 에 대해 익숙하다면 이러한 연산들은 작다는 것을 제외하곤 레이어와 다를 것이 없습니다.


```python
    SigmoidLayer(x) = EWiseDivisionLayer(1.0, AddScalarLayer(ExpLayer(-x), 1.0))
```

Sigmoid 레이어는 세개의 레이어로 구성되어져 있으며 각각의 레이어는 forward, backward function이 정의되어져 있습니다. 작은 연산을 사용 함으로써 새로운 레이어를 빠르게 작성 할 수 있습니다.

## Big Operations Are More Efficient

Sigmoid 레이어를 직접 정의하려면 하나의 레이어가 아닌 3개의 레이어가 필요합니다.

```python
    SigmoidLayer(x) = EWiseDivisionLayer(1.0, AddScalarLayer(ExpLayer(-x), 1.0))
```

위와 같은 코드는 computation, memory 오버헤드를 만들게 되며 이는 하나의 레이어를 사용 했다면 최적화 가능한 것들입니다.

CXXNet, Caffe 와 같은 라이브러리는 다른 접근 방법을 사용합니다. BatchNormalization, SigmoidLayer와 같은 보다 큰 연산들을 지원하기 위해 각 레이어는 하나 혹은 몇개의 CUDA 커널을 통해 수작업으로 디자인 되게 됩니다. 이는 이러한 구현체들을 보다 효율적으로 만듭니다.

## Compilation and Optimization

작은 연산들도 최적화 될 수 있을까요? 물론 이들도 가능합니다. 컴파일 엔진의 최적화 부분을 살펴봅시다. computation graph를 수행하기 위해 2가지의 최적화 방법이 수행 가능합니다.

- 중간의 연산들을 재활용 한다. Memory allocation optimization
- sigmoid와 같은 sub-graph 패턴을 찾고 이들을 보다 큰 operation kernel로 합친다. Operator fusion

메모리 할당을 최적화 하는 것은 작은 compuation graph에만 제한되지 않습니다. 이를 보다 큰 graph에도 적용 가능 합니다. 그러나 CXXNet, Caffe와 같은 라이브러리는 컴파일 단계가 없음으로 메모리 할당 최적화가 필요하지 않을 수 있습니다. 그러나 CXXNet, Caffe와 같은 라이브러리에는 어떻게 보면 멍청한 컴파일 단계가 존재하며 이는 레이러를 고정된 forward, backprop 실행 계획으로 번역하며 각 연산들을 하나씩 실행하게 됩니다.(?? 뭔소리지 CXXNet, Caffe와 같은 라이브러리엔 멍청한 컴파일 단계가 존재한다는 소리인가?)

작은 computation graph의 경우 이러한 최적화는 성능에 큰 영향을 미치게 됩니다. 연산들이 매우 적기 때문에 수많은 최적화 가능한 sub-graph 패너들이 존재하게 됩니다. 또한 최종적으로 생성된 연산들은 enumerable 하지 않을 수 있음으로 커널을 다시 컴파일 하는 것이 필요합니다. 이는 big operation 라이브러리(CXXNet, Caffe?)에서 미리 컴파일된 일정 양의 커널들을 사용하는 것과 반대입니다. 이로 인해 작은 연산들을 지원하는 symbolic 라이브러리는 컴파일 오버헤드가 생기게 되며 이를 지원해야 하는 라이브러리는 컴파일 단계에서의 최적화를 위한 엔지니어링 오버헤드가 생기게 됩니다.

symblic vs imperative 경우와 마찬가지로 bigger operation 라이브러리는 공통적으로 사용되는 레이어에 대해 어떠한 제한을 요구함으로써 sub-graph 매칭을 수행하게 됩니다. 이는 컴파일 오버헤드를 사람이 생각 할 수 있도록 하게 만들고 그리 나쁘지 않습니다.

## Expression Template and Statically Typed Language

작은 연산들을 작성하고 이들을 정의하는 작업이 필요합니다. Caffe 와 같은 라이브러리는 이와 같은 bigger block을 작성하기 위해 hand-crafted kernel을 사용합니다. (cuda로 실제로 짜버린다는 뜻인듯?) 그렇지 않으려면 파이썬을 사용하여 더 작은 연산을 작성해야 합니다. (혹은 파이썬 함수로 짜버린다.)
 
 꽤 잘 작동하는 세번째 방법이 있습니다. 이는 expression template라고 불립니다. 기본적으로 template programming을 사용하여 컴파일 단계에서의 expression tree를 사용하여 일반적인 kernel을 생성하게 됩니다. CXXNet 에서는 expression template 를 확장하여 사용 할 수 있게 만듦으로써 hand-crafted kernel과 성능이 같은 보다 짧고 간단한 코드를 사용 할 수 있게 합니다.
 
 expression template과 파이썬 kernel을 생성하는 것의 차이점은 expression evaluation은 C++ 의 컴파일 시간에 수행되므로 추가적인 실행시간 오버헤드가 없다는 것입니다. 이는 원칙적으로 template를 지원하는 모든 언어에 적용이 가능하지만 우리는 C++에서만 이러한 것들을 보았습니다.
 
 expression template 라이브러리는 C++ 사용자가 작은 연산을 조합하여 큰 연산을 정의할 수 있게 함으로써 python 연산과 hand-crafted big kernel의 중간 정도의 추상화 수준을 보입니다. 이는 충분히 고려할 만한 옵션입니다.
 
 ## Mix the Approaches
 
 지금까지 프로그래밍 모델에 대해 비교해 보았습니다. 그렇다면 어떤 모델을 선택해야 할까요 이를 살펴보기 전에 풀어야 할 문제가 무엇인지에 따라 우리의 비교 자체가 별로 큰 영향이 없을 수 있다는 것을 알아야 합니다.
 
 [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) 를 기억해 보자면 풀어야 할 문제에서 별로 성능에 중요하지 않는 부분을 최적화 한다면 원하는 만큼의 최적화 성능을 얻지 못할 것입니다.
 
 지금까지 살펴 보았듯이 효율성, 유연성, 엔지니어링 복잡성 간의 trade-off가 일반적으로 존재합니다. 풀어야 하는 문제에 따라 적합한 프로그래밍 스타일이 정해지게 됩니다. 예를 들어 imperative 프로그램은 파라미터 업데이트에 좀더 효율적이며 symbolic 프로그램은 gradient 연산에 효율적입니다.
 
 우리는 이러한 접근 방식을 혼합하는 것을 지지합니다. 때때로 우리가 유연해지기 원하는 부분이 성능에 치명적이지 않을 수 있습니다. 이런 경우 더 유연한 인터페이스를 유지하기 위해 약간의 효율을 버리는 것도 괜찮습니다. 머신러닝에서는 여러 방법론을 결합하는 것이 하나만 사용하는 것보다 더 나은 성능을 보입니다.
 
 두가지 프로그래밍 모델을 올바르게 결합 할 수 있다면 단일 프로그래밍 모델을 사용 할 때 보다 더 나은 결과를 얻을 수 있습니다. 이 섹션에서는 이를 수행하는 방법에 대해 설명합니다.
 
 ## Symbolic and Imperative Programs
 
 symbolic, imperative 프로그램을 혼합하는 2가지 방법이 존재합니다.
 
 - symbolic 프로그램을 기본적으로 작성하고 imperative 프로그램으로 호출한다
 - imperative 프로그램을 기본적으로 사용하고 부분을 symbolic 프로그램으로 작성한다
 
 파라미터 업데이트는 imperatively 하게 수행하고 gradient 연산을 symbolic 프로그램으로 작성하는 것이 도움이 된다는 것을 발견했습니다.
 
 파이썬에서의 symbolic 라이브러리는 파이썬이 imperative 하므로 이미 일종의 mix-program입니다. 예를 들어 다음과 같은 코드는 symbolic graph와 imperative 라이브러리 numpy를 혼합한 경우입니다.
 
```python
    A = Variable('A')
    B = Variable('B')
    C = B * A
    D = C + Constant(1)
    # compiles the function
    f = compile(D)
    d = f(A=np.ones(10), B=np.ones(10)*2)
    d = d + 1.0
```

이 symbolic graph는 imperatively 실행될 수 있는 함수로 컴파일 됩니다. 컴파일 되는 내부는 유저에게는 보여지지 않습니다. 이것은 내부적으로 C++ 프로그램을 작성하고 파이썬으로 Api를 제공하는 것과 같습니다.

파라미터가 GPU 메모리에 있기 때문에 Numpy를 imperative component로 사용하고 싶지 않을 수 있습니다. (Numpy는 GPU 연산을 지원하지 않기 때문에) 이런 경우 컴파일된 symbolic function과 상호작용 할 수 있거나 symbolic 프로그램의 실행에서 매우 제한된 양의 updating syntax 만을 제공하는 GPU-compatible imperative 라이브러리를 지원하는 것이 좋은 선택이 될 수 있습니다. 

## Small and Big Operations

작은 연산과 큰 연산들을 결합해야할 이유가 있을 수 있씁니다. loss function을 변경해야 하거나 기존에 존재하는 구조에 사용자 정의 레이어를 추가하는 프로그램을 생각해 봅시다. 기존의 것들은 큰 연산들을 사용하고 새로운 것을 정의하기 위해 작은 연산들을 사용 할 수 있습니다.

Amdahl 의 법칙에 따라 새로운 요소들이 항상 compuation bottleneck의 원인은 아닙니다. 기존이 요소들이 큰 연산들로 인해 이미 최적화 되어 있기 때문에 추가적인 사용자 정의 operation은 최적화를 하지 않거나 제한된 최적화만 수행하여도 괜찮습니다.

## Choose Your Own Approach

이 문서에서는 딥러닝 프로그래밍 을 위한 여러가지 접근방법엗 해나 비교를 수행했습니다. 우리는 각각의 효율성, 사용성에 대해 비교하였으며 이들이 상충되는 개념이 아님을 발견했습니다. 이들 접근 방식중 하나를 선택하거나 결합함으로써 재미있고 지능적인 딥러닝 라이브러리를 만들 수 있습니다.


 
 

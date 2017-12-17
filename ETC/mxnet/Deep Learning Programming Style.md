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



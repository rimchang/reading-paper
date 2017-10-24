# Tutorial: Implementing Layer-Wise Relevance Propagation

http://www.heatmapping.org/tutorial/

### 1. Prerequisites

이 튜토리얼을 위해선 Python, Numpy, PIL이 설치되어 있어야 합니다. 그 후 현재 작업 폴더에 `modules.py` 와 `utils.py` 를 추가하십시요 이는 뉴럴네트워크의 간단한 구현과 데이터를 시각화 하기 위한 메소드 들입니다.

```python
import numpy, copy
import modules, utils
```
또한 MNIST 데이터셋을 다운로드 받는 것이 필요합니다. 이 튜토리얼에서 우리는 두가지 학습된 fully connected neural network와 convolutional neural network를 사용합니다. 또한 이 introduction을 읽기 전에 더 나은 이해를 위해 Depp Taylor decomposition에 대한 글을 읽기를 추천합니다.

### 2. Relevance Propagation for Fully-Connected Networks

우리는 먼저 300, 1000의 크기를 가진 두가지 hidden layer를 가지고 Relu를 activation으로 사용하는 fully connected 네트워크를 고려합니다. 이 네트워크는 밑과 같은 그림으로 나타낼 수 있습니다.

네트워크의 각 레이어의 bias, weight는 mlp.tar.gz 에서 주어집니다. 이 네트워크는 data augmentation을 사용하여 100 epoch를 달성하였고 MNIST에서 1.05%의 error를 달성했습니다. 이러한 파라미터를 가진 네트워크는 밑과 같이 인스턴스화 될 수 있습니다.

```python
nn = modules.Network([
    modules.Linear('mlp/l1'),modules.ReLU(),
    modules.Linear('mlp/l2'),modules.ReLU(),
    modules.Linear('mlp/l3'),
])
```

![mlp](heatmapping.org/tutorial/mlp.png)

뉴럴네트워크가 트레이닝에 사용된 softmax layer가 제거 된것을 주목하세요 그러므로 네트워크의 출력은 unnormalized class log-prob 에 대응됩니다. 이것이 우리가 input pixel의 관점에서 설명하려고 하는 값입니다.

### 2.1 Implementing the propagation procedure

relevance propagation에 필수적인 backward propagation 과정을 구현하기 위해 module.py의 각 클래스를 확장하고 propagation을 위한 메소드를 만들것입니다.

우리는 먼저 network의 가장 위부터 global propagation을 구현합니다. 이 Network 클래스는 원래의 Network 클래스를 확장한 것이고 propagation을 수행하는 "relprop" 메소드를 추가합니다.

```python
class Network(modules.Network):
    def relprop(self,R):
        for l in self.layers[::-1]: R = l.relprop(R)
        return R
```

이 메소드는 네트워크의 layer를 역순으로 방문하며 각 layer의 propagation 함수를 호출합니다. 그리고 각각의 호출된 함수의 결과는 이전 layer의 입력으로 가게 됩니다. 이 코드는 원래의 gradient propagation과 매우 비슷합니다.

relevance propagation 연산은 표준적인 gradient backprop 보다 더 세분화 된 수준에서 연산합니다. 특히 Relu, liner의 쌍을 layer로 취급합니다. gradient propagation은 일반적으로 Relu, linear를 분리하여 취급합니다. 뉴럴네트워크의 구현을 바꾸는 하나의 옵션은 이 두 layer를 하나로 합치는 것입니다. 대안적으로 우리는 Linear + Relu layer를 생각하는데 transformation이 없는 형태입니다.전체의 propagation 은 밑에 기술되었습니다.

```python
class ReLU(modules.ReLU):
    def relprop(self,R): return R
```

linear-detection layer 에 대한 propagation의 선택은 input domain에 의존합니다. z+ 규칙은 양수인 input domain에 적용합니다 (hidden layer가 이전의 레이어로 부터 Relu feature를 받는 경우) z-B 규칙은 box-constrained domain에 적용합니다 (input pixel value를 받는 input layer)
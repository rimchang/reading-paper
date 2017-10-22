# Matrix Multiplication with CUDA | A basic introduction to the CUDA programming model

http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf


### Ch1 Matrix Multiplication

### 1.1 Overview

계산의 속도 향상을 위해 그래픽 카드의 병령성을 이용하는 일이 많아지고 있다. 이를 하기 위한 하나의 플랫폼은 NVIDIA's cuda 이다. 우리는 CUDA 환경에서의 gpu 연산의
기본을 소개하기 위한 예제로 매트릭스 곱을 사용한다. 이 튜토리얼은 c 프로그래밍에 익숙한 학생들을 위한 튜토리얼이며 다른 배경지식은 필요하지 않다.

이 모듈의 목적은 학생들이 그래픽 카드에서의 병렬 연산을 어떻게 하는지를 보여주는 것이고 광범위한 병렬 환경에서의 코드의 실행을 어떻게 새앆해야 하는지에 대한 아이디어를 줄 것이다.


### 1.2 Counting path in graphs

매트릭스는 숫자들의 사각형 배열이다. 더이상의 설명은 필요하지 않습니다. 이러한 간단함에도 불구하고 이들은 가장 과학적 연산에서의 가장 유용하고 근본적인 수학적 객체입니다. 컴퓨터그래픽스, 방정식 풀이, DNA 시퀀스 비교, 등등 
에서 매트릭스를 다양하게 이용하고 있습니다. 수학적 객체로써의 매트릭스는 더하기, 빼기, 곱하기, 나누기 등을 가진 객체입니다. 우리는 특히 매트릭스 곱에대해서 다룰 것입니다.

만약 매트릭스 곱에대해 처음 보는 것이라면 이해를 도울만한 몇가지를 살펴보는 것이 좋을 것입니다. 그렇지 않으면 이는 매우 이상하게 보일 수 있습니다. 우리가 살펴볼 것은 네트워크에서의 경로의 수를 세는 것입니다. 이러한 질무은
transportation network, DNA sequence 비교, DNA 염기서열, 복잡한 네트워크에서의 노드의 중심성을 계산하는데 사용됩니다.
만약 graph의 vertice가 어떤 state의 집합이고 edge가 state의 전이확률이라면 이 그래프 모델은 Markov chain입니다. 그리고 이러한 것들의 새로운 응용이 많은데. 연령 층화 인구 모델, 마약 디자인? 등 이 있습니다. 그래서 우리는 네트워크의 path를 세는 알고리즘을 살펴봅시다.

우리는 위와 같은 그래프의 path에 대한 질문을 할것입니다. 이 그래프에서 C-D-F-G-J와 같은 C에서 J로의 하나의 4의 길이를 가지는 path가 있습니다. 또한 C에서 J로의 가장 짧은 C-E-H-J 라는 3의 길이를 가지는 PATH가 존재합니다. (여기서 length라는 것은 거쳐가는 edge의 수이고 이는 거친 vertice의 수보다 1이 작습니다.)
우리는 여기에 3의 길이를 가지는 path는 다른 것이 없다는 것을 확인 할 수 있습니다. 우리는 C에서 J로 가는 특정 길이의 path가 몇개일지에 대한 질문을 할 수 있습니다.

이를 손으로하는 것은 별로 어렵지 않지만 우리가 systematic 하게 한다면 더 쉽게 할 수 있습니다. systematic 하게 하는 것은 우리가 어떤 것도 빠트리지 않고, 두번 세지 않앗다는 것을 보장하게 해줍니다. 우리가 vetex C에서 시작할때의 첫번째 step을 고려해 봅시다. vertex C에서의 첫번째 step은 우리가 A,B,D,E 로 갈 수 있습니다. 
이들 vetices에서 우리가 job을 끝내기 위해선 어떤것이 필요할까요 우리가 C에서 j로가는 4길이의 path를 찾고 있으므로 우리는 A,B,D,E에서의 J로 가는 3의길이의 PATH를 찾는 것이 필요합니다. 피규어1.2 에서는 모든 쌍의 길이 3의 path의 갯수를 나타냅니다.


Figure2  

| from to | A | B | C | D | E | F | G | H | I | J |
|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|A|2|6|6|3|1|3|1|2|1|0|  
|B|6|6|8|7|3|9|3|3|2|3|  
|C|6|8|4|8|5|4|3|2|2|1|  
|D|3|7|8|4|2|8|3|3|2|3|  
|E|1|3|5|2|0|3|2|5|3|1|  
|F|3|9|4|8|3|6|10|11|10|4|  
|G|1|3|3|3|2|10|8|9|8|8|  
|H|2|3|2|3|5|11|9|4|4|9|   
|I|1|2|2|2|3|10|8|4|4|8|  
|J|0|3|1|3|1|4|8|9|8|4|  

우리는 이제 다음과 같은 추론이 가능합니다. 우리가 vertex A로 부터 첫번째 스탭을 가고 있다면, 여기에 가능한 0 way만 있다면 job을 마친다. 테이블을 살펴보면 A에서 j로가는 3 길이의 path는 존재 하지 않습니다. 우리가 B에서 첫번째 스탭을 한다면 가능한 3가지 방법이 있습니다. 테이블을 보면 3길이의 path가 존재하는건 3개가 있습니다. 비슷하게 D에서부터 시작하면 3가지 방법, E에서 부터 시작하면 한가지 방법이 있습니다. 
다 더함으로써 우리는 C에서 j로 가는 4길이의 path는 총 7개가 있습니다ㅣ.

이런 계산을 하는 다른 방법이 있습니다. C에서 시작하여 다른 vertiec로 가는 길이 1의 path를 가진 1x10 vector가 있다고 해봅시다. 이 벡터는 [1,1,0,1,1,0,0,0,0] 과 같은 값을 가질 것이고 이는 [A,B,C,D,E,F,G,H,I,J] 의 순서입니다. 각 vertice에서 J로 가는 테이블1.2의 맨 오른쪽 열과 이 vector와 곱을 생각 할 수 있습니다.
피규어1.3 에서의 수평 vector는 C에서 부터 가능한 길이 1의 path의 갯수를 나타내며 수직의 vector는 각 vertice에서 J로 가는 3길이의 PATH의 길이를 나타냅니다. 여기서 대각선의 선은 곱해져야할 요소를 나타내며 맨 오른쪽에 그 결과가 나타납니다. 그 결과를 다 더해서 마지막의 7개의 path라는 계산을 볼 수 있습니다. 여기서 다른 것은 "0" entry는 C와 어떤 연결도 가지고 있지 않다는것입니다. 예를 들어 수평, 수직 vector의 entry의 의미는 C to A의 path가 1개 있지만 A to J 로의 3길이의 path가 0개 있다는 의미 입니다. 그다음 요소의 의미는 1길이의 C to B의 path가 1개 있고 B to J의 길이3의 path가 3개 있다는 뜻입니다.

이러한 것들이 matrix 곱의 기본 구조입니다. 우리는 row entry와 column entry를 곱하고 그 결과를 더합니다. 

피규어1.4

| from to | A | B | C | D | E | F | G | H | I | J |
|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|A| .| .| .| .| .| .| .| .| .|.|  
|B| .| .| .| .| .| .| .| .| .| .|  
|C| .| .| .| .| .| .| .| .| .| 7|  
|D| .| .| .| .| .| .| .| .| .| .|  
|E| .| .| .| .|.| .| .| .| .| .|  
|F| .| .| .| .| .| .| .| . .| .| .|  
|G| .| .| .| .| .| .| .| .| .| .|  
|H| .| .| .| .| .| ..| .| .| .| .|   
|I| .| .| .| .| .| .| .| .| .| .|  
|J|.| .| .| .| .| .| .| .| .| .|  

우리가 위의 테이블을 모두 완성하고 싶다고 해봅시다. 무엇을 포함해야 할 까요 우리는 vetice의 1길이의 path에 대한 것을 모두 알아야 합니다. 우리가 이 vector를 모두 stack 할 수 있다면 이들은 피규어 1.5처럼 10x10 테이블의 형태가 될 것입니다.

피규어 1.5

from/to|A|B|C|D|E|F|G|H|I|J
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- 
A|0|1|1|0|0|0|0|0|0|0
B|1|0|1|1|0|1|0|0|0|0
C|1|1|0|1|1|0|0|0|0|0
D|0|1|1|0|0|1|0|0|0|0
E|0|0|1|0|0|0|0|1|0|0
F|0|1|0|1|0|0|1|1|1|0
G|0|0|0|0|0|1|0|1|1|1
H|0|0|0|0|1|1|1|0|0|1
I|0|0|0|0|0|1|1|0|0|1
J|0|0|0|0|0|0|1|1|1|0

이 테이블은 2가지 로 생각할 수 있습니다. 첫번째로 매트릭스의 각 숫자는 각 vertice에 해당하는 길이 1의 path의 갯수를 나타냅니다. 다르게는 각 vertice의 쌍에 연결된 edge의 갯수로 생각 할 수 있습니다. 1이면 연결이 존재하고, 0이면 서로간의 연결이 없는
2번째 방법으로 이 테이블을생각하면 이는 adjacency matrix라고 불립니다. 


### 1.3 Multiplying Matrices

이제 매트릭스 곱에 대해서 설명할 차례가 왔습니다. 우리는 밑과같은 테이블을 매트릭스로 다시 부를 것입니다. 밑의 두 매트릭스의 곱은 각 vertice의 쌍의 4길이의 path의 갯수를 찾는 것으로 볼 수 있습니다.

곱해진 매트릭스에서 맨 오른쪽 위의 entry는 A to J 로의 4길이의 path의 갯수입니다. 가능한 path는 A-B-F-I-J 가 될 수 있으며 다른 것은 A-C-E-H-J가 될 수 잇습니다. 이 들의 총 갯수를 얻기 위해서 첫번째 매트릭스에서의 처음 row와 두번째 매트릭스에서의 맨오른쪽 column을 곱하고 이를 더합니다.
이는 0 · 0 + 1 · 3 + 1 · 1 + 0 · 3 + 0 · 1 + 0 · 4 + 0 · 8 + 0 · 9 + 0 · 8 + 0 · 4 = 4 이 되고 우리는 4라는 값을 얻을 수 있습니다.


M1 ||||||| | | 
--- | --- | --- | --- | --- | --- | --- | --- | --- 
0|1|1|0|0|0|0|0|0|0
1|0|1|1|0|1|0|0|0|0
1|1|0|1|1|0|0|0|0|0
0|1|1|0|0|1|0|0|0|0
0|0|1|0|0|0|0|1|0|0
0|1|0|1|0|0|1|1|1|0
0|0|0|0|0|1|0|1|1|1
0|0|0|0|1|1|1|0|0|1
0|0|0|0|0|1|1|0|0|1
0|0|0|0|0|0|1|1|1|0

·

M2 ||||||| | | 
--- | --- | --- | --- | --- | --- | --- | --- | --- 
2|6|6|3|1|3|1|2|1|0
6|6|8|7|3|9|3|3|2|3
6|8|4|8|5|4|3|2|2|1
3|7|8|4|2|8|3|3|2|3
1|3|5|2|0|3|2|5|3|1
3|9|4|8|3|6|10|11|10|4
1|3|3|3|2|10|8|9|8|8
2|3|2|3|5|11|9|4|4|9
1|2|2|2|3|10|8|4|4|8
0|3|1|3|1|4|8|9|8|4


몇가지 매트릭스에 관한 노테이션을 소개합니다. m 행, n열을 가진 mxn 매트릭스가 있다고 했을때 이의 (i,j) entry를 ith row, jth column의 entry를 나타냅니다. 만약 매트릭스의 이름이 M이라면 M_ij 는 M 매트릭스의 (i,j) entyry를 나타냅니다. 
위에서 말한 것을 일반화 하기 위해서 nxn의 두개의 매트릭스 A,B가 있다고 했을때 얻어지는 nxn 매트릭스는 다음과 같습니다. 곱해진 결과의 (i,j) entry를 얻기 위해서는 A의 ith row, 와 B의 jth col을 곱해서 더합니다.

이제 우리는 길이3의 path의 갯수를 나타내는 매트릭스를 어떻게 얻을지를 고려합니다. 이는 우리가 4길이의 path 매트릭스를 얻는 것과 비슷합니다. 우리는 2길이의 path 매트릭스와 adjacent 매트릭스를 곱하면 됩니다. 그렇다면 2길이 path 매트릭스를 어떻게 얻나요? adjacent matrix를 제곱하면 됩니다.


### Chapter 2
### Implementing in CUDA

이제 우리는 매트릭스 곱을 cuda위에서 어떻게 구현하는지에 대해 다룹니다. 우리는 cuda 프로그래밍에 대한 설명으로 시작하여 매트릭스 곱을 구현합니다. 그리고 gpu에서의 shared-memory를 사용하여 속도의 향상을 얻는 방법을 다룰 것입니다.

### 2.1 the CUDA Programming Model

우리는 이 예제를 위해 cuda(NVIDA's compute unified device architecture computing model)를 사용할 것입니다. cuda 그리고 nvcc 컴파일러는 사용자로 부터 C 프로그램을 작성하도록합니다. 이는 리눅스, 윈도우 같은 임의의 host computer에서 실행되고 cuda-enable gpu 디바이스 위에서의 SIMT 코드를 실행할 수 있도록 합니다.
(cuda 에서는 host, device는 각각 임의의 computer와 그래픽 카드를 나타냅니다. 또한 SIMT의 의미는 single instruction, multiple thread 로써 수많은 thread가 병렬적으로 실행되고 한번의 instructuon(지시) 롤 실행된다. 하지만 개별 thread에 대해서 data specific 하다.)
여기서 우리는 필요한 cuda에 대한 내용만을 언급 할 것이다. 더 필요한 것이 있다면 NVIDIA CUDA C programming guide를 참조해라

SIMT (Single Instruction Multiple Thread)
http://haanjack.github.io/cuda/2016-03-27-cuda-prog-model/

SIMT라는 개념은 nVidia에서 자신의 CUDA 동작을 설명하기 위해서, CPU에서 사요하는 용어를 차용해서 만든 조어입니다. CPU에서는 주로 SIMD(Single Instruction Multiple Data)라는 용어를 사용하는데, CPU의 성능을 최대한 활용하기 위해서 하나의 명령어로 여러개의 데이터를 처리하도록 하는 동작을 의미합니다. 이와 비슷하게 CUDA는 하나의 명령어로 여러개의 쓰레드를 동작시킵니다. 그러면 각각의 CUDA 쓰레드는 데이터를 하나씩 처리하지만, CUDA에서 여러개의 쓰레드를 동작시키므로 동시에 여러개의 데이터를 처리할 수가 있습니다.

 ### 2.1.1 Threads, Blocks, Grids
 
 CUDA에서의 스레드는 각각 program counter, register를 가지고 있습니다. 모든 스레드는 "global memory" 이라고 불리는 공간을 통해 memory adress를 공유합니다. 같은 block에 속하는 스레드는 메모리의 제약이 더 있는 "shared memory"라는 매우 빠른 공간을 통해 access를 공유합니다.
 같은 lock의 스레드들은 instruction stream을 공유하고 병렬적으로 instruction을 실행합니다. 스레드의 실행이 갈라지면(if문 등으로) 다른 실행 브랜치는 분기 섹션이 완료될 떄 까지 순차적으로 실행됩니다. 이는 블록의 모든 스레드가 다음번 분기가 발생할 때 까지 다시 실행합니다.(if 문 등으로 실행 분기가 갈라지면.. if문이 완료될 때까지 한 블록 안의 모든 스레드가 기다리게 된다.)
 CUDA 장치는 많은 스레드를 동시에 실행합니다. 예를들어 NVIDIA Tesla C2075는 14개의 멀티프로세서를 가지며 각 프로세서는 32개의 코어를 가지고 있습니다. 그래서 448 개의 스레드를 동시에 실행 가능합니다. 반면 NVIDIA GT 330M 은 6개의 멀티프로세서를 가기조 각 멀티프로세서는 8개의 코어를 가져 48개의 스레드를 동시에 실행 가능합니다. 우리는 각각 스레드가 결과 매트릭스의 각 entry를 담당하도록 할 것입니다. 그래서 우리가 100x200 매트릭스와 200x500 매트릭스를 곱한다면 100x500=50000개의 스레드를 실행 할 것이고 
 최종적으로 곱해진 매트릭스는 100x500 일것입니다.
 
 cuda에 의해 실행되는 스레드는 block이라는 것으로 그룹화 되어져야 합니다. block은 512 or 1024개의 스레드를 가질 수 잇습니다. 우리는 이전에 봤던 100x500 매트릭스를 계산하기 위해서는 여러개의 block을 실행해야 만 합니다. NVIDA가 사용하는 *compute capability* 라는 말은 GPU에서의 일반적인 컴퓨팅 파워를 설명하는데 사용됩니다. 디바이스의 compute capability가 2.0 이하라면 하나의 block은 512 스레드 까지를 가질 수 있습니다. 2.0 이상이라면 1024 개의 스레드를 가질 수 있습니다. 같은 블록에 있는 스레드는 더 빠른 "shared memory" 라는 곳에 접근 할 수 있고 이는 일반적인 컴푸터에서 
 공유된 메모리 처럼 작동하며 다른 block의 스레드는 더 느린 "global memory' 라는 것으로 상호작용 할 수 있습니다.
 
 스레드의 blcok은 1,2,3 차원이 될 수 있고 이는 프로그래머의 선호에 따릅니다. 매트릭스는 2차원임으로 우리는 2차원 block을 사용할 것입니다. 우리의 코드가 모든 compute capability에서 돌아갈수 있게 하기 위해 block의 사이즈를 16x32 로 하여 총 512 개의 스레드를 포함합니다. 모든 block은 다른 block과 같은 차원을 가져야만 하며 그래서 우리는 결과 매트릭스의 모든 entry를 포함할 수 잇는 충분한 block을 실행해야 합니다. 그러므로 만약 매트릭스의 차원이 16의 배수가 아니라면 몇몇의 스레드는 계싼이 되어야 하지 않습니다. 피규어2.1은 70x200 매트릭스를 계산할때 사용되는 block을 보여줍니다.
 
 모든 blcok의 512개의 스레드가 실행되더라도 우리의 스레드가 맨처음 해야할 일은 매트릭스의 어느 부분과 관련되는 지를 계산하는 것입니다. 그리고 매트릭스 바깥에 놓은 매트릭스는 그 즉시 종료되어야 합니다.
 
 block은 그들 자체로 2차원의 grid로 구성됩니다. 피규어21.은 5x7 grid의 스레드 블록을 나타냅니다. 우리가 이를 GPU에서 실행할때 우리는 grid의 차원, block의 차원을 지정해야 합니다.
 
 ### 2.1.2 Thread Location : Dimension and index
 
 모든 실행되는 스레드는 grid, block 에서의 위치를 찾기 위한 구조체에 대한 접근을 할 수 있습니다. 
 
 
 - gridDim : grid에 대한 차원입니다. gridDim.x gridDim.y gridDim.z 를 통해 각 차원에 대해 접근 할 수 있습니다.
 - blockIdx : grid 안의 블록의 위치입니다. 
 - blockDim : block의 차원입니다.
 - threadIdx : 이 구조체는 속한 blcok에서의 스레드의 위치를 알려줍니다. 
 
 ### 2.1.3 Kernels and Mulitprcessor
 
커널이란 host 컴퓨터가 gpu에게 하도록 시킨 unit of work 입니다. cuda에서 커널을 실행하기 위해서는 3가지의 인자를 요구합니다.

- grid의 차원
- block의 차원
- device에서 실행될 커널 함수

커널 함수는 \_\_global\_\_ 과 같은 것으로 선언됩니다. 이는 GPU 에서 이 함수를 실행하라는 특별한 문법으로 block, grid 차원을 명시해야 합니다. 이 커널 함수는 GPU 연산을 위한 entry point를 제공하며 이는 마치 전통적인 c 프로그래밍에서의 main() 이 entry point를 제공하는 것과 같습니다.
밑의 예제는 우리가 MatMulKernel 이라는 global 함수를 실행을 하였고 이의 인자로 d_a d_b d_c 를 통과시켰습니다. dimGrid, dimBlock은 dim3 탕비이며 이는 unsigned interger 3개를 가집니다. 

```
MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
```
GPU에서 커널이 실행될때 grid 가 생성되고 block들이 gpu의 multiprocessor에 queued 됩니다. 피규어2.1의 예제에서는, 35개의 block이 생성됩니다. 
<b>이 블록은 멀티 프로세서가 사용 가능 해지면 multiprocessor에 제공되고 일단 시작되면 multiprocessor에서는 해당 블록의 스레드가 해당 multiprocessor에서 완료 될 때까지 실행됩니다.</b>
(스레드의 실행 완료는 한번 multiprocessor에 들어가면 보장된다!!? multiprocessor는 보통 SM이라고도 말한다.) CUDA 프로그래밍 모델은 이러한 block이 임의의 순서로 연산 될 것을 요구한다. 프로그래머는 이러한 block의 순서와 GPU 스케류이 어떤 순서로 실행 될 것인지를 알 수가 없다.
프로그램은 blcok 스케쥴이 어떤 순서인지에 대해 무관해야 하며 이러한 제약은 커널을 프로그래밍 하는 것을 좀더 어렵게 만들지만 더 많은 병렬성을 달성할 수 있게 한다. 또한 프로그램이 확장성을 가질 수 있게 한다.

나의 노트북은 6개의 multiprocessor를 가진다 만약 35개의 blcok이 내 노트북의 카드에서 스케쥴링 되어졌다면. 6개의 multiprocessor에 하나의 blcok이 차지할 것이다. 이들 스레드가 모두 완료되면 새로운 block이 연산을 위해 multiprocessor를 차지하기 된다. 이것이 35개의 block이 완료될 때 까지 이뤄진다. 
host 코드는 gpu 연산이 끝났다는 정보를 받고 host는 product의 결과를 읽을 수 있게 됩니다.

사실은 그림이 위에서 설명한 것보다는 좀 낫다?. 
<b>몇몇 blcok에서의 스레드가 device의 global memory에 데이터를 요청한다고 가정해보자. 이러한 요청이 완료되는 데에는 비교적 긴 시간이 걸린다. 요청이 처리되는 동안 multiprocessor는 유후 상태로 되게 됩니다. 이러한 idl time을 최소화 하기 위해서 cuda 스케쥴러는 multiprocessor에 두번째 block을 받고 두번째 block이 기다린 시간만큼 스레드를 실행합니다.
두개의 block이 하나의 multiprocessor에 대기하면 하나의 block이 대기하는 동안 다른 block으로 스위칭 할 수 있어 프로세서는 계속 연산을 수행할 수 있습니다.
</b>

cuda에서는 주어진 시간동안 8개의 block이 대기 할 수 있고 이는 multiprocessor의 레지스터와 각 블록의 shared memory에 따라 가능한 유효 블록들이 달라집니다. gpu multiprocessor는 매우 많은 레즈서터로 디자인되었기 때문에 컨텍스트 스위칭, block의 스위칭 하는 것이 매우 비용이 적습니다.


### 2.1.4 An implemenation of matrix multiplication

우리는 이제 매트릭스 곱을 gpu가 할 수 있도록 하는 host code를 작성 할 것이다. 커널 함수는 밑에 나와있다.

```
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row > A.height || col > B.width) return;
    
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    
    C.elements[row * C.width + col] = Cvalue;

}
```
첫번째 라인이 포함하고 있는 \_\_global\_\_ 키워드는 디바이스에서 실행될 함수의 entry-point입니다. float Cvalue = 0 이라는 선언은 row, column entry의 곱을 축적할 레지스터에 대한 선언입니다.

그 다음의 2개의 라인은 스레드가 매트릭스 에서의 row, column을 찾는데 도움을 줍니다. 다음으로 넘어가기 전에 이 라인을 잘 이해하는 것이 중요합니다. 다음 if statement는 product 매트릭스으 경계를 넘어가는 스레드들을 처리합니다.
이는 매트릭스의 맨 오른쪽 혹은 맨 밑의 blcok일 때 if 문이 수행 될 것입니다.

다음의 3개의 loop는 A의 row, B의 column만큼 수행됩니다. A,B의 row, column의 곱을 계산하고 Cvalue에 축적합니다. A,B 는 device의 global-memory에 row-major order로 저장되어져 있습니다. 이의 의미는 매트릭스가 1차원 배열의 형태로 저장되어져 있고 첫번째 row가 나오고 그다음 second row, 그리고 3번째 이런 순서로 저장되어져 있다는 뜻입니다.
그래서 매트릭스의 (i,j) entry를 찾기 위해서 우리는 ith row의 시작 인덱스를 찾아야 합니다. 그리고 난 후에 그 행에 해당하는 jth entry를 더합니다. 마지막으로 커널의 마지막 라인은 곱의 결과를 적절한 저장할 C라는 매트릭스의 적절한 인덱스에 저장합니다. 

### 2.1.5 Quick outline of the cod in chapter 3

multNoShare.c 에는 3가지 함수가 존재합니다.

- main() 이 함수는 commandline으로 받은 차원에 해당하는 두개의 랜덤 매트릭스를 생성합니다. 매트릭스 A는 0,1,2 의 랜덤한 넘버로 채워지고 매트릭스 B는 0,1 로 채워집니다. 그 후 main() 은 MatMul() 을 호출하고 이는 이 둘을 곱하고 매트릭스 C에 저장합니다.
포함된 Makefile은 실행 가능한 multNoShare 를 생성하고 이는 다음과 같이 사용될 수 있습니다. multNoShare 10 15 8 이는 10x15 의 매트릭스 A와 15x8의 매트릭스 B를 생성할 것이고 10x8의 매트릭스과 계산되어 첫번째 10 row, col을 출력할 것입니다.

- MatMul(Matrix A, Matrix B, Matrix C) 이 함수는 매트릭스 A, B를 인풋으로 받습니다. 그리고 매트릭스 C을 채웁니다. 이는 먼저 매트릭스 A를 할당하고 데이터를 디바이스로 카피합니다. 그 후 매트릭스 B에도 똑같이 수행합니다. 매트릭스 C에 해당하는 공간도 디바이스에 할당합니다. product matrix의 전체를 커버할 수 있는 blcok의 갯수를 계산하고 그만큼의 블록으로 커널을 실행합니다. 커널이 끝나면 이는 디바이스에서 매트릭스 C를 읽고 메모리를 해제합니다.

- MatMulKernel(Matrix A, Matrix B, Matrix C) 위에서 설명한 듯이 이함수는 다이바스에서 실행되고 매트릭스 곱을 계산합니다. 이 함수는 A,B가 이미 디바이스의 global-memory에 할당되있다고 가정하며 product matrix 또한 global-memory에 있습니다. 그래서 host는 global-memory에서 읽어올 수있습니다.

### 2.2 The memory hierarchy
위에서 설명했듯이, mutiprocessor은 매우 많은 수의 32-bit 레지스터를 가지고 있습니다. 1.1, 1.1 compute capabilities의 디바이스는 8k개의 레지스터를 가지며, 1.2, 1.3 은 16k 의 레지스터, 2.0 이상의 것들은 32k 의 레지스터를 갖습니다. gpu에서의 메모리를 설명합니다.

- register : 레지스터는 가장 빠른 메모리 입니다. 각 clock cycle 에서 어떠한 지연시간도 가지지 않습니다. 일반적인 cpu 처럼 스레드의 레지스터는 다른 스레드와 공유되지 않습니다.
- shared-memory : shared-memory는 일반적인 cpu에서의 l1-chche memory입니다. 이는 multiprocessor와 매우 가까운곳에 위치하며 매우 짧은 access tiem을 갖습니다. 이는 하나의 블록에서의 모든 스레드가 공유합니다. 
- global-memory : 이는 device 안에 위치하지만. multiprocessor와는 조금 떨어져 있습니다. 그래서 shared-memory 보다 100 배정도의 access tiem을 갖습니다. 커널의 모든 스레드는 global memory에서 데이터를 공유합니다.
- local-memory : global-memory가 저장된 곳의 스레드별의 memory 입니다. 컴파일러가 스레드의 데이터를 저장할 충분한 레지스터가 존재 하지 않다고 생각하면 Variable을 스레드의 local-memory에 저장합니다. 이름이 "local" 이지만 이 메모리는 매우 느립니다.
- Constant-memory : 64k의 constant memory는 multiprocessor의 off-chip에 위치하며 read-only 입니다. host code는 커널을 실행하기 전에 디바이스의 constant-memory를 작성하면 커널은 이를 constant memory에서 읽을 것입니다. costant-memory access는 cached 이며 각 multiprocessor는 8k의 constant memroy를 cache 할 수 있습니다. 그래서 constant memory에서의 read는 매우 빠릅니다. 모든 스레드는 constant memory에 접근 가능합니다.
- texture-memory : texture-mapping에 사용되는 메모리입니다.

### 2.2.1 Matrix Multiplication with shared memory


위에 설명 된 메모리 계층에 비추어 우리가 고려할 수있는 최적화가 무엇인지 살펴 봅시다. 커널 code에서의 loop를 살펴보면 우리는 각 스레드가 (2xA width) 만큼의 element를 global memory에서 load 하는것을 알 수 있습니다.
각 반복마다 두개의 메모리를 불러오고 하나는 매트릭스 A, 하나는 매트릭스 B에서 불러옵니다. global memory에 접근하는 것은 상대적으로 매우 느리므로 커널을 망칠 수 있습니다. 이는 각 액세스에 대해 스레드를 수백 클록 사이클 동안 유휴 상태로 둡니다.

global-memory에 대한 접근의 수를 줄이는 하나의 방법은 매트릭스 A, B를 부분적으로 shared memory에 load 하는 것입니다. 이는 더 빠른 access를 가능하게 합니다. 이상적으로는 우리는 이 두개의 매트릭스 전체를 shared memory에 load 하는것 입니다. 하지만 shared-memory는 매우 부족한 자원임으로 큰 매트릭스를 load 할 수 없습니다. compute-capability가 1.x인 디바이스는 16kb , 2.x 인 것들은 48kb 를 각 mutiprocessor당 가지고 있습니다. 그래서 우리는 매트릭스 A,B 의 부분을 load 하는 것이 필요합니다.

이를 하기 위한 한가지 방법은 피규어2.2 에 나타나있습니다. 매트릭스 A는 왼쪽, 매트릭스 B는 위쪽, 매트릭스 C는 아래쪽에 나타나 있습니다. 이는 매트릭스를 나타내기 위한 좋은 방법입니다. 매트릭스 C의 각 엘리먼트는 왼쪽 A매트릭스의 row와 위쪽 매트릭스 B의 column을 통해 계산됩니다. row, column은 C의 곱셉을 위해 정렬되어져서 나타나 시각화에 좋습니다. 이 그림과 우리의 코드에서 우리는 BLCOK_SIZE X BLOCK_SIZE 만큼의 thread block을 사용하며 매트릭스 A,B 의 차원이 BLCOK_SIZE X 2 라고 가정합니다.

다시한번 말하면 product matrix C의 한 엘리먼트에 대한 계산은 한 스레드가 책임지고 있습니다. 피규어 2.2 에서 빨갛게 칠한 부분입니다. (매트릭스 C의 노란색 사각형은 하나의 block을 나타내며 노란색 사각형 안의 빨간색 네모는 하나의 스레드를 나타냅니다.) 우리의 스레드는 C의 엘리먼트를 A의 빨간색 row, B의 빨간색 column을 통해 계산합니다. 이것은 쪼개져야 하는데 그것을 이제 설명합니다.

우리는 매트릭스 A, B를 겹쳐지지 않는 BLOCK_SIZE X BLOCK_SIZE 의 submatrice 로 나눌 것입니다. 우리가 그림의 red row, colmun을 보면 알 수 있듯이 빨간색 row, column은 같은 크기를 가지므로 이들 submatrice에서의 같은 갯수의 entry를 통과 시킬 것입니다. 만약 우리가 매트릭스 A의 submatrix에서 left-most 로 shared-memory에 저장을 했다면 매트릭스 B에서는 top-most 로 저장을 해야 합니다. 그래야 shared-memory에서 읽기만을 수행하고 서로 곱, 합 연산을 할 수 있습니다. 
이렇게 함으로써 몇가지 장점이 있는데 shared-memory에서의 submatrice를 가지고 있는한 block 안의 모든 스레드가 그들의 일부분의 sum을 shared-memory의 동일한 데이터를 가지고 수행가능합니다.

각 스레드가 연산이 끝나면 우리는 다음 BLCOK_SIZE X BLCOK_SIZE의 submatrix를 load합니다. 그리고 C에서의 term-by-term product를 합니다. 모든 sub-matrice에 대해 진행되면 우리의 C에서의 entry를 계산합니다.  

```
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y, blockCol = blockIdx.x;
    
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    
    // Each thread computes 1 element of Csub accumulating results into Cvalue
    float Cvalue = 0.0;
    
    // Thread row and column within Csub
    int row = threadIdx.y, col = threadIdx.x;
    
    // Loop over all the sub-matrices of A and B required to compute Csub
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        
        // Get sub-matrices Asub of A and Bsub of B
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();
       
       
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
        Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();
    }
    
    
    // Each thread writes one element of Csub to memory
    SetElement(Csub, row, col, Cvalue)
}
```

### 2.2.2 Worth the Trouble?

이런 방식으로 공유 메모리를 이용하는 코드는 좀 더 복잡하며, 그 이익이 비용보다 중요한지 물어 보는 것이 합리적입니다. 이를 분석하는 좋은 방법은 single elemnet 매트릭스 A를 고려해 보는 것입니다. 그리고 그것이 global-memory로부터 몇번 load 되는지를 살펴 보면 좋습니다. 
우리의 shared-memory를 이용하지 않는 첫번재 프로그램에서 각 스레드마다 하나의 entry가 load됩니다. 피규어 2.2 을 살펴보면 each column of B 마다 하나의 A의 entry를 load하게 됩니다. 
<b>그러나 우리의 shared-memory 코드에서는 A의 blcok이 그 entry를 포함하는 경우에만 entry를 load합니다. 그리고 product matrix C에서도 똑같이 일어납니다. 이는 (width of B)/(BLOCK_SIZE) 번을 load합니다. 만약 BLOCK_SIZE 가 16으로 같다면. shared-memory의 구현이 1/16 만큼 적게 load를 하게 됩니다. </b>

1x1 1xn 매트릭스 곱을 생각해보자. 이는 1xn 매트릭스의 출력을 낼 것이다. 1xn 스레드를 계산하기 위해서 global-memory 구현은 총 n번의 global-memory load를 해야한다(A 매트릭스에만). 반면 shard-memory의 구현은 
(width of B )/(BLOCK_SIZE) 만큼.. 만 A를 load 하면 된다. 이번에는 B가 one element 매트릭스라고 생각해보자.. 이떄도 똑같다!!


### 2.2.3 Shared Memory, Device Functions and the syncthreads() Command

GetSubMatrix(), GetElement() and SetElement 함수를 호출하는 것에 주목하자. 이들 또한 커널 함수로써 디바이스에서 돌아가는 코드이다. device 함수는 \_\_device\_\_ 키워드에 의해 선언된 함수로써 chapter 4에서 볼 수 있다. 이들은 특별한 것이 아닌 단지 device를 위한 함수일 뿐이다. 이 함수가 thread 에 의해 실행될 함수라는 것을 명심하면 threadIdx 에 접근 가능하다는 것을 알 수 있다.

\_\_shared\_\_ 키워드는 우리의 커널에서 As, Bs 를 선언하는데 사용되었다. 이는 컴파일러가 이들의 multiprocessor's shared_memory에 저장하도록 요구한다. 적어도 하나의 block을 수용가능한 충분한 shared-memory가 잇는 것이 중요하다. 어떻게 알 수 있는지 살펴보자. 
우리는 BLOCK_SIZE를 16으로 지정했다. 그래서 As, Bs는 16x16 = 256 flaot을 가지게 될 것이다. float은 4 byte를 사용하므로 우리는 총 2x256x4=2k 의 shared-memory가 필요하다. 그래서 16k의 shared-memory에 수용가능하다. 우리의 multiprocessor는 최대 8개의 block을 수용가능하므로 매우 좋다. 더 많은 유효 block을 하나의 multiprocessor에서 사용가능하다. (multiprocessor가 최대 8개의 blcok을 가질 수 있는데 shared-memory도 딱 2k라서 8개 유효블럭이 가능함.) 이는 스케쥴러가 스레드가 메모리 요청을 기다릴 동안 다른 작업을 할 가능성을 높여준다.

(sm은 블록들을 스위칭 하면서 multiprocessor의 역할을 수행한다. 이때 메모리 로드, 저장 시간동안 남는 걸... 다른 블록으로 스위칭 해서 활용한다. 하지만 블록 안의 스레드들은 모두 함께 실행되고,, 함께 끝나는 것이 보장댐!)

### 2.2.4 Calling the kernel


위에 주어진 MatMulKernel () 커널의 커널은 __syncthreads() 호출을 사용합니다. 하나의 어떤 스레드가 __syncthreads() 가 호출할때. 그 thread의 block안의 모든 스레드는 이 포인트 전까지 계산이 끝나야 하고 더이상의 진행이 허락되지 않는다. (마치 군대의 행진에서.. sub군대? 각 부대가 다함께 움직이는 것을 상상해보자.) __syncthreads()를 처음 호출함과 함께 우리는 submatices A,B의 모든 entry들이 shared-memory에 load되었다는 것을 알 수 있다. (이 sync는 block 단위로 진행되는 것..?인 것 같다.) 두번째 __syncthreads() 호출은 C의 submatrix이 처음의 submarix A,B를 가지고 잘 계산이 되었다는 것을 보장한다. 참고로 __syncthreads()는 이러한 스레드간의 동기화를 가능하게 하지만 사용시 병렬화를 최소화 함으로 성능저하가 일어날 수 있습니다.  

```
void MatMul(const Matrix A, const Matrix B, Matrix C) \{
    ...
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaThreadSynchronize();
    printf("Run kernel: %s\n", cudaGetErrorString(err));
    ...
\}

```

이 dim3 구조체는 cuda 라이브러리에 정의되어져 있고 매우 유용합니다. 이는3개의 unsigned int로 정의되어져 있고 gird or block의 차원을 명시하는데 사용됩니다. 참고로 위의 예제에서 이 구조체의 생성자가 오직 2개의 인자만을 가집니다. 3개의 인자보다 적은 인자가 주어지면 없는 인자는 0으로 초기화 됩니다. 저 두 라인은 dimBlock을 BLOCK_SIZE X BLCOK_SIZE 의 정사각형으로 정의하고 dimGrid를 전체 매트릭스를 충분히 덮을 수 있도록 정의합니다. (여기서 우리는 매트릭스의 height, width가 BLOCK_SIZE의 배수라고 가정했습니다.)

그후 커널의 호출이 옵니다. MatMulKernel은 함수의 이름입니다. 이는 \_\_global\_\_ 로 정의되어져 있어야 합니다. <<<와 >>> 사이에는 dim3 구조체가 와야 합니다. 커널을 호출을 마루리 하는 것은 파라미터로써 보통의 c와 비슷합니다.

### 2.2.5 An aside on stride

이 코드에서 사용되는 스트라이드 변수에 관해 말할 가치가 있습니다. 위에서 언급했듯이 매트릭스는 row-major-order로 저장됩니다. 이 의미는 global-memory에서의 매트릭스를 바라볼대 우리는 first row, seconde row, third row,,, 의 식으로 매트릭스가 저장됩니다. stride 라는 것은 매트릭스의 width를 참조합니다. 이는 왜냐하면 우리가 매트릭스의 한 열을 이동하기 위해서 얼마만큼의 step을 이동해야 하느지를 알아야 합니다. 예를들어 10x4 매트릭스가 메모리에 있다고 했을때 매트릭스의 첫번째 열은 4칸씩 떨어져 위치합니다. 밑의 그림이 나타내는 것처럼

우리는 이 코드에서 매트릭스에게 stride 라는 것을  것을 주는데 이는 submatrix 때문입니다. 다음 열을 가기 위한 step을 알기 위해서는 submatrix의 width가 아닌 그 부모 matrix의 width를 사용해야 합니다.

### 2.3 Compiling and Running the Included Code

Makefile은 두개의 실행가능한 파일의 컴파일을 포함합니다. 소스코드를 통한 실험에서 이러한 Makefile이 작업을 빠르게 해줄 수 있습니다.

- make multNoShare 라고 불리는 실행가능한 파일을 컴파일 합니다. 이는 shared-memory가 없이 매트릭스 곱을 실행합니다. 이는 2개의 랜덤 매트릭스를 생성하고 그들의 곱을 계산하며 10번째 row, column 가지 출력합니다. 
- make multShare shared-memory를 사용한 multShare 실행파일을 만듭니다.
- make all 두가지 모두를 만듭니다.

IMPORTANT NOTE : 많은 에디터들이 C코드는 자동적으로 하이라이팅 하지만 .cu 는 하지 못합니다. 그래서 나는 코드를 .c 파일로 작성하고 Makefile에서 컴파일 전에 이들을 .cu 파일로 복사합니다. 이는 nvcc 컴파일러가 .cu 확장자를 요구하기 때문입니다. 이는 다른 방법으로 다뤄질 수 있수 있고 선호에 따릅니다. 하지만 여기서는 Makefile에 이러한 작업이 포함되있는것을 알고 있어야 합니다.

아래에서 다양한 사이즈의 매트릭스에 대한 실험을 준비했습니다. 만약 그래픽 카드 메모리의 양을 넘어간 매트릯르를 실행하기를 원한다고 해봅시다. 프로그램은 컴파일되지만 실행시간에는 프로그램이 매트릭스 요소를 보유하기 위해 그래픽 카드에 메모리를 할당하려고 시도하면
메모리 할당 오류가 발생합니다. 프로그램이 중단되지는 않지만 결과는
유효하지 않습니다. 따라서 메모리 할당 호출의 반환 값을 확인하는 것이 중요합니다.

```
cudaError_t err = cudaMalloc(&d_A.elements, size);
printf("CUDA malloc A: %s",cudaGetErrorString(err));
```
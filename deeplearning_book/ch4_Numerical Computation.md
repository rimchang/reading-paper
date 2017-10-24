<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# Numerical Computation

### 4.1 Overflow and Underflow

softmax의 성질을 이용해서 softmax 함수의 수치에러를 줄일 수 있음 그러나.. Crossentropy를 위해 log를 쒸우는 경우가 많음.. 이때 log를 쒸울때도 수치해석적 에러가 생길 수 있음.. 이 때문에!!
많은 프레임워크에서 log_softmax함수가 있다. 그러니깐 log_softmax를 쓰자!!

### 4.2 Poor Conditioning

condition number : 가장 큰 eigenvalue / 가장 작은 eigenvalue  

이 값이 크면. 역행렬은 입력값의 error(noise) 에 매우 민감하게 된다. 이 민감도는 rounding error 때문이 아닌 매트릭스 자체의 특성이다. 

pooly conditioned matrices는 역행렬을 구하는 과정에서 수치적 에러가 계속 가중되어 영향을 끼칠 것이다.

### 4.3 Gradient-based optimization

line search :  몇몇개의 e에 대해 f(x − e∇xf(x)) 를 계산해보고 smallest objective function value를 가지는 e를 선택. (learning-rate decay 대신에 쓸 수 있는 방법)

hill climbing : gradient descent는 continuous space에서 동작하도록 제한되어 있는데.. 이를 discreate space로 일반화 한것.

### 4.3.1 Beyond the Gradient: Jacobian and Hessian Matrices

2차 미분은 input의 변화에 따라 1차 미분이 얼마나 변하는 지를 알려준다. 이는 gradient step이 우리가 기대한 만큼의 향상을 가져다 주는지를 알려준다. 2차 미분을 곡률의 measure로 생각할 수 있다.

우리가 딥러닝에서 다루는 대부분의 함수들은 symmetric Hessian almost every where 이다. 이 때문에 eigen value가 2차 미분의 크기를 결정한다. 

헤시안을 가지고 특정 방향의 2차 미분을 나타내기 위해선 <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{W}(A,f)&space;=&space;(T,\bar{f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{W}(A,f)&space;=&space;(T,\bar{f})" title="\mathcal{W}(A,f) = (T,\bar{f})" /></a>
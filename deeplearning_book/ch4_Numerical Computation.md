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

헤시안을 가지고 특정 방향의 2차 미분을 나타내기 위해선 <img src="http://www.sciweavers.org/upload/Tex2Img_1508822888/render.png" /> 을 통해 나타낼 수 있다. 여기서 d가 eigen vector 라면 그 방향으로의 2차미분은 egienvalue와 같게 된다. 고유벡터가 아닌 일반적인 방향의 d라고 하면 그 방향의 2차 미분은 eigenvalue의 weighted average가 된다. 

이 때문에 maximum eigenvalue는 maximum 이차미분을 결정하고 minimum도 같다.

second derivative test 

critical point : f\`(x) = 0  (local minimum or saddle point)  
saddle point or flat region : f\`\`(x) = 0  
local minimum : f\`\`(x) > 0 , f\`(x) = 0 
 
 위의 수식의 의미는 2차미분이 양수이므로 x의 변화가 항상 생긴다는 뜻이다. 즉 
f\`(x − e) < 0 and f\`(x + e) > 0 가 된다는 뜻이다. (x를 약간 감소시키면 1차미분은 감소하고 x가 증가하면 1차미분은 증가해야한다.) 다른 말로 하면 x를 증가시키면 기울기가 그 포인트부터 right-uphill이 되고 x를 감소시키면 기울기가 그 점에서부터 left-uphill 이 된다는 의미이다.

다변량에서의 second derivative test

critical point : ∇xf(x) = 0
local minimum : ∇xf(x) = 0, H is positive definite(all eigenvlaue are positive)  
saddle point : 적어도 하나의 eigenvalue는 양수이고 적어도 하나의 eigenvalue가 음수


Hessian의 condition number(maximum |eigenvalue| / minimum  |eigenvalue|) 는 이차 미분이 서로 얼마나 다른지를 말해준다. (condition number가 5라면.. 어느 방향과 다른 방향과의 곡률의 차이가 5배가 난다는 뜻)Hessian이 poor condition number 일때는 gradient descent가 멍청하게 동작한다 이는 어떤 방향에서는 미분이 빠르게 증가하고 다른 방향에서는 천천히 증가하기 때문이다. 

gradient descent는 이떤 방향에서는 빠르고, 느린지에 대한 정보를 알지 못하기 때문에 나쁘게 동작한다. 이는 좋은 learning rate를 정하기도 어렵게 한다. 이는 Hessian 매트릭스의 정보를 활용하여 해결 할 수 있다.

saddle point를 어느 축에서는 local-maximum 다른 축에서는 local-minimum인 점으로 볼 수 있다.

뉴턴메소드는 근처의 critical point가 minimum 일때 적절하고 gradient descent는 saddle point에 잘 빠지지 않는다. (뉴턴 메소드는 saddle point에 빠지는 경우가 많다. 근데 딥러닝은 매우 많은 saddle point를 가지고 있다고 알려져 있음)

gradient-method, newton's-method 는 다양한 함수에 적용 할 수 있지만 수렴한다는 보장이 있지는 않다. 딥러닝이 아닌 다른 분야에서는 알고리즘을 제한하여 수렴성을 보장 하도록 한다.

딥러닝에서는 Lipschitz continuous 하다면 수렴성이 보장된다.

∀x,∀y,|f(x) − f(y)| ≤ L||x − y||

이는 input의 작은 변화가 있을때 output도 작은 변화만 있다는 것을 보장해 준다.


### 4.4 Constrained Optimization

1, gradient-descent 안에 제약조건을 넣는방법 

gradient descent step을 하고 제약 조건 안으로 다시 매핑하는 방법

2, constrained optimization problem 

Lagrangian을 이용하는 방법

### 세미나 동영상
https://www.youtube.com/watch?v=YHB2reIlNsI&index=5&list=PLsXu9MHQGs8cshZb3YUdtBhcu3LQp0Ax9

Hessian은 symmetric하다. 근데 real, symmetric 이므로 정규직교고유벡터로 분해가 가능하다. 그래서 without loss of generality Hessian은 diagonal 하다고 할 수 있음

뉴턴 메소드는

x_i+1 = x_i -ef\`(x)H(X)

H(x) 의 역할 rotating and rescaling it and rotating back

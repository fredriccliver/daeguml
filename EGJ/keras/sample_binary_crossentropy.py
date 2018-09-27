# Keras (Deep Learning, Neural Networks)
# 크다 vs 작다 (이진문제)
# 1 ~ 10 사이의 숫자를 입력해서 0 (작다) 또는 1 (크다) 의 결과를 추출

##
# 패키지 호출
# tensorflow 1.2 부터 keras 를 포함하기 시작함
from tensorflow.keras.models import Sequential  # Keras 모델
from tensorflow.keras.layers import Dense       # Keras 레이어
import numpy as np                              # 수학적 계산을 위함

##
# 모델 생성
model = Sequential()

##
# 레이어 생성 / 네트워크 추가 (Dense 레이어 5개 생성 및 모델의 네트워크에 추가)
model.add(Dense(64, input_dim=1, activation='relu'))    # 레이어: 입력 = 1, 출력 = 64
model.add(Dense(64, activation='relu'))                 # 레이어: 입력 = 64, 출력 = 64 (입력은 암묵적으로 이전 레이어의 값을 따른다.)
model.add(Dense(64, activation='relu'))                 # 레이어: 입력 = 64, 출력 = 64 (입력은 암묵적으로 이전 레이어의 값을 따른다.)
model.add(Dense(64, activation='relu'))                 # 레이어: 입력 = 64, 출력 = 64 (입력은 암묵적으로 이전 레이어의 값을 따른다.)
model.add(Dense(1, activation='sigmoid'))               # 레이어: 입력 = 64, 출력 = 1 (입력은 암묵적으로 이전 레이어의 값을 따른다.)

##
# Dense 간략 설명
# 첫번째 인자: 출력 뉴런의 수
# input_dim: 입력 뉴런의 수
# activation: 출력을 어떻게 변화할 것인가를 정의

##
# Activation Function 설명
#
# [sigmoid]
# 1. 연속이기에 미분가능
# 2. 0.0~1.0 사이 값을 가짐 (변화의 기울기가 가파름)
# 3. 이진문제 => 분류에 적합
#
# [relu]
# 1. [sigmoid] 에서 문제가 된, Gradient Vanishing 를 해결
# 2. 이하 상세

##
# ReLU (Rectified Linear Unit)
#
# 1. 개요: [Sigmoid] 함수는 0에서 1사이의 값을 가지며, 경사하강법(gradient descent)을 사용해 역전파(Backpropagation)를 수행합니다.
# 역전파(Backpropagation) 수행시, 레이어를 지나면서 경사값(미분값, gradient)을 계속 곱하므로 결국 결과는 0으로 수렴합니다.
# 따라서 레이어가 많아질 수 록 잘 작동하지 않기에, 이러한 문제를 해결하기 위해 [ReLU] 함수를 활성함수로 사용하게 됩니다.
# [ReLU] 는 입력값이 0보다 작으면 0이고 크면 입력값 그대로를 내보내는 연산을 수행합니다.
# f = { (x < 0) f(x) = 0 & (x >= 0) f(x) = x }
#
# 참고. 캐나다 고등 연구소(CIFAR, Canadian Institute For Advanced Research)에 지원 했던, Hinton 교수가 2006년에 발전시킨 함수.
#
# 2. 이점
# 가. Sparse Activation: 0 이하 입력에 대해 0을 출력함으로 부분적으로 활성화가 가능
# 나. Efficient Gradient Propagation: gradient 값의 손실이 없음
# 다. Efficient Computation: 선형함수이기에 미분 계산이 간단함
# 라. Scale-Invariant: max(0, ax) = a max(0, x)
#
# [참고] https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

##
# 입력값 생성 및 모델 출력값 확인
X_sample = np.array([[3]])
Y_sample = model.predict(X_sample)

print("\r")
print("---- 학습 전 ----")
print("샘플 입력값([[3]])에 대한 샘플 출력값")
print(Y_sample)                         # 네트워크 정의만 했을 뿐, 학습을 하지 않은 상태

##
# 네트워크 학습: 목표함수 & 최적화기가 필요
#
# 이진분류이므로 목표함수는 binary_crossentropy 선택
# 최적화기는 일반적으로 사용하는 adam 선택
model.compile(loss='binary_crossentropy', optimizer='adam')

# 예상 연산 식: Y' = w * X + b
# Y': 네트워크에 의해 계산된 값
# X : 네트워크의 입력값
# w, b: 네트워크의 가중치

# 문제와 정답 준비
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]])

# Fitting
model.fit(X, Y, epochs=100, batch_size=10)

##
# fit 인자 간략 설명
# epochs: 문제와 정답을 몇 번 반복해서 학습하느냐를 나타냄
# batch_size: 몇 문항을 풀고나서 정답과 비교하느냐를 나타냄

X_value = np.array([[3]])
Y_value = model.predict(X_value)

print("\r")
print("---- 학습 후 ----")
print("입력값([[3]])에 대한 출력값")
print(Y_value)                          # 학습을 한 이후의 상태

print("\r")
print("---- 학습 전/후 비교 ----")
print(Y_sample, " vs ", Y_value)

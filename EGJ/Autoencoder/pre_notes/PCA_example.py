# PCA; 주성분 분석
#
# MLP 와 동일한 코드로 작성
# 출력의 갯수 = 입력의 갯수
# 활성함수 사용하지 않음; 모든 뉴런이 선형
# 비용함수는 MSE

# 파이썬 2와 파이썬 3 지원
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 모듈 경로때문에 필요; PyChar(intelliJ) 기준
import os
import sys
DIR_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(DIR_PATH)
sys.path.append(os.path.abspath(DIR_PATH + '../../'))

# Custom modules
import EGJ.utils as utils

# 맷플롯립 설정
# %matplotlib inline
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
# plt.rcParams['font.family'] = 'NanumBarunGothic'
# plt.rcParams['axes.unicode_minus'] = False

# 3D 데이터 셋 생성
rnd.seed(4)
m = 200
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:100])
X_test = scaler.transform(data[100:])

# 출력 그래프를 일정하게 하기 위함
utils.reset_graph()

# 오토인코더 생성
n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

# MSE; 평균 제곱 오차 == 비용함수
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        # 레이블 없음; 비지도 학습
        training_op.run(feed_dict={X: X_train})
    codings_val = codings.eval(feed_dict={X: X_test})

fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
utils.save_fig(file_name="linear_autoencoder_pca_plot", root_path=ROOT_PATH)
plt.show()

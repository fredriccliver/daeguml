# Stacked Auto-Encoder Example (= Deep Auto-Encoder)
#
# Deep MLP 과 비슷하게 구현 가능
# example code = MNIST; Modified National Institute of Standards and Technology database

# 파이썬 2와 파이썬 3 지원
from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial

# 모듈 경로때문에 필요; PyChar(intelliJ) 기준
import os
import sys
DIR_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.abspath(DIR_PATH)
sys.path.append(os.path.abspath(DIR_PATH + '../../'))

# Custom modules
import EGJ.utils as utils

# 데이터 생성
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# 유사난수 초기화
utils.reset_graph()

# 오토인코더 생성
n_inputs = 28 * 28
n_hiddens = [300, 150, 300]
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

# What is this? Not found contrib.layers in tensorflow
he_init = tf.variance_scaling_initializer()
# == lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))

l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hiddens_1 = my_dense_layer(X, n_hiddens[0])                         # 은닉층1
hiddens_2 = my_dense_layer(hiddens_1, n_hiddens[1])                 # 은닉층2, 중간-은닉층: 코딩층
hiddens_3 = my_dense_layer(hiddens_2, n_hiddens[2])                 # 은닉층3
outputs = my_dense_layer(hiddens_3, n_outputs, activation=None)

# MSE; 평균 제곱 오차 == 비용함수
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# 28*28 흑백 이미지를 그리기 위한 유틸성 함수
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


# 모델을 로드하고 테스트 세트에서 이를 평가
def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            utils.load_tf(sess=sess, file_name=model_path, root_path=ROOT_PATH)

        outputs_val = outputs.eval(feed_dict={X: X_test[:n_test_digits]})

    # fig
    plt.figure(figsize=(8, 3 * n_test_digits))

    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])


show_reconstructed_digits(X, outputs, "stacked_example_model")
utils.save_fig("reconstruction_plt")

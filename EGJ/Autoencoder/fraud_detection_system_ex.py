# Credit Card Fraud Detection

"""
# 오토 인코더 활용; 금융 거래 분석 모델 (ex)

# 구조: 딥 네트워크 기반의 비지도 학습 모델로, 뉴럴 네트워크 두개를 뒤집어서 붙여놓은 형태 ( 은닉층 구조; 인코더 <-> 코딩층 <-> 디코더 )

# 원리
1. 인코더를 통해 입력 데이터에 대한 특징 추출
2. 추출된 결과를 가지고 뉴럴 네트워크를 역으로 붙여서 원본 데이터 생성
3. 위 과정에서 입력과 출력값이 최대한 같아지도록 튜닝함으로써, Feature 를 잘 추출할 수 있게 하는

# 비정상거래 검출 시스템 적용에 대한 추가설명
학습 되지 않은 데이터의 경우, 디코더에 의해 복원이 제대로 되지 않고 원본 데이터와 비교했을 때 차이값이 크다.
따라서 정상 거래로 학습된 모델은 비정상 거래가 들어왔을 때 결과값이 입력값과 많이 다를 것이라고 가정한다.

그렇다면 입력값 대비 출력값이 얼마나 다르면 비정상 거래로 판단할 것인가에 대한 임계치 설정이 필요하다.
이는 실제 데이터를 비교 분석하거나 통계 데이터에 의존할 수 밖에 없다.
예를 들어 전체 거래의 0.1%가 비정상 거래라 가정한다면, 입력값과 출력값의 차이가 큰 것들 중 순서대로 상위 0.1%만을 비정상 거래로 판단한다.

하지만, 오토인코더는 비지도 학습이다.
결과값으로 정상/비정상을 판단하기 보다는 비정상 거래일 가능성을 염두해두고, 거래를 비정상 거래일 것이라고 예측한다.
예측한 비정상 거래 후보에 대해서 실제 확인이나 다른 지표에 대한 심층 분석을 통해 비정상 거래임을 확인 판단하는 것이 더 정확하다.

이러한 과정을 거쳐서 비정상 거래가 판별이 되면, 비정상 거래에 대한 데이터를 라벨링한다.
이를 통해 다음 모델 학습시 임계치 값을 설정하거나 다른 지도 학습 알고리즘으로 변경하는 방법등으로 변경하는 과정이 이루어 진다.
"""

# 파이썬 2와 파이썬 3 지원
from __future__ import division, print_function, unicode_literals

# import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# 모듈 경로때문에 필요; PyCharm(intelliJ) 기준
import os
import sys
DIR_PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(DIR_PATH + '../../'))

# Custom Modules
import EGJ.Autoencoder.utils as utils

# 공통 변수
ROOT_PATH = os.path.abspath(DIR_PATH)
FILE_NAME = utils.path_base_name(__file__)

# TensorFlow logging
tf.logging.set_verbosity(tf.logging.INFO)

# import data
#
# df = pd.read_csv("./data/fraud_detection_system.ignore.csv")
# df.describe()


# TensorFlow 를 이용하여, 데이터 로드 및 변환
def read_and_decode(filename_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0.0]] * 31
    columns = tf.decode_csv(value, record_defaults=record_defaults)

    # first column is time field from 1 to 28 column is feature, 2
    value = tf.convert_to_tensor(columns[1: 29], dtype=tf.float32)
    value.set_shape([28])
    label = tf.cast(columns[30], tf.int32)

    return value, label


# value, label = read_and_decode("./data/fraud_detection_system.ignore.csv")
filename_queue = tf.train.string_input_producer(["./data/fraud_detection_system.ignore.csv"])
value, label = read_and_decode(filename_queue)

print("value -> ")
print(value)
print("label -> ")
print(label)


# Credit Card Fraud Detection

"""
# 오토 인코더 활용; 금융 거래 분석 모델

# 데이터 셋; [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
# 주요 칼럼 설명
1st. 시간
30th. 비정상 거래 유무 (1: 비정상, 0 정상)
31th. 거래 금액
"""

# 파이썬 2와 파이썬 3 지원
from __future__ import division, print_function, unicode_literals

import pandas as pd
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

# import data
df = pd.read_csv("./data/" + FILE_NAME + ".ignore.csv")
df.describe()

"""
# 시간대별 트랜젝션 분석
# create figure that has 2 graph (in a column), it shares X axis
fig_of_time, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(12, 4))
bins = 100  # 가로의 버킷수

# draw frauded graph. X is time, Y is amount of transaction for the time
ax1.hist(df.Time[df.Class == 1], bins=bins, color="r")
ax1.set_title("비정상거래")

ax2.hist(df.Time[df.Class == 0], bins=bins, color="b")
ax2.set_title("정상거래")

plt.xlabel("시간(초)")
plt.ylabel("트랜젝션(수)")
# plt.show()
utils.save_fig(FILE_NAME + "_시간대별_트랜젝션_분석", root_path=ROOT_PATH)
"""

"""
# 금액별 트랜젝션 분석
fig_of_money, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(12, 4))
bins = 40

ax1.hist(df.Amount[df.Class == 1], bins=bins, color="r")
ax1.set_title("비정상거래")

ax2.hist(df.Amount[df.Class == 0], bins=bins, color="b")
ax2.set_title("정상거래")

plt.xlabel("금액($)")
plt.ylabel("트랜젝션(수)")
plt.yscale("log")  # Y 축 스케일링 기준 칼럼 지정
# plt.show()
utils.save_fig(FILE_NAME + "_금액별_트랜젝션_분석, root_path=ROOT_PATH)
"""

# V1 ~ V28 features
features = df.ix[:, 1:29].columns

"""
# 금액별 트랜젝션 기준, Feature V1 ~ V28 분석
for i, cn in enumerate(df[features]):
    plt.plot(df.Amount[df.Class == 0], df[cn][df.Class == 0], 'bo')
    plt.plot(df.Amount[df.Class == 1], df[cn][df.Class == 1], 'ro')
    # plt.show()
    utils.save_fig(FILE_NAME + "_금액별_트랜젝션_기준_feature_분석_" + str(cn), root_path=ROOT_PATH)
"""

"""
# V1 ~ V28 히스토그램 
# plt.figure(figsize=(12, 28 * 4))
# gs = gridspec.GridSpec(28, 1)  # import matplotlib.gridspec as gridspec
for i, cn in enumerate(df[features]):
    # ax = plt.subplot(gs[i])
    # sns.distplot(df[cn][df.Class == 1], bins=50, color='r')  # import seaborn as sns
    # sns.distplot(df[cn][df.Class == 0], bins=50, color='b')
    # ax.set_xlabel('')
    # ax.set_title('histogram of feature: ' + str(cn))

    fig_of_histogram, (ax1, ax2) = plt.subplots(2, 1, sharex="all", figsize=(12, 4))
    ax1.hist(df[cn][df.Class == 1], bins=50, color="r")
    ax1.set_title("비정상거래")

    ax2.hist(df[cn][df.Class == 0], bins=50, color="b")
    ax2.set_title("정상거래")

    # plt.show()
    utils.save_fig(FILE_NAME + "_히스토그램_of_feature_" + str(cn), root_path=ROOT_PATH)
"""

"""
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

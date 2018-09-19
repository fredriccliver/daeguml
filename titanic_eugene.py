# titanic_eugene.py
# pyCharm IDE 기준으로 작성되었습니다.
#                       date. 2018.09.17.
#                       author. eugene

# 기본 데이터 칼럼 설명
# Survival - 생존 여부. 0이면 사망, 1이면 생존한 것으로 간주합니다.
# Pclass - 티켓 등급. 1등석(1), 2등석(2), 3등석(3)이 있으며, 1등석일수록 좋고 3등석일수록 좋지 않습니다.
# Sex - 성별. 남자(male)와 여자(female)이 있습니다.
# Age - 나이입니다. 틈틈히 빈 값이 존재하며, 소수점 값도 존재합니다.
# SibSp - 해당 승객과 같이 탑승한 형재/자매(siblings)와 배우자(spouses)의 총 인원 수입니다.
# Parch - 해당 승객과 같이 탑승한 부모(parents)와 자식(children)의 총 인원 수입니다.
# Ticket - 티켓 번호입니다. 다양한 텍스트(문자열)로 구성되어 있습니다.
# Fare - 운임 요금입니다. 소수점으로 구성되어 있습니다.
# Cabin - 객실 번호입니다. 많은 빈 값이 존재하며, 다양한 텍스트(문자열)로 구성되어 있습니다.
# Embarked - 선착장입니다. C는 셰르부르(Cherbourg)라는 프랑스 지역, Q는 퀸스타운(Queenstown)이라는 영국 지역, S는 사우스햄튼(Southampton)이라는 영국 지역입니다.


import pandas as pd                                 # 데이터 분석용 패키지
import seaborn as sns                               # 데이터 시각화 패키지
import matplotlib.pyplot as plt                     # 데이터 시각화 패키지
import graphviz                                     # 학습 모델의 시각화를 위한 패키지
from sklearn.tree import DecisionTreeClassifier     # 의사 결정 도구 (머신러닝 알고리즘)
from sklearn.tree import export_graphviz            # graphviz 를 이용하기 위한 패키지

# 데이터 읽어오기
# 참고: pyCharm IDE 기준, 경로 "./data/titanic/train.csv"
# index column 지정, PassengerId

# 학습용 데이터 읽어오기
train = pd.read_csv("./data/titanic/train.csv", index_col="PassengerId")

# 예측할 데이터 읽어오기
test = pd.read_csv("./data/titanic/test.csv", index_col="PassengerId")

# 데이터 확인
print("- 학습용 데이터 확인 -")
print(train)
print("- 예측용 데이터 확인 -")
print(test)

################

# 데이터 분석

################

##
# 성별(Sex) 분석: 남/여 생존자와 사망자를 분석

# [그래프]
sns.countplot(data=train, x="Sex", hue="Survived")
plt.show()

# [도표]: pivot_table
print("- 성별(Sec): 생존자/사망자 분석 -")
print(pd.pivot_table(train, index="Sex", values="Survived"))

# [결론]
# 남자 승객의 생존률: 18.9%
# 여성 승객의 생존률: 74.2%

##
# 객실등급(Pclass) 분석: 1등석/2등석/3등석의 생존자와 사망자를 분석

# [그래프]
sns.countplot(data=train, x="Pclass", hue="Survived")
plt.show()

# [도표]
print("- 객실등급(Pclass): 생존자/사망자 분석 -")
print(pd.pivot_table(train, index="Pclass", values="Survived"))

# [결론]
# 1등급 승객의 생존률: 62.9%
# 2등급 승객의 생존률: 47.2%
# 3등급 승객의 생존률: 24.2%

##
# 탑승한 위치, 선착장(Embarked) 분석: 탑승위치에 따른 생존자와 사망자를 분석
# C: 셰르부르(Cherbourg)라는 프랑스 지역
# Q: 퀸스타운(Queenstown)이라는 영국 지역
# S: 사우스햄튼(Southampton)이라는 영국 지역

# [그래프]
sns.countplot(data=train, x="Embarked", hue="Survived")
plt.show()

# [도표]
print("- 선착장(Embarked): 생존자/사망자 분석 -")
print(pd.pivot_table(train, index="Embarked", values="Survived"))

# [결론]
# C 지역 탑승객의 생존률: 55.3%
# Q 지역 탑승객의 생존률: 38.9%
# S 지역 탑승객의 생존률: 33.6%

##
# 나이(Age)와 운임요금(Fare) 분석: 나이와 운인요금에 따른 생존자/사망자 분석
# fit_reg: 회귀선, 회귀선은 봐도 의미가 없으므로 false 처리

# [그래프]
sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)
plt.show()

# 운임요금(Fare)이 $500 이상인 점이 3개 존재 > 과적합 의심 > 제거
# [ 과적합의심되는 데이터를 보고 아웃라이어(Outlier)라고 합니다. ]
# 판다스 색인을 이용하여 운임요금 $500 미만인 데이터만 가져옴
processed_fare = train[train["Fare"] < 500]

# 3개 데이터 제거 확인: (891,11) -> (888, 11)
# print(train.shape, processed_fare.shape)

# [그래프]
sns.lmplot(data=processed_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
plt.show()

# [결론]
# 어리면 생존률이 높고, 나이가 많은 경우 사망률이 높다는 것을 특정지을 수 있음

################

# 데이터 전처리 (Data Pre-processing)

# [scikit-learn](scikit-learn.org) 에서 제공하는 기본 알고리즘 이용 (아래 조건 만족해야 함)
# 1. 모든 데이터는 숫자(정수형, 소수점 등)로 구성되어 있어야 한다.
# 2. 데이터에 빈 값이 없어야 한다.

################

##
# 성별(Sex) 전처리
# 성별은 현재, male/female 이므로 Decision Tree 가 이해할 수 있는 숫자로 변환

# male: 0
# female: 1

# set train
train.loc[train["Sex"] == "male", "Sex_encoded"] = 0
train.loc[train["Sex"] == "female", "Sex_encoded"] = 1

# set test
test.loc[test["Sex"] == "male", "Sex_encoded"] = 0
test.loc[test["Sex"] == "female", "Sex_encoded"] = 1

##
# 운임요금(Fare) 전처리
# 운임요금은 현재 테스트 데이터(test.csv)에 공백이 존재하므로 값을 채우도록 함

# 평균값 혹은 0

# set train
train["Fare_filled"] = train["Fare"]

# set test
test["Fare_filled"] = test["Fare"]
test.loc[test["Fare"].isnull(), "Fare_filled"] = 0

# 확인
# print(test.loc[test["Fare"].isnull(), ["Fare", "Fare_filled"]])

# [팁]
# 판다스에서 loc 를 쓰면, 다음과 같은 같은 의미 입니다.
# 속도는 loc 를 사용하는 쪽이 더 빠릅니다. (참고, https://stackoverflow.com/)
# table["column1", "column2"] == table.loc["column1", "column2"]

##
# 선착장(Embarked) 전처리
# 문자열이기 때문에 Decision Tree 가 이해할 수 있도록 숫자로 변환

# 주의! 만약 C = 1, Q = 2, S = 3 이라고 그냥 처리 해버리면 아래와 같은 오류 발생가능
# C + Q ==? S

# 따라서, 인코딩 한 데이터를 서로 더할 수 없도록 처리가 필요
# 이때 사용하는 것이 [One Hot Encoding](https://minjejeon.github.io/learningstock/2017/06/05/easy-one-hot-encoding.html)

# C == [1, 0, 0]
# S == [0, 1, 0]
# Q == [0, 0, 1]

# set train
train["Embarked_C"] = train["Embarked"] == "C"
train["Embarked_Q"] = train["Embarked"] == "Q"
train["Embarked_S"] = train["Embarked"] == "S"

# set test
test["Embarked_C"] = test["Embarked"] == "C"
test["Embarked_Q"] = test["Embarked"] == "Q"
test["Embarked_S"] = test["Embarked"] == "S"

# 참고! True == 1 / False == 0

################

# 학습(Train, fitting)
# Decision Tree == [지도학습(Supervised Learning)](http://solarisailab.com/archives/1785)

# Decision Tree 로 학습/판단하기 위해서는 아래 두 가지 종류의 데이터가 필요
# 1. Label: 맞춰야 하는 정답
# 2. Feature: 정답을 맞추는데 판단을 할 수 있는 지표

################

# Titanic 생존자 예측학기 모델의 Feature / Label 선정

# Label: 생존여부(Survived)
label_name = "Survived"

# Feature: 티켓등급(Pclass), 가공한 성별(Sex_encoded), 가공한 운임요금(Fare_filled), 가공한 선착장(Embarked_C, Embarked_Q, Embarked_S)
feature_names = ["Pclass", "Sex_encoded", "Fare_filled", "Embarked_C", "Embarked_Q", "Embarked_S"]

# train_features == X_train
# train_label = y_train
train_features = train[feature_names]
train_label = train[label_name]

##
# 학습 모델 생성
# Decision Tree 는 내부적으로 트리를 이용하여 학습/판단
# max_depth 를 설정하면 트리의 깊이를 조절/제한 가능
decision_tree_model = DecisionTreeClassifier(max_depth=5)

# 모델 확인
print("- 학습 모델 확인 -")
print(decision_tree_model)

# 학습(Train, fitting)
decision_tree_model.fit(train_features, train_label)

##
# 학습 모델의 시각화
# graphviz 이용
# sklearn.tree.export_graphviz 를 이용하여 Decision Tree 시각화
graphviz.Source(
        export_graphviz(decision_tree_model,                 # 시각화할 트리
                        feature_names=feature_names,         # 트리를 만들때 사용한 feature(s)
                        class_names=["Perish", "Survived"],  # 시각화에 표현할 결과: Perish(사망), Survived(생존)
                        out_file=None)                       # 시각화한 결과를 저장할 파일명
)

################

# 예측(Predict)

################

# 예측할 데이터 추출: 학습된 Features 와 동일해야 함
# test_features == X_test
test_features = test[feature_names]

# 학습 모델로 판단 데이터 예측
predictions = decision_tree_model.predict(test_features)

# 예측결과 확인
print("- 예측 결과 데이터 확인 -")
print(predictions)

################

# 예측결과 파일로 저장

################

# 캐글(Kaggle) 에 제출하기 해당 포맷을 이용할 예정
kaggle_submission = pd.read_csv("data/titanic/gender_submission.csv", index_col="PassengerId")

# 캐글 제출용 데이터 형식 확인
print("- 캐글(Kaggle) 제출용 데이터: 교체 전 -")
print(kaggle_submission)

# 결과 데이터(생존여부, Survived)만 교체
kaggle_submission["Survived"] = predictions

# 교체된 데이터 확인
print("- 캐글(Kaggle) 제출용 데이터: 교체 후 -")
print(kaggle_submission)

# 제출 파일 생성/저장
kaggle_submission.to_csv("data/titanic/kaggle_submission_eugene.csv")

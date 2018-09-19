import sys
import pandas as pd
import keras
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.utils import np_utils

train = pd.read_csv("data/titanic/train.csv")
test = pd.read_csv("data/titanic/test.csv")
train = train.append(test) ## test 데이터도 학습에 이용.

# inplace=True 로 해야 모든 컬럼에 대해 fillna 가 이뤄짐.
train.fillna(0, inplace=True) 
test.fillna(0, inplace=True)

# cols = PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

# 데이터 전처리 : One Hot Encoding
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
enc = encoder.fit(train[['Embarked']].astype(str))
train = pd.merge(train, pd.DataFrame(enc.transform(train[['Embarked']].astype(str))) , right_index=True, left_index=True)
test = pd.merge(test, pd.DataFrame(enc.transform(test[['Embarked']].astype(str))) , right_index=True, left_index=True)
enc = encoder.fit(train[['Sex']].astype(str))
train = pd.merge(train, pd.DataFrame(enc.transform(train[['Sex']].astype(str))) , right_index=True, left_index=True)
test = pd.merge(test, pd.DataFrame(enc.transform(test[['Sex']].astype(str))) , right_index=True, left_index=True)
train.pop('Embarked'), test.pop('Embarked')
train.pop('Sex'), test.pop('Sex')

# 데이터 전처리 : 학습에 필요없는 column 제거.
train.pop('Name'), test.pop('Name')
train.pop('Ticket'), test.pop('Ticket')
train.pop('Cabin'), test.pop('Cabin')

# label
y_train = train[['Survived']]

# feature
train.pop('Survived')
x_train = train
x_test = test

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=len(x_train.columns)))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1, activation='relu'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

print(model.summary)

hist = model.fit(x_train, y_train, epochs=10, batch_size=10)

classes = model.predict(x_test)
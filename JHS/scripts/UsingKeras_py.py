#%% import
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
# from pylab import *
from keras import models, layers, optimizers, metrics, utils, callbacks

#%% csv 불러오기
training = pd.read_csv('../input/train.csv')
testing = pd.read_csv('../input/test.csv')

#%% 데이터 요약
# msno.matrix(training)
# plt.show()
print(training.shape)
print(training.describe())
print(training.dtypes)

print(training.keys())
print(testing.keys())

#%% 데이터 분할
# x_train (학습에 쓸 문제), y_train (학습에 쓴 문제 정답)
# x_test (제출할 문제), y_test (추론해야 하는 것. 제출할 정답)
x_train = training.drop(columns=['Survived'])
y_train = training['Survived']
x_test = testing.copy()

#%% column 추가 (내 아이디어 아님)
x_train['People'] = x_train['SibSp'] + x_train['Parch'] + 1
x_train['IsAlone'] = x_train['People'].apply(lambda x: 1 if x == 1 else 0)
x_train['HasCabin'] = x_train['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

x_test['People'] = x_test['SibSp'] + x_test['Parch'] + 1
x_test['IsAlone'] = x_test['People'].apply(lambda x: 1 if x == 1 else 0)
x_test['HasCabin'] = x_test['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

#%% 결측치 제거 (mean? median?)
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())

x_train['Embarked'] = x_train['Embarked'].fillna('S')

x_test['Fare'] = x_test['Fare'].fillna(testing['Fare'].mean())

#%% one-hot encoding
x_train['Sex'] = x_train['Sex'].map({'male': 0, 'female': 1})
x_train['Embarked'] = x_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

x_test['Sex'] = x_test['Sex'].map({'male': 0, 'female': 1})
x_test['Embarked'] = x_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

#%% 예측에 안 쓸 칼럼 제거
x_train = x_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
x_test = x_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

#%% create model
model = models.Sequential()
model.add(layers.Dense(units=64, activation='relu', input_dim=10))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dense(units=8, activation='relu'))
model.add(layers.Dense(units=4, activation='relu'))
model.add(layers.Dense(units=2, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

utils.plot_model(model, to_file='model.png', show_shapes=True)

#%% fitting
# early_stopping = callbacks.EarlyStopping(monitor='val_acc')
fit_history = model.fit(x_train, y_train, epochs=1000, validation_split=0.2, shuffle=True, batch_size=64)

#%% let's visualize
acc = fit_history.history['acc']
val_acc = fit_history.history['val_acc']
loss = fit_history.history['loss']
val_loss = fit_history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.plot(epochs, loss, 'ro', label='loss')
plt.plot(epochs, val_loss, 'r', label='validation loss')

plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

#%% predict
prediction = model.predict_classes(x_test)
ids = testing['PassengerId'].copy()
new_output = ids.to_frame()
new_output['Survived'] = prediction
new_output.to_csv('../output/UsingKeras_py.csv', index=False)

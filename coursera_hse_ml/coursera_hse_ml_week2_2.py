import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('perceptron-train.csv', header=None).ix[:, 1:]
X_test = pd.read_csv('perceptron-test.csv', header=None).ix[:, 1:]
y_train = pd.read_csv('perceptron-train.csv', header=None).ix[:, 0]
y_test = pd.read_csv('perceptron-test.csv', header=None).ix[:, 0]

model = Perceptron(random_state=241)
model.fit(X_train, y_train)
acc_before = accuracy_score(y_test, model.predict(X_test))
print(acc_before)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Perceptron(random_state=241)
model.fit(X_train_scaled, y_train)
#predict_train=model.predict(X_train_scaled)
predict_test=model.predict(X_test_scaled)

acc_after = accuracy_score(y_test, predict_test)
print(acc_after)
print(acc_after - acc_before)
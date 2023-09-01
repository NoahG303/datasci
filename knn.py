# KNN classifies a point based on the labels of the k nearest points to it
# smaller k can lead to overfitting, larger k can lead to slower computation + higher bias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.datasets import load_iris

df = pd.read_csv(r"Fun\Python\new\datasci\datasets\fake_bills.csv", sep=';')
df = df.dropna()
X = df.drop('is_genuine', axis=1).to_numpy()
y = df['is_genuine'].to_numpy()

# df = pd.read_csv(r"Fun\Python\new\datasci\datasets\health care diabetes.csv")
# X = df.drop('Outcome', axis=1).to_numpy()
# y = df['Outcome'].to_numpy()

# irisData = load_iris()
# X = irisData.data
# y = irisData.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

neighbors = [i for i in range(1, 25, 2)]
# train_accuracy = []
# test_accuracy = []
cross_val_means = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    # knn.fit(X_train, y_train)
    # train_accuracy.append(knn.score(X_train, y_train))
    # test_accuracy.append(knn.score(X_test, y_test))
    cross_val_means.append(np.mean(scores))

best_score = max(cross_val_means)
best_k = cross_val_means.index(best_score)*2+1
print(f"Best k: {best_k}, value: {best_score:.8f}.")

# best_score1 = max(test_accuracy)
# best_k1 = test_accuracy.index(best_score1)*2+1
# print(f"Best k1: {best_k1}, value1: {best_score1}.")

# plt.plot(neighbors, test_accuracy, label = 'Testing Dataset Accuracy')
# plt.plot(neighbors, train_accuracy, label = 'Training Dataset Accuracy')
plt.plot(neighbors, cross_val_means, label = 'Training Cross Validation Score')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
test_acc = best_knn.score(X_test, y_test)
print(f"Test accuracy with best k ({best_k}): {test_acc:.8f}.")
# KNN classifies a point based on the labels of the k nearest points to it
# smaller k can lead to overfitting, larger k can lead to slower computation + higher bias

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# df = pd.read_csv(r"pathtocsv")
# X = df.drop('diagnosis', axis=1).to_numpy()
# y = df['diagnosis'].to_numpy()

irisData = load_iris()
X = irisData.data
y = irisData.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

neighbors = [i for i in range(1, 25)]
train_accuracy = []
test_accuracy = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

best_score = max(test_accuracy)
best_k = test_accuracy.index(best_score)+1
print(f"Best k: {best_k}, value: {best_score}.")

plt.plot(neighbors, test_accuracy, label = 'Testing Dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Dataset Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
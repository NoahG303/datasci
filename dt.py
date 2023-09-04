# decision tree essentially creates a flow chart of properties to predict the outcome

import pandas as pd

# from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# iris = load_iris()
# X = iris.data
# y = iris.target

df = pd.read_csv(r"Fun\Python\new\datasci\datasets\car_evaluation.csv")
df = df.dropna()

# for column in df.columns:
#    print(column, set(df[column].tolist()))

to_floats = {"low": 1, "med": 2, "high": 3, "vhigh": 4, "5more": 5, "more": 5, "small": 1, "big": 3, "unacc": 0, "acc": 1, "good": 2, "vgood": 3, "2": 2, "3": 3, "4": 4}
for column in df.columns:
    df.replace({column: to_floats},inplace=True)

X = df.drop('choice', axis=1).to_numpy()
y = df['choice'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
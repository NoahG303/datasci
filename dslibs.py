import pandas as pd # pandas!
 
data_dict = {
    'Name': ['Noah', 'Otherperson'],
    'Age': [21, 17],
    'Job': ['Data Scientist', 'Fool'],
    'Favorite Color': ['Pink', 'Yellow']
}
df = pd.DataFrame(data_dict)
# print(df) # all
# print(df[['Age', 'Job']]) # just these columns

nfl_data = pd.read_csv(r"C:\Users\noahg\Desktop\School\code\Fun\Python\new\datasci\nfl_teams.csv", index_col ="Name")
# print(nfl_data) # all
# print(nfl_data.loc["San Francisco 49ers"]) # sf info
# print(nfl_data["Conference"]) # name and conf

#########################################################################################

import numpy as np # numpy!

a = np.array([1, 2, 3, 5]) # 1 2 3 5
# print(a[1])# 2
# print(a[0:2]) # 1 2
# print(a[1:]) # 2 3 5
# print(a[-2:]) # 3 5
# print(a[a>2]) # 3 5
# print(a[a%2==0]) # 2
# print(a[(a%2==0) | (a > 2)]) # 2 3 5
z = np.zeros(3) # 0 0 0
o = np.ones(2) # 1 1
r = np.arange(4) # 0 1 2 3
e = np.arange(2, 9, 2) # 2 4 6 8
f = e + 1
# print(f) # 3 5 7 9
arr_ex = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]],  [[0, 1, 2, 3], [4, 5, 6, 7]],  [[0, 1, 2, 3], [4, 5, 6, 7]]])
# print(arr_ex) # 3D array, 3 elements, each of which is 2 arrays
# print(arr_ex.ndim) # 3 dims
# print(arr_ex.size) # 24 elements
# print(arr_ex.shape) # 3 x 2 x 4
data1 = np.array([1.0, 2.0])
# print(data1*1.6) # 1.6 3.2
# print(data1.min()) # 1
# print(data1.max()) # 2
# print(data1.sum()) # 3
rng = np.random.default_rng()
x = rng.integers(5, size=(2, 4)) 
# print(x) # 2x4 random nums [0,5)
dupes = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
uniques, occurrence_count = np.unique(dupes, return_counts=True)
# print(uniques) # set, basically
# print(occurrence_count) # uniques order, shows amount
arr1 = np.arange(6)
arr2 = arr1.reshape((2, 3))
arr3 = arr2.transpose()
arr4 = arr2.T
# print(arr1) # 1x6
# print(arr2) # 2x3
# print(arr3) # 3x2
# print(arr4) # 3x2

#########################################################################################

from matplotlib import pyplot as plt # matplotlib!

x = [5, 2, 9, 4, 7]
y = [10, 5, 8, 4, 2]
# plt.scatter(x,y) # scatterplot
# plt.bar(x,y) # bar graph
# plt.plot(x,y) # line
plt.show()

#########################################################################################

# sklearn
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data # features
y = iris.target # responses

# feature_names = iris.feature_names
# target_names = iris.target_names
# print("Feature names:", feature_names) # features of data to split on
# print("Target names:", target_names) # categories to decide on
# print("\nType of X is:", type(X)) # np array
# print("\nFirst 5 rows of X:\n", X[:5]) # first 5 rows (flower exs) of features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
 
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn import metrics
# print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred))

sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
preds = knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
# print("Predictions:", pred_species)

# import joblib
# joblib.dump(knn, 'iris_knn.pkl')
# knn = joblib.load('iris_knn.pkl') 

# nfl_data2 = pd.read_csv(r"C:\Users\noahg\Desktop\School\code\Fun\Python\new\datasci\nfl_teams.csv")
# print("Shape:", nfl_data.shape)
# print("\nFeatures:", nfl_data.columns)
# X = nfl_data[nfl_data.columns[:-1]] # features thru last column
# y = nfl_data[nfl_data.columns[-1]] # last one (s/b result, here its just division bc no result exists)
# print("\nFeature matrix:\n", X.head())
# print("\nResponse vector:\n", y.head())

#########################################################################################

# pytorch/tensorflow?
# SQL
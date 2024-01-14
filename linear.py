import pandas as pd 
import numpy as np  
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv('data/student-mat.csv', sep=";")

# Check the dataset 
# print(data.head())

# Feature Variables
data = data[['G1','G2','G3','studytime','failures','absences']]

# Variable to Predict
predict = "G3"

# Set up x and y as array
x = np.array(data.drop(predict, axis=1))
y = np.array(data[predict]) 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size= 0.1)


# best = 0
# for _ in range(30):
#     # Split the dataset
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size= 0.1)

#     # Algorithm for Linear Regression
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train,y_train)
#     acc = linear.score(x_test,y_test)
#     print("Accuracy: ", acc)

#     if acc > best:
#         # Saving the model
#         best = acc
#         with open("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)


# Reading the saved model
pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

# Model Evaluation (Provide Metrics of the Model)

print("Coefficient: \n", linear.coef_)
print("Intercept \n", linear.intercept_)

# Performing Predictions Based on the Model
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


style.use("ggplot")
p = "G1"
plt.scatter(data[p],data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
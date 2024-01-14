import pandas as pd 
import numpy as np  
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

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

# Split the dataset
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size= 0.1)

# Algorithm for Linear Regression
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train)

# Model Evaluation (Provide Metrics of the Model)
acc = linear.score(x_test,y_test)
print("Accuracy: ", acc)

print("Coefficient: \n", linear.coef_)
print("Intercept \n", linear.intercept_)


# Performing Predictions Based on the Model
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

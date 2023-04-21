# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.

##Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
import the required libraries and read the dataframe.
Assign hours to X and scores to Y.
Implement training set and test set of the dataframe.
Plot the required graph both for test data and training data.
Find the values of MSE , MAE and RMSE.

##Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NARENDRAN B
RegisterNumber: 212222240069

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('dataset/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

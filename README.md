# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:YOGABHARATHI S 
RegisterNumber:212222230179
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)  
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
# df.head()
![Screenshot 2023-08-24 140414](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/15f956d3-9238-4bc8-956d-ed84f4957dd0)
# df.tail()
![Screenshot 2023-08-24 140557](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/cc24c9ae-a6ff-406e-a2d7-d019ee8a1368)
# Array value of X
![Screenshot 2023-08-24 140657](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/657feba3-1b3a-4d86-a33f-09b62fe985e5)
# Array value of Y
![Screenshot 2023-08-24 140750](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/4b7e3935-4206-4065-973e-a2410a34c7de)
# Values of y prediction 
![Screenshot 2023-08-24 151025](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/03bc82d8-beb6-428f-8af3-77b913b636a0)

# Array values of Y test
![Screenshot 2023-08-24 151039](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/c79327ab-0760-4932-b059-82dde52fa8be)

# Training set graph
![Screenshot 2023-08-24 151052](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/dcc674db-03a0-42e5-b1fa-7c970b0253a9)

# Test set graph
![Screenshot 2023-08-24 152423](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/c3b4148f-0bd9-4bc2-9860-323c79a7b952)

# Values of MSE,MAE and RMSE
![Screenshot 2023-08-24 151113](https://github.com/Yogabharathi3/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118899387/ebf187be-1101-4376-8ea7-3b153bcd2b0a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

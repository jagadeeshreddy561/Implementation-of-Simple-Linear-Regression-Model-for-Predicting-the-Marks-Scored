# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step1
Import the standard Libraries.

### Step2
Set variables for assigning dataset values.

### Step3
Import linear regression from sklearn.

### Step4
Assign the points for representing in the graph.

### Step5
Predict the regression for marks by using the representation of the graph.

### Step6
Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
## Developed by: jagadeesh reddy
## RegisterNumber: 212222240059

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:


![ML_ex-2 1](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/f9013457-fc5b-4fb8-bc21-2c015cdb6f0e)



![ML_exp-2 2](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/d9db4215-976b-43fc-b281-d93f6d4539b7)



![ML_exp-2 3](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/4df49453-d9af-4fc4-89b3-de840074c041)



![ML_exp-2 4](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/0f5f269c-60ac-4e02-8fbb-9218f794b494)



![ML_exp-2 5](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/5b600805-2a89-44cc-a6bd-1166fb7adb27)



![ML_exp-2 6](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/4502bc3c-ed9f-45df-9a1f-3ed7ca0ab1df)



![ML_exp-2 7](https://github.com/jagadeeshreddy561/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120623104/894f2e79-1b49-4999-910f-c72fb59baf47)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

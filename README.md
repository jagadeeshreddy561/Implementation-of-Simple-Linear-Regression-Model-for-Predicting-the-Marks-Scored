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
## Developed by: Mallu Jagadeeswar Reddy
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

### df.head()

![ML_ex-2 1](https://github.com/user-attachments/assets/78eb9916-bb3f-457a-97b9-f554d5d24e85)

### df.tail()

![ML_exp-2 2](https://github.com/user-attachments/assets/8a59626e-2189-4c56-aafc-bef498d15809)

### Array of value X

![ML_exp-2 3](https://github.com/user-attachments/assets/0c366346-37dd-43fe-b24d-f011b6f1c2db)

### Array of value Y

![ML_exp-2 4](https://github.com/user-attachments/assets/19c9a3a2-04f7-48c1-b238-d648b89ca1d8)

### Values of Y prediction

![ML_exp-2 5](https://github.com/user-attachments/assets/49b55934-3dd9-4646-8f9c-23ae468b4070)

### Array values of Y test

![ML_exp-2 6](https://github.com/user-attachments/assets/0d8982a0-e2f4-4382-a78e-f343c11c0180)

### Training Set Graph & Test Set Graph

![ML_exp-2 7](https://github.com/user-attachments/assets/e01462cd-7347-4d40-980f-c04cd398a102)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

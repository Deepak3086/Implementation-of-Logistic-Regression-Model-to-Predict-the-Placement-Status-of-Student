# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
```
```

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DEEPAK JG
RegisterNumber: 212224220019
*/

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```
## Output:
![image](https://github.com/user-attachments/assets/91040c0d-d050-4b05-bacd-2a8faf187d2d)

![image](https://github.com/user-attachments/assets/f821a910-d4ad-4f59-a4c7-d9e44079a298)

![image](https://github.com/user-attachments/assets/ea4f97b2-e206-4bb5-8216-006fee1e70fc)

![image](https://github.com/user-attachments/assets/706bf827-1fb7-4b86-a2e1-7d431b66d7a6)

![image](https://github.com/user-attachments/assets/487d9dc4-ebb1-4358-8c5a-3585d20551b0)

![image](https://github.com/user-attachments/assets/83c46d47-08bd-488e-89cd-e24f869fe89d)

![image](https://github.com/user-attachments/assets/53647285-fd07-4ec3-8356-ba196c4f1b53)

![image](https://github.com/user-attachments/assets/e72635a6-b2b9-41c0-9820-de2e31ded565)

![image](https://github.com/user-attachments/assets/c265b363-6f1d-4f5c-977a-e45c5d7d6fef)

![image](https://github.com/user-attachments/assets/b73b484a-f487-41ab-8a69-e9a1897e9116)

![image](https://github.com/user-attachments/assets/ce0282f1-76db-495d-97da-c5baf610dd63)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

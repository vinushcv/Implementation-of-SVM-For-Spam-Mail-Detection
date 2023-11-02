# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import chardet
2. Read the dataset
3. Import SVC from sklearn
4. Fit the data in the model and run the algorithm

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Vinush.CV
RegisterNumber:  212222230176
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows - 1252')

data.head()

data.info()

data.isnull().sum()

x=data['v1'].values
y=data['v2'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Result output:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121243595/21a2734d-7b70-4787-9326-25a49255b1c3)

### data head():
![image](https://github.com/ShanmathiShanmugam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121243595/bbc6f73e-9ff7-45d9-8171-ea9e542a3653)

### data.info():
![image](https://github.com/ShanmathiShanmugam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121243595/985a8fb9-fcbd-4ad5-8ca9-d0f53d3ea7ac)

### data.isnull().sum():
![image](https://github.com/ShanmathiShanmugam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121243595/4e2d42d9-3051-4b27-ab62-70dd1cd9398c)

### Y_prediction value:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121243595/1f9f5616-5aff-4a15-b3e4-99034741b9ae)

### Accuracy value:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121243595/cf44edad-1fbb-464a-8b3e-b54a59d4ec66)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

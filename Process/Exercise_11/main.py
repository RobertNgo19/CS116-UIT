# -*- coding: utf-8 -*-
"""19522028_BT11.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GX1oXjkHreWUegwze7qq5pKQRcQCDwQ2
"""

import cv2 as cv
import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')
data

x = data.iloc[:,:-1].values
x.shape

y= data.iloc[:,-1].values
y.shape

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=1)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_pre = scaler.fit_transform(x_train)

X_val_pre = scaler.transform(x_val)
X_test_pre = scaler.transform(x_test)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

c = [0.1,1,10,100,1000]
kernel = ['rbf','poly','sigmoid','linear']
#degree = [1,2,3,4,5,6]
gamma = [1, 0.1, 0.01, 0.001, 0.0001]
acc = 0
configs = []
best_config = None
model = None
for h1 in c:
  for h2 in kernel:
      for h4 in gamma:
        svm = SVC(C=h1, kernel=h2, gamma=h4)
        svm.fit(X_train_pre, y_train)

        y_val_pred = svm.predict(X_val_pre)
        val_score = f1_score(y_val, y_val_pred)

        config = [h1,h2,h4,val_score]
        configs.append(config)
        #print("C: {}, kernel: {}, gamma: {}, val: {}".format(h1,h2,h4,val_score))
        if acc < val_score:
          acc = val_score
          model = svm
          best_config = config
          
#print("C: {}, kernel: {}, gamma: {}, acc: {}".format(para_C,para_kernel,para_gamma,acc))

print(best_config)

# test
y_test_pred = model.predict(X_test_pre)
print(classification_report(y_test, y_test_pred))
#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION WITH CLEANED DATA

# In[1]:


# IMPORTING REQUIRED PACKAGES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# READ DATASET

df = pd.read_csv('Iris.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df =  df.drop(['Id'],axis = 1)
df.columns


# In[5]:


X = df.iloc[ : , :-1]
Y = df.iloc[ : ,-1]
X.shape,Y.shape


# In[6]:


# DATA VISUALIZATION

for i in X.columns:
    i = str(i)
    print(i)
    sb.distplot(X[i])
    plt.show()


# In[7]:


# SPLITTING DATA

from sklearn.model_selection import train_test_split as tts

X_train,X_test,Y_train,Y_test = tts(X,Y,test_size = 0.33,random_state = 42)
X_train.shape,Y_train.shape


# In[8]:


# IMPLEMENTING MODEL

from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(X_train,Y_train)


# In[9]:


# TRAINING ACCURACIES

y_train_pred = reg.predict(X_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy_score(Y_train,y_train_pred)


# In[10]:


confusion_matrix(Y_train,y_train_pred)


# In[11]:


# TEST ACCURACIES

y_test_pred = reg.predict(X_test)

accuracy_score(Y_test,y_test_pred)


# In[12]:


confusion_matrix(Y_test,y_test_pred)


# In[13]:


print(classification_report(Y_test,y_test_pred))


# In[14]:


# TESTING OUR OWN DATA

reg.predict([[5.1,3.5,1.4,0.2]])


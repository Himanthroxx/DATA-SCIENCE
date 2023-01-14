#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("Salary_Data.csv")
data


# In[3]:


X = data['YearsExperience']
Y = data['Salary']
len(X),len(Y)


# In[4]:


X = X.values.reshape(-1,1)
Y = Y.values.reshape(-1,1)


# In[5]:


X.shape,Y.shape


# In[6]:


import seaborn as sb


# In[7]:


plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
sb.distplot(X)

plt.subplot(1,2,2)
sb.distplot(Y)


# In[8]:


plt.scatter(X,Y)


# In[9]:


data.describe()


# In[10]:


plt.boxplot(Y)


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# In[13]:


import sklearn
from sklearn.model_selection import train_test_split as tts


# In[14]:


X_train,X_test,Y_train,Y_test = tts(X,Y,test_size=0.33,random_state = 2)
len(X_train),len(Y_train)


# In[15]:


plt.figure(figsize = (10,6))

plt.subplot(1,2,1)
sb.distplot(X_train)


# In[16]:


plt.title('simple linear regression')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.scatter(X_train,Y_train,color = 'g', marker = '*')
plt.show()


# In[17]:


X_train.shape


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


reg = LinearRegression()


# In[20]:


reg.fit(X_train,Y_train)


# In[21]:


reg.coef_


# In[22]:


reg.intercept_


# In[23]:


y_train_predictions = reg.predict(X_train)


# In[24]:


y_train_predictions


# In[25]:


from sklearn.metrics import r2_score as rs


# In[26]:


rs(Y_train,y_train_predictions)


# In[27]:


y_test_predictions = reg.predict(X_test)
y_test_predictions


# In[28]:


rs(Y_test,y_test_predictions)


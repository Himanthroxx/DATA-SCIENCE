#!/usr/bin/env python
# coding: utf-8

# MULTIPLE LINEAR REGRESSION WITH CLEANED DATA

# In[1]:


# IMPORTING REQUIRED PACKAGES


# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[3]:


# READING DATASET


# In[4]:


df = pd.read_csv('50_Startups.csv')


# In[5]:


df.head(10)


# In[6]:


# CHECKING FULL INFORMATION


# In[7]:


df.info()


# In[8]:


df['State'].unique()


# In[9]:


# creating dummies 


# In[10]:


a = pd.get_dummies(df['State'] , drop_first=True)
a


# In[11]:


df = pd.concat([df , a] , axis = 1)


# In[12]:


df.head()


# In[13]:


df = df.drop(['State'],axis = 1)
df.head()


# ### spliting the data 

# In[14]:


X = df.drop(['Profit'],axis = 1)  # independent 
y = df['Profit'] # dependent 


# In[15]:


X.head()


# In[16]:


y.head()


# In[17]:


# checking null values


# In[18]:


df.isnull().sum()


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


df.shape


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)


# In[22]:


len(X_train) , len(y_train)


# In[23]:


X_train.head()


# In[24]:


# import and fit 


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


reg = LinearRegression()


# In[27]:


# y = m1x1 + m2x2 + m3x3 + m4x4 + m4x5 + c


# In[28]:


reg.fit(X_train,y_train)


# In[29]:


# for m(slopes)

reg.coef_


# In[30]:


# for c(intercept)

reg.intercept_

# our model is y = 8.64329624e-01 * X1 + 7.12539493e-03 * X2 + 3.06409736e-02 * X3 + -5.50485207e+02 * X4 + -6.33665237e+03 * X5 + 41591.69166575266
# In[31]:


# Making predictions for training data


# In[32]:


X_train.shape


# In[33]:


y_train.shape


# In[34]:


y_train_1 = y_train.values.reshape(-1,1)
y_train_1.shape


# In[35]:


y_train_pred = reg.predict(X_train)
y_train_pred


# In[38]:


train_data_comparision = pd.DataFrame({'y_train_Actual': y_train , 'y_train_predictions': y_train_pred})
train_data_comparision


# In[42]:


# training accuracy


# In[43]:


from sklearn.metrics import r2_score

r2_score(y_train,y_train_pred)


# Test Data Performance

# In[44]:


y_test_pred = reg.predict(X_test)


# In[45]:


r2_score(y_test,y_test_pred)


# In[46]:


# checking with real data 


# In[53]:


import warnings
warnings.filterwarnings('ignore')


# In[54]:


reg.predict([[1.1,10.9,12,0,0]])


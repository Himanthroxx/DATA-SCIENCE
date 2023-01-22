#!/usr/bin/env python
# coding: utf-8

# ##SIMPLE LINEAR REGRESSION WITH CLEANED DATA

# ## Import Packages 
# 

# In[1]:


import os
import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# ## Load The data 

# In[2]:


df = pd.read_csv('Salary_Data.csv')


# In[3]:


df.head()


# In[4]:


df.info()   # finding the information 


# In[5]:


# finding the null values

df.isnull().sum()


# ## Complete data Visualization 

# In[6]:


plt.title('Simple Linear Regression')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.scatter(df['YearsExperience'] , df['Salary'] , color = 'r' , marker='*')
plt.show()


# In[7]:


len(df)


# In[8]:


df


# In[9]:


## Represent Independent varriables with x and dependent variables with y 


# In[10]:


df['YearsExperience'].shape 


# In[11]:


X = df['YearsExperience']  # Independent 
y = df['Salary']         # Dependent 


# In[12]:


X = X.values.reshape(-1,1)
X


# ## Spliting the data into train and test 

# In[13]:


# import sklearn -> scikit learn 


# In[14]:


import sklearn 


# In[15]:


# for spliting the data sklearn providing a method called train_test_split 


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.33 ,random_state=10)


# In[18]:


X_train


# In[19]:


y_train


# In[20]:


len(X_train) , len(y_train)


# ## Plot Training data

# In[21]:


plt.title('Simple Linear Regression')
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.scatter(X_train, y_train , color = 'b')
plt.show()


# ## Feeding to the Algorithm

# In[22]:


# Simple Linear Regression y = mx + c


# In[23]:


from sklearn.linear_model import LinearRegression 


# In[24]:


reg = LinearRegression()


# In[25]:


# Giving data to y = mx + c


reg.fit(X_train , y_train)


# In[26]:


# for m(slope)
reg.coef_


# In[27]:


# for c(intercept)
reg.intercept_


# ## y = 9514.52350354 * x + 25630.510121837848  is our Model to do predictions 

# In[28]:


## checking performance of training data 


# In[29]:


y_train_predictions =reg.predict(X_train)


# In[30]:


y_train_predictions


# In[31]:


y_train


# In[32]:


y_train.shape


# In[33]:


X_train


# In[34]:


X_train.shape


# In[35]:


X_train_1 = X_train.flatten() # convert any dimension to normal values 
X_train_1.shape


# In[36]:


train_data_comparision = pd.DataFrame({'X_train':X_train_1, 'y_train_Actual': y_train , 'y_train_predictions': y_train_predictions})


# In[37]:


train_data_comparision


# In[38]:


# Difference between Actual and prediction is called residual 

# Residual formula = (y - y_pred)**2 


# In[39]:


plt.title('Simple Linear Regression')
plt.xlabel('X_train')
plt.ylabel('y_train')
plt.scatter(X_train, y_train , color = 'r' , marker='*')
plt.plot(X_train , y_train_predictions,color = 'b' , marker = 'o')
plt.show()


# ## Finding the accuracy for training data 

# In[40]:


from sklearn.metrics import r2_score


# In[41]:


r2_score(y_train , y_train_predictions)


# ### Test data 

# In[42]:


# based on test accuracy we will decide model performance 


# In[43]:


X_test


# In[44]:


y_test


# In[45]:


y_test_pred = reg.predict(X_test)


# In[46]:


test_data_comparision = pd.DataFrame({'X_test':X_test.flatten(), 'y_test_Actual': y_test , 'y_test_predictions': y_test_pred})


# In[47]:


test_data_comparision


# In[48]:


## Accuracy for the test data 


# In[49]:


r2_score(y_test , y_test_pred)


# In[50]:


plt.title('Simple Linear Regression')
plt.xlabel('X_test')
plt.ylabel('y_test')
plt.scatter(X_test, y_test , color = 'r' , marker='*') # actual 
plt.plot(X_test , y_test_pred,color = 'b' , marker = 'o')  # pred
plt.show()


# In[51]:


## checking with real data 


# In[52]:


reg.predict([[10.5]])


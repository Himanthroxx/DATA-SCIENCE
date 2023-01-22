#!/usr/bin/env python
# coding: utf-8

# ## Naive Bayes with cleaned data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn


# In[2]:


df = pd.read_csv('Iris.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.Species.unique()


# In[6]:


df['Species'] = df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
df.head(10)


# In[7]:


df.tail()


# In[8]:


df.Species.unique()


# In[9]:


df = df.drop(['Id'], axis = 1)
df.head()


# In[10]:


# data visualization


# In[11]:


sb.distplot(df['SepalLengthCm'],color = 'r')
plt.show()


# In[12]:


sb.pairplot(df)


# In[13]:


X = df.iloc[: , :-1]
Y = df.iloc[: , -1]


# In[14]:


X.head()


# In[15]:


Y.head()


# In[16]:


from sklearn.model_selection import train_test_split as tts


# In[17]:


X_train,X_test,Y_train,Y_test = tts(X,Y,test_size = 0.33,random_state = 42)
X_train.shape,Y_train.shape,X_test.shape,Y_test.shape


# In[18]:


from sklearn.naive_bayes import MultinomialNB


# In[19]:


reg = MultinomialNB()


# In[20]:


reg.fit(X_train,Y_train)


# In[21]:


Y_train_pred = reg.predict(X_train)


# In[22]:


Y_train_pred


# In[23]:


dataframe_trained = pd.DataFrame(data = {'Y_train': Y_train,'Y_train_pred': Y_train_pred})
dataframe_trained


# In[24]:


from sklearn.metrics import accuracy_score ,classification_report, confusion_matrix


# In[25]:


accuracy_score(Y_train,Y_train_pred)


# In[26]:


confusion_matrix(Y_train,Y_train_pred)


# In[27]:


print(classification_report(Y_train,Y_train_pred))


# In[28]:


Y_test_pred = reg.predict(X_test)
Y_test_pred


# In[29]:


dataframe_test = pd.DataFrame(data = {'Y_test':Y_test,'Y_test_pred':Y_test_pred})
dataframe_test


# In[30]:


accuracy_score(Y_test,Y_test_pred)


# In[31]:


confusion_matrix(Y_test,Y_test_pred)


# In[32]:


print(classification_report(Y_test,Y_test_pred))


# In[33]:


import warnings
warnings.filterwarnings('ignore')


# In[34]:


##checking with real data


# In[35]:


sol = reg.predict([[6.3,2.5,5.0,1.9]])
sol


# In[36]:


if sol[0] == 0:
    print('Iris-setosa')
elif sol[0] == 1:
    print('Iris-versicolor')
elif sol[0] == 2:
    print('Iris-virginica')


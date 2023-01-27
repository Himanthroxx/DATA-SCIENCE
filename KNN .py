#!/usr/bin/env python
# coding: utf-8

# # KNN with cleaned data

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('breast-cancer.csv')


# In[4]:


df.head(10)


# In[5]:


df.shape 


# In[6]:


# check for null values


# In[7]:


df.isnull().sum()


# In[8]:


df['Unnamed: 32']


# In[9]:


# droping unwanted columns 


# In[10]:


df.columns


# In[11]:


df = df.drop(['id','Unnamed: 32'] , axis = 1)


# In[12]:


df.columns


# In[13]:


df.head()


# In[14]:


df['diagnosis'].unique()


# In[15]:


# m = malignine
# b = benin 


# In[16]:


df['diagnosis'].value_counts()


# In[17]:


# converting dep varia into num 


# In[18]:


df['diagnosis'] = df['diagnosis'].map({'M':0 , 'B':1}).astype(int)


# In[19]:


df['diagnosis'].value_counts()


# In[20]:


df.head()


# ### split the data into training and test 

# In[21]:


X = df.iloc[: , 1:]
y = df.iloc[: , 0]


# In[22]:


X.head()


# In[23]:


y.head()


# In[24]:


import sklearn 


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[27]:


len(X_train) , len(y_train)


# In[28]:


len(X_test) , len(y_test)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier


# In[30]:


reg = KNeighborsClassifier()  # default k value 5 


# In[31]:


reg.fit(X_train , y_train)


# ## checkig with training accuracy 

# In[32]:


y_train_pred = reg.predict(X_train)


# In[33]:


data_plot = pd.DataFrame({'Actual_points':y_train , 'Predicted_points':y_train_pred})


# In[34]:


data_plot


# In[35]:


## confusion matrix for training data


# In[36]:


from sklearn.metrics import confusion_matrix , accuracy_score , classification_report


# In[37]:


confusion_matrix(y_train , y_train_pred)


# In[38]:


len(X_train)


# In[39]:


#  tp + tn  / tp + tn + fp + fn 

(229 + 122) / (122 + 7 + 23 + 229)


# In[40]:


accuracy_score(y_train , y_train_pred)


# In[41]:


print(classification_report(y_train , y_train_pred))


# In[42]:


145 + 236


# In[43]:


y_train.head()


# ### Test_data 

# In[44]:


y_test_pred = reg.predict(X_test)


# In[45]:


confusion_matrix(y_test , y_test_pred)


# In[46]:


accuracy_score(y_test , y_test_pred)


# In[47]:


print(classification_report(y_test , y_test_pred))


# # Finding the best k_value

# In[48]:


k = np.arange(1,101,2)
print(k)


# In[49]:


len(k) 


# In[50]:


training_accuracy_result = np.empty(len(k)) # creating empty array of len 50
test_accuracy_result = np.empty(len(k))     # creating empty array of len 50

for i , j in enumerate(k):
    knn = KNeighborsClassifier(n_neighbors=j)  # KNN classifier with different K values
    knn.fit(X_train , y_train)
    # finding the accuracy 
    training_accuracy_result[i] = knn.score(X_train,y_train)  # training accuracy 
    test_accuracy_result[i] = knn.score(X_test,y_test)        # test accuarcy
    # result on every k-value Accuracy 
    print(training_accuracy_result[i] , test_accuracy_result[i])


# In[51]:


## Visualizing


plt.figure(figsize=(7,5))
plt.subplot(1,2,1)
plt.title('training_accuracy_results')
plt.xlabel('k_values')
plt.ylabel('train_accuracy_values')
plt.plot(k , training_accuracy_result , color = 'r')

plt.subplot(1,2,2)
plt.title('test_accuracy_results')
plt.xlabel('k_values')
plt.ylabel('test_accuracy_values')
plt.plot(k , test_accuracy_result ,color = 'g')

plt.show()


# In[52]:


best_k_value = np.where(test_accuracy_result == max(test_accuracy_result)) 


# In[53]:


best_k_value # consists of index position of best k value in the array of k


# In[54]:


k[best_k_value] # so this will be the best_k_value


# In[55]:


## lest find complete metrics with best k value = 11


# In[56]:


knn1 = KNeighborsClassifier(n_neighbors=11) 
knn1.fit(X_train , y_train)


# In[57]:


# Training accuracy


# In[58]:


y_train_pred_1 = knn1.predict(X_train)
accuracy_score(y_train,y_train_pred_1)


# In[59]:


confusion_matrix(y_train,y_train_pred_1)


# In[60]:


print(classification_report(y_train,y_train_pred_1))


# In[61]:


# checking for test data


# In[62]:


y_test_pred_1 = knn1.predict(X_test)
accuracy_score(y_test,y_test_pred_1)


# In[63]:


confusion_matrix(y_test,y_test_pred_1)


# In[64]:


print(classification_report(y_test,y_test_pred_1))


# In[65]:


# Checking for own data


# In[66]:


sol = knn1.predict([[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3]])
sol = sol[0]
if sol == 1:
    print('Patient having Benin Cancer')
else:
    print('Patient having malignine Cancer')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('mnist.csv')


# In[5]:


data.head()


# In[7]:


a = data.iloc[3,1:].values


# In[8]:


a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[9]:


x= data.iloc[:,1:]
y= data.iloc[:,0]


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=4)


# In[11]:


y_train.head()


# In[15]:


rf = RandomForestClassifier(n_estimators=100)


# In[16]:


rf.fit(x_train, y_train)


# In[17]:


predictions = rf.predict(x_test)


# In[18]:


predictions


# In[19]:


s= y_test.values

count=0
for i in range(len(predictions)):
    if predictions[i] == s[i]:
        count = count+1


# In[20]:


count


# In[21]:


len(predictions)


# In[22]:


accuracy=count/len(predictions)


# In[23]:


accuracy


# #we HAVE 96% ACCURACY IN OUR MODEL

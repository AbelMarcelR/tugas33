#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plp
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


# In[5]:


df = pd.read_csv('heart.csv')
df


# In[6]:


df.isnull().sum()


# In[7]:


df.dtypes


# In[8]:


df.describe()


# In[9]:


plp.figure(figsize= (17,6))
sns.heatmap(df.corr(),annot = True)


# In[10]:


df['sex'].value_counts()


# In[11]:


df['output'].value_counts()


# In[12]:


sns.countplot(x = 'sex', data = df)
plp.xticks([0,1],['female','male'])
plp.show()


# In[13]:


sns.countplot(x='sex',hue="output", data = df)
plp.xticks([1,0],['male','female'])
plp.legend(labels=['No-disease','Disease'])
plp.show


# In[14]:


x= df.drop("output", axis=1)
y= df["output"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state =1)


# In[15]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

model = LogisticRegression(max_iter=300)

# Membuat instance dari KFold untuk cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Menggunakan cross_val_score untuk melakukan cross-validation
scores = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')

# Menampilkan hasil cross-validation
print("Hasil Cross-Validation:")
print(scores)


# In[17]:


pk.dump(model, open("heartattack123.pkl", "wb"))


# In[ ]:





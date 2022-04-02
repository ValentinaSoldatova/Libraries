#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
    


# In[2]:


import numpy as np


# In[3]:


from sklearn.datasets import load_boston


# In[4]:


boston = load_boston()
data = boston["data"]


# In[5]:


feature_names = boston["feature_names"]

X = pd.DataFrame(data, columns=feature_names)
X.head()


# In[6]:


target = boston["target"]

Y = pd.DataFrame(target, columns=["price"])
Y.head()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


lr = LinearRegression()


# In[11]:


lr.fit(X_train, Y_train)


# In[12]:


y_pred_lr = lr.predict(X_test)
check_test_lr = pd.DataFrame({
    "Y_test": Y_test["price"], 
    "Y_pred_lr": y_pred_lr.flatten()})

check_test_lr.head()


# In[13]:


from sklearn.metrics import mean_squared_error

mean_squared_error_lr = mean_squared_error(check_test_lr["Y_pred_lr"], check_test_lr["Y_test"])
print(mean_squared_error_lr)


# In[14]:


#Задание 2#



from sklearn.ensemble import RandomForestRegressor


# In[15]:


clf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)


# In[16]:


clf.fit(X_train, Y_train.values[:, 0])


# In[17]:


y_pred_clf = clf.predict(X_test)
check_test_clf = pd.DataFrame({
    "Y_test": Y_test["price"], 
    "Y_pred_clf": y_pred_clf.flatten()})

check_test_clf.head()


# In[18]:


mean_squared_error_clf = mean_squared_error(check_test_clf["Y_pred_clf"], check_test_clf["Y_test"])
print(mean_squared_error_clf)


# In[19]:


print(mean_squared_error_lr, mean_squared_error_clf)


# In[21]:


#Алгоритм "Случайный лес" показывает точнее в 2,3 раза#


# Задание 3#


# In[22]:


print(clf.feature_importances_)


# In[23]:


feature_importance = pd.DataFrame({'name':X.columns, 
                                   'feature_importance':clf.feature_importances_}, 
                                  columns=['feature_importance', 'name'])
feature_importance


# In[24]:


feature_importance.nlargest(2, 'feature_importance')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a= np.array([[1, 6],
           [2, 8],
           [3, 11],
           [3, 10],
           [1, 7]])


# In[3]:


a


# In[4]:


a.shape


# In[5]:


mean_a = np.mean(a, axis=0)


# In[6]:


mean_a


# In[7]:


a_centered= a- mean_a


# In[8]:


a_centered


# In[9]:


a_centered_sp = a_centered.T[0] @ a_centered.T[1]


# In[10]:


a_centered_sp / (a_centered.shape[0] - 1)


# In[11]:


import pandas as pd


# In[12]:


authors = pd.DataFrame({'author_id':[1, 2, 3], 
                        'author_name':['Тургенев', 'Чехов', 'Островский']}, 
                       columns=['author_id', 'author_name'])


# In[13]:


authors


# In[14]:


book = pd.DataFrame({'author_id':[1, 1, 1, 2, 2, 3, 3], 
                     'book_title':['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'], 
                     'price':[450, 300, 350, 500, 450, 370, 290]}, 
                    columns=['author_id', 'book_title', 'price'])


# In[15]:


book


# In[16]:


authors_price = pd.merge(authors, book, on = 'author_id', how = 'outer')


# In[17]:


authors_price


# In[18]:


top5 = authors_price.nlargest(5, 'price')


# In[19]:


top5


# In[20]:


authors_stat = authors_price.groupby('author_name').agg({'price':['min', 'max', 'mean']})
authors_stat = authors_stat.rename(columns={'min':'min_price', 'max':'max_price', 'mean':'mean_price'})


# In[21]:


authors_stat


# In[ ]:





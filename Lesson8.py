#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Импортируйте библиотеки pandas, numpy и matplotlib.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn.
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()


# In[5]:


#Создайте датафреймы X и y из этих данных.
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
X.info()


# In[6]:


#Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) 
#с помощью функции train_test_split так, чтобы размер тестовой выборки составлял 20% от 
#всех данных, при этом аргумент random_state должен быть равен 42.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[8]:


#Масштабируйте данные с помощью StandardScaler.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# In[9]:


#Постройте модель TSNE на тренировочный данных с параметрами: 
#n_components=2, learning_rate=250, random_state=42.
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
print('До:\t{}'.format(X_train_scaled.shape))
print('После:\t{}'.format(X_train_tsne.shape))


# In[10]:


#Постройте диаграмму рассеяния на этих данных.
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
plt.show()


# In[12]:


#2. С помощью KMeans разбейте данные из тренировочного набора на 3 кластера, 
#используйте все признаки из датафрейма X_train.
#Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
labels_train = kmeans.fit_predict(X_train_scaled)


# In[17]:


#Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE, и раскрасьте точки 
#из разных кластеров разными цветами.
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)
plt.show()


# In[16]:


#Вычислите средние значения price и CRIM в разных кластерах.
print('Средние значения price:')
print('Кластер 0: {}'.format(y_train[labels_train == 0].mean()))
print('Кластер 1: {}'.format(y_train[labels_train == 1].mean()))
print('Кластер 2: {}'.format(y_train[labels_train == 2].mean()))
print('Средние значения CRIM:')
print('Кластер 0: {}'.format(X_train.loc[labels_train == 0, 'CRIM'].mean()))
print('Кластер 1: {}'.format(X_train.loc[labels_train == 1, 'CRIM'].mean()))
print('Кластер 2: {}'.format(X_train.loc[labels_train == 2, 'CRIM'].mean()))


# 

# In[ ]:





# In[ ]:





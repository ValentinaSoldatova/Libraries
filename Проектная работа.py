#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree

from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


matplotlib.rcParams.update({'font.size': 14})


# In[7]:


def evaluate_preds(train_true_values, train_pred_values, test_true_values, test_pred_values):
    print("Train R2:\t" + str(round(r2(train_true_values, train_pred_values), 3)))
    print("Test R2:\t" + str(round(r2(test_true_values, test_pred_values), 3)))
    
    plt.figure(figsize=(18,10))
    
    plt.subplot(121)
    sns.scatterplot(x=train_pred_values, y=train_true_values)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Train sample prediction')
    
    plt.subplot(122)
    sns.scatterplot(x=test_pred_values, y=test_true_values)
    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('Test sample prediction')

    plt.show()


# In[11]:


os.getcwd()


# In[ ]:





# In[12]:


os.chdir('/Users/User/Downloads/real-estate-price-prediction-moscow/')


# In[13]:


TRAIN_DATASET_PATH = '/Users/User/Downloads/real-estate-price-prediction-moscow/train.csv'
TEST_DATASET_PATH = '/Users/User/Downloads/real-estate-price-prediction-moscow/test.csv'


# In[14]:


train_df = pd.read_csv(TRAIN_DATASET_PATH)
train_df.tail()


# In[15]:


train_df.dtypes


# In[16]:


test_df = pd.read_csv(TEST_DATASET_PATH)
test_df.tail()


# In[17]:


print('Строк в трейне:', train_df.shape[0])
print('Строк в тесте', test_df.shape[0])


# In[18]:


train_df.shape[1] - 1 == test_df.shape[1]


# In[19]:


train_df.dtypes


# In[20]:


train_df['Id'] = train_df['Id'].astype(str)
train_df['DistrictId'] = train_df['DistrictId'].astype(str)


# In[21]:


plt.figure(figsize = (16, 8))

train_df['Price'].hist(bins=30)
plt.ylabel('Count')
plt.xlabel('Price')

plt.title('Target distribution')
plt.show()


# In[22]:


train_df.describe()


# In[23]:


train_df.select_dtypes(include='object').columns.tolist()


# In[24]:


train_df['DistrictId'].value_counts()


# In[25]:


train_df['Ecology_2'].value_counts()


# In[26]:


train_df['Ecology_3'].value_counts()


# In[27]:


train_df['Shops_2'].value_counts()


# In[28]:


train_df['Rooms'].value_counts()


# In[29]:


train_df['Rooms_outlier'] = 0
train_df.loc[(train_df['Rooms'] == 0) | (train_df['Rooms'] >= 6), 'Rooms_outlier'] = 1
train_df.head()


# In[30]:


train_df.loc[train_df['Rooms'] == 0, 'Rooms'] = 1
train_df.loc[train_df['Rooms'] >= 6, 'Rooms'] = train_df['Rooms'].median()


# In[31]:


train_df['Rooms'].value_counts()


# In[32]:


train_df['KitchenSquare'].value_counts()


# In[33]:


train_df['KitchenSquare'].quantile(.975), train_df['KitchenSquare'].quantile(.025)


# In[34]:


condition = (train_df['KitchenSquare'].isna())              | (train_df['KitchenSquare'] > train_df['KitchenSquare'].quantile(.975))
        
train_df.loc[condition, 'KitchenSquare'] = train_df['KitchenSquare'].median()

train_df.loc[train_df['KitchenSquare'] < 3, 'KitchenSquare'] = 3


# In[35]:


train_df['KitchenSquare'].value_counts()


# In[36]:


train_df['Square'].describe()


# In[37]:


train_df['Square_outlier'] = 0
train_df.loc[(train_df['Square'] < 8), 'Square_outlier'] = 1


# In[38]:


train_df.loc[(train_df['Square'] < 8), 'Square'] = train_df['Square'].median()
train_df[train_df['Square_outlier'] == 1]


# In[39]:


train_df['LifeSquare'].describe()


# In[40]:


train_df['LifeSquare_outlier'] = 0
train_df.loc[(train_df['LifeSquare'] < 6), 'LifeSquare_outlier'] = 1


# In[41]:


train_df.loc[(train_df['LifeSquare'] < 6), 'LifeSquare'] = train_df['LifeSquare'].median()
train_df[train_df['LifeSquare_outlier'] == 1]


# In[42]:


train_df['HouseFloor'].sort_values().unique()


# In[43]:


train_df['Floor'].sort_values().unique()


# In[44]:


(train_df['Floor'] > train_df['HouseFloor']).sum()


# In[45]:


train_df['HouseFloor_outlier'] = 0
train_df.loc[train_df['HouseFloor'] == 0, 'HouseFloor_outlier'] = 1
train_df.loc[train_df['Floor'] > train_df['HouseFloor'], 'HouseFloor_outlier'] = 1


# In[46]:


train_df.loc[train_df['HouseFloor'] == 0, 'HouseFloor'] = train_df['HouseFloor'].median()


# In[47]:


floor_outliers = train_df.loc[train_df['Floor'] > train_df['HouseFloor']].index
floor_outliers


# In[48]:


train_df.loc[floor_outliers, 'Floor'] = train_df.loc[floor_outliers, 'HouseFloor']                                                .apply(lambda x: random.randint(1, x))


# In[49]:


(train_df['Floor'] > train_df['HouseFloor']).sum()


# In[50]:


train_df['HouseYear'].sort_values(ascending=False)


# In[51]:


train_df['HouseYear_outlier'] = 0
train_df.loc[train_df['HouseYear'] > 2020, 'HouseYear_outlier'] = 1


# In[52]:


train_df.loc[train_df['HouseYear'] > 2020, 'HouseYear'] = 2020


# In[53]:


train_df.isna().sum()


# In[54]:


train_df[['Square', 'LifeSquare', 'KitchenSquare']].head(10)


# In[55]:


train_df['LifeSquare_nan'] = train_df['LifeSquare'].isna() * 1

condition = (train_df['LifeSquare'].isna())              & (~train_df['Square'].isna())              & (~train_df['KitchenSquare'].isna())
        
train_df.loc[condition, 'LifeSquare'] = train_df.loc[condition, 'Square']                                             - train_df.loc[condition, 'KitchenSquare'] - 3


# In[56]:


train_df['Healthcare_1'].describe()


# In[57]:


train_df['Healthcare_1'].sort_values().unique()


# In[58]:


train_df['Healthcare_1'].value_counts()


# In[59]:


train_df['Healthcare_1_nan'] = train_df['Healthcare_1'].isna() * 1


# In[60]:


grid = sns.jointplot(train_df['Healthcare_1'], train_df['Price'], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[61]:


train_df['Helthcare_2'].sort_values().unique()


# In[62]:


grid = sns.jointplot(train_df['Healthcare_1'], train_df['Helthcare_2'], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[63]:


grid = sns.jointplot(train_df['Healthcare_1'], train_df['HouseYear'], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[64]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.4)

corr_matrix = train_df.corr()
corr_matrix = np.round(corr_matrix, 1)
corr_matrix[np.abs(corr_matrix) < 0.5] = 0

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# In[65]:


med_hc1_by_distr = train_df.groupby(['DistrictId'], as_index=False).agg({'Healthcare_1':'median', 'Price': 'mean', 'HouseYear': 'median','Helthcare_2': 'mean'})
                           
med_hc1_by_distr['Healthcare_1'].sort_values().unique()
med_hc1_by_distr


# In[66]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.4)

corr_matrix = med_hc1_by_distr.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.2] = 0

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# In[67]:


train_df


# In[68]:


med_hc1_by_hc2 = train_df.groupby(['Helthcare_2'], as_index=False).agg({'Healthcare_1':'median', 'Price': 'median', 'HouseYear': 'median'})
                           
#med_hc1_by_hc2['Healthcare_1'].sort_values().unique()
med_hc1_by_hc2


# In[69]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.4)

corr_matrix = med_hc1_by_hc2.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# In[70]:


grid = sns.jointplot(med_hc1_by_hc2['Price'], med_hc1_by_hc2['Healthcare_1'], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[71]:


train_df[train_df['Helthcare_2'] == 1]['Healthcare_1'].describe()


# In[72]:


med_hc1_by_hc2.isna().sum()


# In[73]:


med_hc1_by_hc2.drop('Price', axis=1, inplace=True)
med_hc1_by_hc2.drop('HouseYear', axis=1, inplace=True)
med_hc1_by_hc2


# In[74]:


med_hc1_by_hc2.rename(columns={'Helthcare_2':'Helthcare_2', 'Healthcare_1':'Healthcare_1_by_Helthcare_2'}, inplace = True)
med_hc1_by_hc2


# In[75]:


train_df = train_df.merge(med_hc1_by_hc2, on=['Helthcare_2'], how='left')
train_df


# In[76]:


med_hc1_by_hc2_distr = train_df.groupby(['Helthcare_2', 'DistrictId'], as_index=False).agg({'Healthcare_1':'median', 'Price': 'mean', 'HouseYear': 'median'})
                           
med_hc1_by_hc2_distr['Healthcare_1'].sort_values().unique()
med_hc1_by_hc2_distr


# In[77]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.4)

corr_matrix = med_hc1_by_hc2_distr.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.2] = 0

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# In[78]:


med_hc1_by_year = train_df.groupby(['HouseYear'], as_index=False).agg({'Healthcare_1':'median', 'Price': 'median', 'Helthcare_2': 'median'})
                           
med_hc1_by_year['Healthcare_1'].sort_values().unique()
#med_hc1_by_year


# In[79]:


plt.figure(figsize=(15, 10))

sns.set(font_scale=1.4)

corr_matrix = med_hc1_by_year.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0

sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')

plt.title('Correlation matrix')
plt.show()


# In[81]:


grid = sns.jointplot(med_hc1_by_year['Price'], med_hc1_by_year['Healthcare_1'], kind='reg')
grid.fig.set_figwidth(8)
grid.fig.set_figheight(8)

plt.show()


# In[82]:


med_hc1_by_year.isna().sum()


# In[83]:


med_hc1_by_year.loc[med_hc1_by_year['Healthcare_1'].isna(), 'Healthcare_1'] = med_hc1_by_year['Healthcare_1'].median() 


# In[84]:


med_hc1_by_year.drop('Price', axis=1, inplace=True)
med_hc1_by_year.drop('Helthcare_2', axis=1, inplace=True)
med_hc1_by_year


# In[85]:


med_hc1_by_year.rename(columns={'Healthcare_1':'Healthcare_1_by_HouseYear'}, inplace = True)
med_hc1_by_year


# In[86]:


train_df = train_df.merge(med_hc1_by_year, on=['HouseYear'], how='left')
train_df


# In[87]:


train_df['Healthcare_1'].fillna(train_df['Healthcare_1_by_HouseYear'], inplace=True)


# In[88]:


class DataPreprocessing:
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.medians = None
        self.kitchen_square_quantile = None
        
    def fit(self, X):
        """Сохранение статистик"""       
        # Расчет медиан
        self.medians = X.median()
        self.kitchen_square_quantile = X['KitchenSquare'].quantile(.975)
    
    def transform(self, X):
        """Трансформация данных"""

        # Rooms
        X['Rooms_outlier'] = 0
        X.loc[(X['Rooms'] == 0) | (X['Rooms'] >= 6), 'Rooms_outlier'] = 1
        
        X.loc[X['Rooms'] == 0, 'Rooms'] = 1
        X.loc[X['Rooms'] >= 6, 'Rooms'] = self.medians['Rooms']
        
        #Square
        X['Square_outlier'] = 0
        X.loc[(X['Square'] < 8), 'Square_outlier'] = 1
        X.loc[X['Square'] < 8, 'Square'] = self.medians['Square']
        
        #LifeSquare
        X['LifeSquare_outlier'] = 0
        X.loc[(X['LifeSquare'] < 6), 'LifeSquare_outlier'] = 1
        X.loc[X['LifeSquare'] < 6, 'LifeSquare'] = self.medians['LifeSquare']
        
        # KitchenSquare
        condition = (X['KitchenSquare'].isna())                     | (X['KitchenSquare'] > self.kitchen_square_quantile)
        
        X.loc[condition, 'KitchenSquare'] = self.medians['KitchenSquare']

        X.loc[X['KitchenSquare'] < 3, 'KitchenSquare'] = 3
        
        # HouseFloor, Floor
        X['HouseFloor_outlier'] = 0
        X.loc[X['HouseFloor'] == 0, 'HouseFloor_outlier'] = 1
        X.loc[X['Floor'] > X['HouseFloor'], 'HouseFloor_outlier'] = 1
        
        X.loc[X['HouseFloor'] == 0, 'HouseFloor'] = self.medians['HouseFloor']
        
        floor_outliers = X.loc[X['Floor'] > X['HouseFloor']].index
        X.loc[floor_outliers, 'Floor'] = X.loc[floor_outliers, 'HouseFloor']                                            .apply(lambda x: random.randint(1, x))
        
        # HouseYear
        current_year = datetime.now().year
        
        X['HouseYear_outlier'] = 0
        X.loc[X['HouseYear'] > current_year, 'HouseYear_outlier'] = 1
        
        X.loc[X['HouseYear'] > current_year, 'HouseYear'] = current_year
        
        # Healthcare_1
        X['Healthcare_1_nan'] = X['Healthcare_1'].isna() * 1
        
        med_hc1_by_hc2 = X.groupby(['Helthcare_2'], as_index=False).agg({'Healthcare_1':'median'})                                 .rename(columns={'Healthcare_1': 'Healthcare_1_by_Helthcare_2'})
        X = X.merge(med_hc1_by_hc2, on=['Helthcare_2'], how='left')
        X['Healthcare_1_by_Helthcare_2'].fillna(X['Healthcare_1_by_Helthcare_2'].median(), inplace=True)
    
        med_hc1_by_year = X.groupby(['HouseYear'], as_index=False).agg({'Healthcare_1':'median'})                                 .rename(columns={'Healthcare_1': 'Healthcare_1_by_HouseYear'}) 
        X = X.merge(med_hc1_by_year, on=['HouseYear'], how='left')
        X['Healthcare_1_by_HouseYear'].fillna(X['Healthcare_1_by_HouseYear'].median(), inplace=True)
            
        # LifeSquare
        X['LifeSquare_nan'] = X['LifeSquare'].isna() * 1
        condition = (X['LifeSquare'].isna()) &                       (~X['Square'].isna()) &                       (~X['KitchenSquare'].isna())
        
        X.loc[condition, 'LifeSquare'] = X.loc[condition, 'Square'] - X.loc[condition, 'KitchenSquare'] - 3
        
        
        X.fillna(self.medians, inplace=True)
        
        return X


# In[89]:


binary_to_numbers = {'A': 0, 'B': 1}

train_df['Ecology_2'] = train_df['Ecology_2'].replace(binary_to_numbers)
train_df['Ecology_3'] = train_df['Ecology_3'].replace(binary_to_numbers)
train_df['Shops_2'] = train_df['Shops_2'].replace(binary_to_numbers)


# In[90]:


district_size = train_df['DistrictId'].value_counts().reset_index()                    .rename(columns={'index':'DistrictId', 'DistrictId':'DistrictSize'})

district_size.head()


# In[91]:


train_df = train_df.merge(district_size, on='DistrictId', how='left')
train_df.head()


# In[92]:


(train_df['DistrictSize'] > 100).value_counts()


# In[93]:


train_df['IsDistrictLarge'] = (train_df['DistrictSize'] > 100).astype(int)


# In[94]:


train_df['LifeSquare_to_Square'] = train_df['LifeSquare'] / train_df['Square']
train_df['LifeSquare_to_Square'].describe()


# In[95]:


med_price_by_district = train_df.groupby(['DistrictId', 'Rooms'], as_index=False).agg({'Price':'median'})                            .rename(columns={'Price':'MedPriceByDistrict'})

med_price_by_district.head()


# In[96]:


med_price_by_district.shape


# In[97]:


train_df = train_df.merge(med_price_by_district, on=['DistrictId', 'Rooms'], how='left')
train_df.head()


# In[98]:


def floor_to_cat(X):

    X['floor_cat'] = 0

    X.loc[X['Floor'] <= 3, 'floor_cat'] = 1  
    X.loc[(X['Floor'] > 3) & (X['Floor'] <= 5), 'floor_cat'] = 2
    X.loc[(X['Floor'] > 5) & (X['Floor'] <= 9), 'floor_cat'] = 3
    X.loc[(X['Floor'] > 9) & (X['Floor'] <= 15), 'floor_cat'] = 4
    X.loc[X['Floor'] > 15, 'floor_cat'] = 5

    return X


def floor_to_cat_pandas(X):
    bins = [X['Floor'].min(), 3, 5, 9, 15, X['Floor'].max()]
    X['floor_cat'] = pd.cut(X['Floor'], bins=bins, labels=False)
    
    X['floor_cat'].fillna(-1, inplace=True)
    return X


def year_to_cat(X):

    X['year_cat'] = 0

    X.loc[X['HouseYear'] <= 1941, 'year_cat'] = 1
    X.loc[(X['HouseYear'] > 1941) & (X['HouseYear'] <= 1945), 'year_cat'] = 2
    X.loc[(X['HouseYear'] > 1945) & (X['HouseYear'] <= 1980), 'year_cat'] = 3
    X.loc[(X['HouseYear'] > 1980) & (X['HouseYear'] <= 2000), 'year_cat'] = 4
    X.loc[(X['HouseYear'] > 2000) & (X['HouseYear'] <= 2010), 'year_cat'] = 5
    X.loc[(X['HouseYear'] > 2010), 'year_cat'] = 6

    return X


def year_to_cat_pandas(X):
    bins = [X['HouseYear'].min(), 1941, 1945, 1980, 2000, 2010, X['HouseYear'].max()]
    X['year_cat'] = pd.cut(X['HouseYear'], bins=bins, labels=False)
    
    X['year_cat'].fillna(-1, inplace=True)
    return X


# In[99]:


bins = [train_df['Floor'].min(), 3, 5, 9, 15, train_df['Floor'].max()]
pd.cut(train_df['Floor'], bins=bins, labels=False)


# In[100]:


train_df = year_to_cat(train_df)
train_df = floor_to_cat(train_df)
train_df.head()


# In[101]:


med_price_by_floor_year = train_df.groupby(['year_cat', 'floor_cat'], as_index=False).agg({'Price':'median'}).                                            rename(columns={'Price':'MedPriceByFloorYear'})
med_price_by_floor_year.head()


# In[102]:


train_df = train_df.merge(med_price_by_floor_year, on=['year_cat', 'floor_cat'], how='left')
train_df.head()


# In[103]:


class FeatureGenetator():
    """Генерация новых фич"""
    
    def __init__(self):
        self.DistrictId_counts = None
        self.binary_to_numbers = None
        self.med_price_by_district = None
        self.med_price_by_floor_year = None
        self.house_year_max = None
        self.floor_max = None
        self.house_year_min = None
        self.floor_min = None
        self.district_size = None
        
    def fit(self, X, y=None):
        
        X = X.copy()
        
        # Binary features
        self.binary_to_numbers = {'A': 0, 'B': 1}
        
        # DistrictID
        self.district_size = X['DistrictId'].value_counts().reset_index()                                .rename(columns={'index':'DistrictId', 'DistrictId':'DistrictSize'})
                
        # Target encoding
        ## District, Rooms
        df = X.copy()
        
        if y is not None:
            df['Price'] = y.values
            
            self.med_price_by_district = df.groupby(['DistrictId', 'Rooms'], as_index=False).agg({'Price':'median'})                                            .rename(columns={'Price':'MedPriceByDistrict'})
            
            self.med_price_by_district_median = self.med_price_by_district['MedPriceByDistrict'].median()
            
        ## floor, year
        if y is not None:
            self.floor_max = df['Floor'].max()
            self.floor_min = df['Floor'].min()
            self.house_year_max = df['HouseYear'].max()
            self.house_year_min = df['HouseYear'].min()
            df['Price'] = y.values
            df = self.floor_to_cat(df)
            df = self.year_to_cat(df)
            self.med_price_by_floor_year = df.groupby(['year_cat', 'floor_cat'], as_index=False).agg({'Price':'median'}).                                            rename(columns={'Price':'MedPriceByFloorYear'})
            self.med_price_by_floor_year_median = self.med_price_by_floor_year['MedPriceByFloorYear'].median()
        

        
    def transform(self, X):
        
        # Binary features
        X['Ecology_2'] = X['Ecology_2'].map(self.binary_to_numbers)  # self.binary_to_numbers = {'A': 0, 'B': 1}
        X['Ecology_3'] = X['Ecology_3'].map(self.binary_to_numbers)
        X['Shops_2'] = X['Shops_2'].map(self.binary_to_numbers)
        
        # DistrictId, IsDistrictLarge
        X = X.merge(self.district_size, on='DistrictId', how='left')
        
        X['new_district'] = 0
        X.loc[X['DistrictSize'].isna(), 'new_district'] = 1
        
        X['DistrictSize'].fillna(5, inplace=True)
        
        X['IsDistrictLarge'] = (X['DistrictSize'] > 100).astype(int)
        
        # More categorical features
        X = self.floor_to_cat(X)  # + столбец floor_cat
        X = self.year_to_cat(X)   # + столбец year_cat
        
        # Доля жилой площади от общей
        X['LifeSquare_to_Square'] = X['LifeSquare'] / X['Square']
        
        # Target encoding
        if self.med_price_by_district is not None:
            X = X.merge(self.med_price_by_district, on=['DistrictId', 'Rooms'], how='left')
            X['MedPriceByDistrict'].fillna(self.med_price_by_district_median, inplace=True)
            
        if self.med_price_by_floor_year is not None:
            X = X.merge(self.med_price_by_floor_year, on=['year_cat', 'floor_cat'], how='left')
            X['MedPriceByFloorYear'].fillna(self.med_price_by_floor_year_median, inplace=True)
        
        return X
    
    def floor_to_cat(self, X):
        bins = [self.floor_min, 3, 5, 9, 15, self.floor_max]
        X['floor_cat'] = pd.cut(X['Floor'], bins=bins, labels=False)

        X['floor_cat'].fillna(-1, inplace=True)
        return X
     
    def year_to_cat(self, X):
        bins = [self.house_year_min, 1941, 1945, 1980, 2000, 2010, self.house_year_max]
        X['year_cat'] = pd.cut(X['HouseYear'], bins=bins, labels=False)

        X['year_cat'].fillna(-1, inplace=True)
        return X


# In[104]:


train_df.isna().sum()


# In[105]:


train_df.columns.tolist()


# In[106]:


feature_names = ['Rooms', 'Square', 'LifeSquare', 
                 'KitchenSquare', 'Floor', 
                 'HouseFloor', 'HouseYear',
                 'Ecology_1', 'Ecology_2', 'Ecology_3', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1',
                 'Helthcare_2', 'Shops_1', 'Shops_2']

new_feature_names = ['Rooms_outlier', 'Square_outlier', 'LifeSquare_outlier', 'HouseFloor_outlier', 'HouseYear_outlier', 'LifeSquare_nan', 'Healthcare_1_nan', 
                     'Healthcare_1_by_Helthcare_2', 
                     'Healthcare_1_by_HouseYear', 'DistrictSize', 'LifeSquare_to_Square',
                     'new_district', 'IsDistrictLarge',  # 'MedPriceByDistrict', 
                     'MedPriceByFloorYear']

target_name = 'Price'


# In[107]:


train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_df = pd.read_csv(TEST_DATASET_PATH)

X = train_df.drop(columns=target_name)
y = train_df[target_name]


# In[108]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=21)


# In[109]:


preprocessor = DataPreprocessing()
preprocessor.fit(X_train)

X_train = preprocessor.transform(X_train)
X_valid = preprocessor.transform(X_valid)
test_df = preprocessor.transform(test_df)

X_train.shape, X_valid.shape, test_df.shape


# In[110]:


features_gen = FeatureGenetator()
features_gen.fit(X_train, y_train)

X_train = features_gen.transform(X_train)
X_valid = features_gen.transform(X_valid)
test_df = features_gen.transform(test_df)

X_train.shape, X_valid.shape, test_df.shape


# In[111]:


X_train = X_train[feature_names + new_feature_names]
X_valid = X_valid[feature_names + new_feature_names]
test_df = test_df[feature_names + new_feature_names]


# In[112]:


X_train.isna().sum().sum(), X_valid.isna().sum().sum(), test_df.isna().sum().sum()


# In[113]:


rf_model = RandomForestRegressor(criterion='mse',
                                 max_depth=20, # глубина дерева  
                                 min_samples_leaf=2, # минимальное кол-во наблюдений в листе дерева
                                 random_state=42, 
                                 n_estimators=400  # кол-во деревьев
                                 )

rf_model.fit(X_train, y_train)

y_train_preds = rf_model.predict(X_train)
y_test_preds = rf_model.predict(X_valid)

evaluate_preds(y_train, y_train_preds, y_valid, y_test_preds)


# In[114]:


gb_model = GradientBoostingRegressor(criterion='mse',
                                     max_depth=5,
                                     min_samples_leaf=4,
                                     random_state=42,  
                                     n_estimators=200)
gb_model.fit(X_train, y_train)

y_train_preds = gb_model.predict(X_train)
y_test_preds = gb_model.predict(X_valid)

evaluate_preds(y_train, y_train_preds, y_valid, y_test_preds)


# In[115]:


feature_importances = pd.DataFrame(zip(X_train.columns, rf_model.feature_importances_), 
                                   columns=['feature_name', 'importance'])

feature_importances.sort_values(by='importance', ascending=False)


# In[116]:


test_df.shape


# In[117]:


test_df


# In[119]:


submit = pd.read_csv('/Users/User/Downloads/real-estate-price-prediction-moscow/sample_submission.csv')
submit.head()


# In[120]:


predictions = gb_model.predict(test_df)
predictions


# In[121]:


submit['Price'] = predictions
submit.head()


# In[122]:


submit.to_csv('gb_submit.csv', index=False)


# In[ ]:





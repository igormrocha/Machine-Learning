#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # Project

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv('bengaluru_house_prices.csv')
df


# # Cleaning the dataset 

# __Removing unimportant features__

# In[5]:


df1 = df.drop(['society','area_type','availability'],1)


# __Removing NaN values__

# In[6]:


df1.isnull().sum()
df1['balcony'] = df1.balcony.fillna(0)
df1.isnull().sum()
df2 = df1.dropna()


# __Organizing bedroom column__

# In[7]:


df2['size'] = df2['size'].apply(lambda x: x.split(' ')[0])
df2.rename(columns={'size':'bedrooms'}, inplace = True)


# __Organizing the squared feet column__

# In[8]:


df2.total_sqft.unique()


# In[9]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[10]:


df2.total_sqft = df2.total_sqft.apply(convert_sqft_to_num)
df2 = df2[df2.total_sqft.notnull()]


# __Converting data types__

# In[11]:


df2['bedrooms'] = pd.to_numeric(df2['bedrooms'])
df2['price'] = df2['price'] * 100000  # Putting the right scaling in the price column


# __Reducing the number of location categories__

# In[12]:


location_stats = df2.groupby('location')['location'].agg('count')
location_stats.sort_values(ascending=False)
location_stats_lt_10 = location_stats[location_stats<=10]
df2['location'] = df2['location'].apply(lambda x: 'other' if x in location_stats_lt_10 else x)


# __Removing outliers__

# In[13]:


df2['price/sqft'] = df2['price']/df2['total_sqft']
df2['sqft/bedroom'] = df2['total_sqft']/df2['bedrooms']
df2['price/sqft.bedroom'] = df2['price']/(df2['total_sqft']*df2['bedrooms'])


# In[14]:


df2


# In[15]:


stats = df2['sqft/bedroom'].describe()
m = stats[1]
st = stats[2]
df2 = df2[(df2['sqft/bedroom'] <= m+st) & (df2['sqft/bedroom'] > m-st)]
df2


# In[16]:


def remove_outliers(df, group, feature):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby(group):
        m = np.mean(subdf[feature])
        st = np.std(subdf[feature])
        reduced_df = subdf[(subdf[feature]>(m-st)) & (subdf[feature]<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[17]:


df3 = remove_outliers(df2, 'location', 'price/sqft')
df3 = remove_outliers(df3, 'location','price/sqft.bedroom')

df3


# In[18]:


df4 = df3[df3.bedrooms + 2 >= df3.bath]
df_model = df4.drop(['price/sqft','sqft/bedroom','price/sqft.bedroom'],1)
df_model


# # __Model Building__

# In[68]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Getting dummies

dummies = pd.get_dummies(df_model['location'])
dummies.drop(['other'],1,inplace = True)
dummies
df_model1 = pd.concat([df_model,dummies],1)
df_model1.drop(['location'],1,inplace = True)    
df = df_model1

X = df.drop(['price'],1)
y = df['price']

# Grid Searching

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'Lasso': Lasso()    
}

params = {
    'LinearRegression': 
    {
        'normalize':[True,False]        
    },
    'DecisionTreeRegressor':
    {
        'criterion':['mse','friedman_mse'],
        'splitter':['best','random']    
    },
    'Lasso':
    {
        'alpha':[1,2],
        'selection':['random','cyclic']
    }
}

scores = []

for name in models.keys():
    est = models[name]
    est_params = params[name]
    rscv = RandomizedSearchCV(est,est_params, cv=cv ,n_jobs = -1)
    rscv.fit(X,y)
    scores.append({
        'Model': name,
        'Best params': rscv.best_params_,
        'Best score': rscv.best_score_
    })
    print(name, 'OK')

results = pd.DataFrame(scores, columns = ['Model', 'Best params', 'Best score'])
print(results)

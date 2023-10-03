#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import requests
from io import StringIO


# In[5]:


# Define the URL of the CSV file
csv_url = 'https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data'

# Download the CSV data
response = requests.get(csv_url)
df = pd.read_csv(StringIO(response.text))
df.head()


# In[6]:


#Meta data
df.info()


# In[9]:


#No. of records
df.shape


# In[14]:


#group the data and count the number of occurences in each group
grouped_data = df.groupby(['famhist', 'chd']).size().reset_index(name='count')
grouped_data


# In[16]:


sn.barplot(x = 'chd', y = 'count', hue = 'famhist', data = grouped_data)


# In[17]:


influential_features = ['age', 'sbp']
df[influential_features].corr()


# In[19]:


sn.heatmap(df[influential_features].corr(), annot=True)


# In[26]:


sn.scatterplot(x='age', y='sbp', data=df, alpha=0.5)
sn.regplot(x='age', y='sbp', data=df)
plt.show()


# In[37]:


sn.distplot(df[df['chd']==1]['tobacco'], color='y')
sn.distplot(df[df['chd']==0]['tobacco'], color='r')


# In[38]:


influential_features = ['sbp', 'obesity', 'age', 'ldl']
sn.pairplot(df[influential_features], size=2)


# In[40]:


def categorize_age(age):
    if age <= 15:
        return 'young'
    elif age <= 35:
        return 'adults'
    elif age <= 55:
        return 'mid'
    else:
        return 'old'
df['agegroup'] = df['age'].apply(categorize_age)
df


# In[43]:


chd_by_age = df.groupby('agegroup')['chd'].mean().reset_index()
sn.barplot(x = 'agegroup', y='chd', data = chd_by_age)


# In[44]:


sn.boxplot(x = 'ldl', y='agegroup', data=df)


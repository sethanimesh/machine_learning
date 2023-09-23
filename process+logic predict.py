#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('ecommerce_data.csv')


# In[3]:


df.head()


# In[4]:


df['time_of_day'].hist()


# In[5]:


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    
    #converting to numpy array
    #df.as_matrix() or df.values
    data = df.to_numpy()
    
    #shuffle it
    np.random.shuffle(data)
    
    #split features and labels
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)
    
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, :(D-1)] = X[:, :(D-1)] #non-categorical columns
    
    #one-hot encoding
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1 

    #method 2
    #Z = np.zeros((N, 4))
    #Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    #Z[(r1, r2, r3, ...), (c1, c2, c3, ...)] = value
    #X2[:, -4:] = z
    
    #assign X2 back to X
    X = X2
    
    #split train and test
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]
    
    #normalise columns 1 and 2
    for i in (1, 2):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, 1].std()
        Xtrain[:, i] = (Xtrain[:, i]-m) / s
        Xtest[:, i] = (Xtest[:, i]-m) / s
    
    return Xtrain, Ytrain, Xtest, Ytest


# In[6]:


Xtrain, Ytrain, Xtest, Ytest = get_data()


# In[7]:


def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain<=1]
    Y2train = Ytrain[Ytrain<=1]
    X2test = Xtest[Ytest<=1]
    Y2test = Ytest[Ytest<=1]
    
    return X2train, Y2train, X2test, Y2test


# In[8]:


X2train, Y2train, X2test, Y2test = get_binary_data()


# In[9]:


X2train.shape


# In[10]:


X, Y, _, _ = get_binary_data()


# In[11]:


# randomly initialise weights

D = X.shape[1] #number of features
W = np.random.randn(D)
b = 0


# In[12]:


# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# In[13]:


def forward(X, W, b):
    return sigmoid(X.dot(W)+b)


# In[14]:


P_Y_given_X = forward(X, W, b)


# In[15]:


P_Y_given_X.shape


# In[16]:


predictions = np.round(P_Y_given_X)
predictions


# In[17]:


def classification_rate(Y, P):
    return np.mean(Y == P)


# In[18]:


print("Score: ", classification_rate(Y, predictions))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 


# In[5]:


nRowsRead = 1000 
df1 = pd.read_csv('C:/Users/DELL/Downloads/estimated_numbers.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'estimated_numbers.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[28]:


print(df1)


# In[7]:


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[8]:


plotPerColumnDistribution(df1, 10, 5)


# In[12]:


def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') 
    df = df[[col for col in df if df[col].nunique() > 1]] 
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[13]:


plotCorrelationMatrix(df1, 8)


# In[15]:


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# In[16]:


plotScatterMatrix(df1, 18, 10)


# In[17]:


df2 = pd.read_csv('C:/Users/DELL/Downloads/incidence_per_1000_pop_at_risk.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'incidence_per_1000_pop_at_risk.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[18]:


print(df2)


# In[19]:


plotPerColumnDistribution(df2, 10, 5)


# In[20]:


plotCorrelationMatrix(df2, 8)


# In[21]:


plotScatterMatrix(df2, 18, 10)


# In[23]:


nRowsRead = 1000 
df3 = pd.read_csv('C:/Users/DELL/Downloads/reported_numbers.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'reported_numbers.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[24]:


print(df3)


# In[25]:


plotPerColumnDistribution(df2, 10, 5)


# In[26]:


plotCorrelationMatrix(df3, 8)


# In[27]:


plotScatterMatrix(df3, 18, 10)


# In[ ]:





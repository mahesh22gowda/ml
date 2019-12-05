#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
#create collection of datapoints

X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])


# In[ ]:


X


# In[3]:


#plot the data points
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0], X[:,1], label='True Position')


# In[6]:


#use the scipy library function to draw the dendrograms considering the data points
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=labelList)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





		#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the following libraries.
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


# In[3]:


#Load the data.
iris = datasets.load_iris()


# In[4]:


#Define the target (Y as target) and predictors (X as sepal length and sepal width).
X = iris.data[:, :2]
y = iris.target


# In[5]:


X


# In[7]:


y


# In[8]:


#visualize data through a scatter plot.
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Spea1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)


# In[9]:


# create k means cluster and fit the model. Consider three clusters and a random state of 21.
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(X)


# In[ ]:





# In[10]:


km


# In[11]:


# display the three center points of the three clusters.
centers = km.cluster_centers_
print(centers)


# In[19]:


#Plot the original clusters 
figure, axes = plt.subplots(1, 2, figsize=(16,8))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
#Plot the identified clusters 
new_labels = km.labels_
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
# give the title for the graph
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('Predicted', fontsize=18)
# give the X,Y axis label to the graph
axes[0].set_xlabel('Sepal length', fontsize=18)
axes[0].set_ylabel('Sepal width', fontsize=18)
axes[1].set_xlabel('Sepal length', fontsize=18)
axes[1].set_ylabel('Sepal width', fontsize=18)


# In[15]:





# In[ ]:





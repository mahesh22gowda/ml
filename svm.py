#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:



import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from sklearn import svm


# In[5]:


import sklearn.metrics as metric


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


filedata='data1'


# In[8]:


data1=pd.read_csv(filedata)


# In[9]:


data1


# In[10]:


#We separate the X training data from the y training data


# In[11]:


X1=data1['X1']


# In[12]:


X2=data1['X2']


# In[13]:


X_training=np.array(list(zip(X1,X2)))


# In[14]:


X_training


# In[15]:


y_training=data1['Y']


# In[16]:


y_training


# In[17]:


target_names=['-1','+1']


# In[18]:


target_names


# In[19]:


#Let us plot this data. Can you imagine a line separating the two classes?


# In[20]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='b',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=2)
plt.xlabel('X1')
plt.ylabel('X2');
plt.savefig('chart0.png')


# In[21]:


#To use Vector Support Classification (SVC) algorithm, we need define the model kernel. Let us use linear kernel. Then, we use the fit() function to train the model with our training data.


# In[22]:


svc = svm.SVC(kernel='linear').fit(X_training,y_training)
svc


# In[23]:


#To view the internal model parameters use get_params() method.


# In[24]:


svc.get_params(True)


# In[25]:


#The trained model can be plotted with specifying the decision_function() method.

#First, we set the boundary of the plot.


# In[26]:


import math


# In[27]:


lbX1=math.floor(min(X_training[:,0]))-1
ubX1=math.ceil(max(X_training[:,0]))+1
lbX2=math.floor(min(X_training[:,1]))-1
ubX2=math.ceil(max(X_training[:,1]))+1
[lbX1,ubX1,lbX2,ubX2]


# In[28]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='b',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=4)

X,Y = np.mgrid[lbX1:ubX1:100j,lbX2:ubX2:100j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=['k'], linestyles=['-'],levels=[0])

plt.title('Linearly Separable')
plt.savefig('chart1.png')


# In[29]:


#The following plot show the margin and the support vectors


# In[30]:


idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c='b',s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c='r',s=50)
plt.legend(target_names,loc=2)
X,Y = np.mgrid[lbX1:ubX1:100j,lbX2:ubX2:100j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)

plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=['k','k','k'], linestyles=['--','-','--'],levels=[-1,0,1])
plt.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],s=120,facecolors='none')
plt.scatter(X_training[:,0],X_training[:,1],c=y_training,s=50,alpha=0.95);

plt.title('Margin and Support Vectors')
plt.savefig('chart2.png')


# In[31]:


#The number of support vectors for each class can be revealed using `nsupport' attribute


# In[32]:


svc.n_support_


# In[33]:


#To get the indices (= the row numbers in the original dataset) of the support vectors, use support_ attribute


# In[34]:


svc.support_ 


# In[35]:


#To identify the support vector, use support_vectors_ attribute. The data that become the support vector aresvc.support_vectors_


# In[36]:


svc.support_vectors_


# In[37]:


#For linear model, we can reveal the discriminant line that separate the classes using coef_ and intercept_ attributes.

weight=svc.coef_
intercept=svc.intercept_
a = -weight[0,0] / weight[0,1]
print('x2=',a,' * x1 + ',-intercept[0]/weight[0,1])


# In[38]:


#Training Performances

#To get the normalize accuracy, of the training, we can use score(X,y) function.

svc.score(X_training, y_training)


# In[39]:


#Alternatively, if you have test sample, you can also use the metric from sklearn. To use this on the training sample, we first need to define the y-prediction (which is based on the prediction of the model with X comes from the training sample) and the y-true value (which is based on the y of the training sample).

y_pred=svc.predict(X_training)
y_pred


# In[40]:


y_true = y_training
y_true


# In[41]:


#The absolute accuracy is measured as follow.

metric.accuracy_score(y_true, y_pred, normalize=False)


# In[47]:


#Confusion matrix is useful to see if there is misclassification. If there is no missclassification, then the corect values would be in the diagonal.

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))


# In[43]:


#SVM Prediction

#Now we can also use the trained SVM to predict something that is outside the training data. Let us predict the class y of the given test data [X1, X2] = [3, 6]

svc.predict([[3,6]])


# In[44]:


#The test data is now plotted.

x



# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Creating an Object Recognition System

# _Applied Machine Learning, Module 1:  A simple classification task_
# 
# The dataset is a small dataset derived from one originally created by Dr. Iain Murray at the University of Edinburgh for the task of training a classifier to **distinguish between different types of fruit**.

# ### Import required modules and load data file

# In[1]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

fruits = pd.read_table('fruit_data_with_colors.txt')


# In[2]:

fruits.head()


# In[3]:

fruits.shape # Contains 59 rows


# In[4]:

# create a mapping from fruit label value to fruit name to make results easier to interpret
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
lookup_fruit_name


# The file contains the mass, height, and width of a selection of oranges, lemons and apples. The heights were measured along the core of the fruit. The widths were the widest width perpendicular to the height.

# ### Examining the data

# In[8]:

# plotting a scatter matrix
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print (X_train.shape, X_test.shape)


# ### 1. Feature Pair plot
# This gives a general idea about interaction between different features and determine which can be good classifying features. This usually works on less than 20 variables.

# In[10]:

from matplotlib import cm

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# ### 2. 3-D Feature Scatter Plot

# In[12]:

# plotting a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()


# # k-NN Classfier
# 
# The K-Nearest Neighbors algorithm can be used for classification and regression. k-NN classifiers are an example of what's called **instance based or memory based supervised learning**. What this means is that instance based learning methods work by memorizing the labeled examples that they see in the training set. And then they use those memorized examples to classify new objects later. 
# 
# This works in 3 steps
# 1. When given a new previously unseen instance of something to classify, a k-NN classifier will look into its set of memorized training examples to find the k examples that have closest features.
# 2. The classifier will look up the class labels for those k-Nearest Neighbor examples.
# 3. combine the labels of those examples to make a prediction for the label of the new object. Typically, for example, by using simple majority vote. 

# In[13]:

# For this example, we use the mass, width, and height features of each fruit instance
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# #### 1. Create classifier object

# In[16]:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn


# ### Train the classifier (fit the estimator) using the training data

# In[17]:

knn.fit(X_train, y_train)


# ### Estimate the accuracy of the classifier on future data, using the test data

# In[18]:

knn.score(X_test, y_test)


# ### Use the trained k-NN classifier model to classify new, previously unseen objects

# In[19]:

# first example: a small fruit with mass 20g, width 4.3 cm, height 5.5 cm
fruit_prediction = knn.predict([[20, 4.3, 5.5]])
lookup_fruit_name[fruit_prediction[0]]


# In[20]:

# second example: a larger, elongated fruit with mass 100g, width 6.3 cm, height 8.5 cm
fruit_prediction = knn.predict([[100, 6.3, 8.5]])
lookup_fruit_name[fruit_prediction[0]]


# ### Plot the decision boundaries of the k-NN classifier

# In[22]:

from adspy_shared_utilities import plot_fruit_knn

plot_fruit_knn(X_train, y_train, 5, 'distance')   # we choose 5 nearest neighbors


# ### How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?

# In[23]:

k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);


# We see that, larger values of k do lead to worse accuracy for this particular dataset and fixed single train test split. Keep in mind that, these results are only for this particular training test split. To get a more reliable estimate of likely future accuracy for a particular value of k, we would want to look at results over multiple possible train test splits.
# 
# In general the best choice of the value of k, that is the one that leads to the highest accuracy, can vary greatly depending on the data set. In general with k-nearest neighbors, using a larger k suppresses the effects of noisy individual labels. But results in classification boundaries that are less detailed.

# ### How sensitive is k-NN classification accuracy to the train/test split proportion?

# In[24]:

t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');


# In[ ]:




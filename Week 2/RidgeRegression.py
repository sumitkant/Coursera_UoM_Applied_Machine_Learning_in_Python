
# coding: utf-8

# # Ridge Regression

# Ridge regression uses the same least-squares criterion, but with one difference. During the training phase, it adds a penalty for feature weights. The large weights means mathematically that the sum of their squared values is large. The addition of the penalty term is called **_regularization_**. It's a way to avoid over-fitting and **improve the likely generalization** of the model. The effect of this restriction from regularisation, is to **reduce the complexity** of the final estimated model.
# \begin{equation}
# RSS_{RIDGE} (w,b) = \sum_{i=1}^{N} (y_i - (w . x_i + b))^2 + \alpha \sum_{j=1}^p w_j^2
# \end{equation}
# 
# The amount of regularization is determined by the alpha paramter. Larger alpha means more regularization and simpler linear models with weights closer to zero. setting $\alpha$ = 0 means ordinary least squares regression
# 
# The practical effect of using ridge regression, is to find the feature weights that fit the data well in at least square sense, and that set lots of the feature weights two values that are very small. We don't see this effect with a single variable linear regression example, but for regression problems with dozens or hundreds of features, the accuracy improvement from using regularized linear regression like ridge regression could be significant.

# ### 1. Import Libraries

# In[2]:

get_ipython().magic('matplotlib notebook')
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap


# ### 2. Load Crime data

# In[3]:

# Communities and Crime dataset
from adspy_shared_utilities import load_crime_dataset
(X_crime, y_crime) = load_crime_dataset()
X_crime.head()


# In[4]:

# Variables in dataset
X_crime.columns


# ### 3. Test train split

# In[5]:

X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime, random_state = 0)
print ('Training Set Dimensions : ', X_train.shape)
print ('Test Set Dimensions : ', X_test.shape)


# ### 4. Ridge Regression

# In[6]:

from sklearn.linear_model import Ridge
linridge = Ridge(alpha = 20).fit(X_train, y_train)
linridge


# In[7]:

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))


# ### 5. Accuracy

# In[8]:

print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))


# ### Comparing with Linear Regression

# In[9]:

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg


# In[10]:

print('R-squared score (training): {:.3f}'
     .format(reg.score(X_train, y_train)))
print('R-squared score (test): {:.3f}'
     .format(reg.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(reg.coef_ != 0)))


# You'll notice here that the results are not that impressive. The R-squared score on the test set is pretty comparable to what we got for least-squares regression. 
# 
# #### The need for feature preprocessing and normalization. 
# Ridge regression is regularizing the linear regression by imposing that sum of squares penalty on the size of the $w$ coefficients. So the effect of increasing alpha is to shrink the $w$ coefficients toward zero and towards each other. But if the input variables, the features, have very different scales, then when this shrinkage happens of the coefficients, input variables with different scales will have different contributions to this L2 penalty, because the L2 penalty is a sum of squares of all the coefficients. So transforming the input features, so they're all on the same scale, means the ridge penalty is in some sense applied more fairly to all features without unduly weighting some more than others, just because of the difference in scales.

# ### 6. MinMax Scaler

# In[13]:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = Ridge().fit(X_train_scaled,y_train)
r2_score = clf.score(X_test_scaled, y_test)
r2_score


# In[15]:

# Alternatively fitting and transforming simultaenously
X_train_scaled = scaler.fit_transform(X_train)


# ### 7. Ridge Regression with Normalization and Feature Scaling

# In[16]:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X_crime, y_crime,
                                                   random_state = 0)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

print('Crime dataset')
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_test_scaled, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))


# In[28]:

print('Ridge regression: effect of alpha regularization parameter\n')
train_score = []
test_score = []
alpha = [0, 1, 10, 20, 50, 100, 1000]
for this_alpha in alpha:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, y_train)
    r2_train = linridge.score(X_train_scaled, y_train)
    r2_test = linridge.score(X_test_scaled, y_test)
    train_score = np.append(train_score, r2_train)
    test_score = np.append(test_score,r2_test) 
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {}, r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))


# In[31]:

test_score


# In[40]:

fig = plt.figure()
plt.plot(alpha, train_score, '-.', alpha, test_score, '-')
plt.ylim(-1, 1)
plt.legend(labels = ['Training Score','Test Score'])
plt.xticks(alpha)


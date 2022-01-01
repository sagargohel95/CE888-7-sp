#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, roc_auc_score, roc_curve, log_loss


# # Heart Disease dataset

# ## Task: 1

# In[2]:


# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# Dataset using from UCI Online 
data1 = pd.read_csv('heart.csv')
data1.head()


# In[3]:


data1.shape # shape of data


# In[4]:


data1.info() # Information about data


# In[5]:


data1.target.value_counts() # Balanced data


# In[6]:


data1[data1.duplicated()==True] # One Duplicate row


# In[7]:


data1 = data1.drop_duplicates() # Remove duplicate
data1.shape


# In[8]:


data1.describe().T # Descriptive analysis of data


# In[9]:


# Numerical Data separate for exploration
numerical_data = data1[['age','trestbps','chol','thalach','oldpeak','target']]
numerical_data.head() 


# In[10]:


# One hot encodingon categorical data
ohe_data = data1[['sex', 'cp', 'restecg', 'exang', 'slope', 'ca', 'thal']] # for convert dummies data
ohe_data.head()


# In[11]:


# Data Encoding
# for use of one hot encoding
for i in ohe_data:
    ohe_data[i] =ohe_data[i].astype(object)
ohe_data.info()


# In[12]:


dummies = pd.get_dummies(ohe_data, drop_first=True) # get dummies to make dummy column
dummies.head()


# In[13]:


# num_data = numerical_data.drop('target', axis=1)
#from sklearn.preprocessing import StandardScaler 
# X = StandardScaler().fit_transform(num_data)
#X[:5] # Scaling data

num_data = numerical_data.drop('target', axis=1)
from sklearn.preprocessing import StandardScaler 
X = StandardScaler().fit_transform(num_data)
X[:5] # Scaling data


# In[14]:


scaler_data = pd.DataFrame(X, index=num_data.index, columns=num_data.columns)
scaler_data.head() # Scaler data in pandas dataframe


# In[15]:


new_data = pd.concat([scaler_data, dummies],axis=1) 
new_data.head()  # concat our scalerdata and dummies data


# In[16]:


new_data['target'] = data1.target # add target columnin new_data
new_data.head()


# In[17]:


X = new_data.drop('target', axis=1)
y =new_data.target
# Split data in train and test dataset
X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=10) # training size is 0.75
X_train.shape, X_test.shape 


# ###  TAsk: 2 Random Forest

# In[18]:


rforest=RandomForestClassifier(n_estimators=100,  random_state=100)
rforest.fit(X_train, y_train)


# In[19]:


preds = rforest.predict(X_test)


# In[20]:


round(accuracy_score(y_test, preds), 2) # Accuracy


# In[21]:


new_data = confusion_matrix(y_test, preds)
sns.heatmap(new_data, annot=True)
plt.title('Normal random forest')


# In[22]:


print(classification_report(y_test, preds)) # classification report of random forest


# ### Task 3: Get probabilities and make new dataframe

# In[23]:


probabilities = rforest.predict_proba(X_train)
probabilities[:5] # Getting probabilities for training data


# In[24]:


probabilities_test = rforest.predict_proba(X_test)
probabilities_test[:5] # Getting probabilities for testing data


# In[25]:


# Both probabilities are in array so we need to convert them dataframe so we can concat them
probab_df = pd.DataFrame(probabilities)
probab_test_df = pd.DataFrame(probabilities_test)
probab_df.head() 


# In[26]:


# Reset index to avoid NAN Value.
X_train.reset_index(drop=True, inplace=True)
probab_df.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)
probab_test_df.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Concat training data and training probabilities
full = pd.concat([X_train,probab_df,y_train], axis=1)
# Concat Testing data and training probabilities
full_test = pd.concat([X_test,probab_test_df,y_test], axis=1)
full_test.head()


# In[27]:


X_train_new = full.iloc[:,:-1] # New tarining data
X_test_new = full_test.iloc[:,:-1] # New Testing data
X_train_new.head()


# In[28]:


y_train_new = full.target # Training target
y_test_new = full_test.target # Testing target
X_test_new.shape, y_test_new.shape # Shape of testing dta


# ### Task 4:  Decision Tree Classifier On new Data

# In[29]:


dt = DecisionTreeClassifier()
dt.fit(X_train_new, y_train_new) # Using new data


# In[30]:


dt_pred = dt.predict(X_test_new)
round(accuracy_score(y_test_new, dt_pred), 2) # Accuracy


# In[31]:


print(classification_report(y_test_new, dt_pred))


# In[32]:


fig = plt.figure(figsize=(20,10))
tree.plot_tree(dt, filled=True)
plt.title('Decision Tree Diagram');


# #### Hyper parameter tunning on Decision Tree 

# In[33]:


# Make dictionary of all parameter
model_params = {
      'Decision Tree':{
          'model' : DecisionTreeClassifier(),
          'params': {
              'criterion': ['gini','entropy'],
              'splitter':['best', 'random'],
              'max_depth':[1,5,10,20,50,100],
              'max_features':['auto','sqrt','log2'],
          }
      }
}


# In[35]:


# Hyper parameter tunning using GridSearchCV
scores = [] # To make list of best features
for model_name, mp in model_params.items():
    clf_dt = GridSearchCV(mp['model'],mp['params'],cv = 5, return_train_score=False)
    clf_dt.fit(X_train_new,y_train_new)
    scores.append({
        'model': model_name,
        'best_score': clf_dt.best_score_,
        'best_params': clf_dt.best_params_
    })


# In[36]:


scores # List of our best parameter with best score


# In[37]:


preds_clfdt = clf_dt.predict(X_test_new)
round(accuracy_score(y_test_new, preds_clfdt), 2) # accuracy


# In[38]:


print(classification_report(y_test_new, preds_clfdt))


# ### Task 5: Decision Tree with original data using best hyper parameter

# In[39]:


clf_dt.best_params_ # Now need to use these all parameter for build Decision tree


# In[40]:


dt_orig = DecisionTreeClassifier(criterion='entropy',max_depth=50,max_features='auto',splitter='best')
dt_orig.fit(X_train,y_train) # original training and testing data


# In[41]:


dt_orig_pred = dt_orig.predict(X_test)
round(accuracy_score(y_test, dt_orig_pred), 2)


# In[42]:


print(classification_report(y_test,dt_orig_pred))


# ### Task 6: 

# In[43]:


print(f'Accuracy of Random foreston Original Data : {round(accuracy_score(y_test, preds), 2)}')
print(f'Accuracy of Decision Tree knowledge distillation from random forests : {round(accuracy_score(y_test_new, dt_pred), 2)}')
print(f'Accuracy of Decision Tree using Hyper parameter tunning :{round(accuracy_score(y_test_new, preds_clfdt), 2)}')
print(f'Accuracy of Decision Tree using Hyper parameter tunning  On original data :{round(accuracy_score(y_test_new, preds_clfdt), 2)}')


# ###### Best Model is our second Decison tree model who is build from knowledge distillation of random forest

# In[ ]:





# In[ ]:





# In[ ]:





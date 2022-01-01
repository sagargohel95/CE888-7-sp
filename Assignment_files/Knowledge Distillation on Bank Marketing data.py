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


# In[2]:


# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# dataset using from UCI online 
Bank_data = pd.read_csv("bank-full.csv",delimiter = ";")
Bank_data.head()


# In[3]:


Bank_data.dtypes


# In[4]:


cat_attr = Bank_data.columns[Bank_data.dtypes==object].tolist()
cat_attr


# In[5]:


for i in cat_attr:
    print('Unique values for ' + i)
    print(Bank_data[i].unique())
    print('')


# In[6]:


for col in cat_attr:
    plt.figure(figsize=(10,4))
    sns.barplot(Bank_data[col].value_counts().values, Bank_data[col].value_counts().index)
    plt.title(col)
    plt.tight_layout()


# In[7]:


Bank_data.corr()


# In[8]:


missing_values = Bank_data.isnull().mean()*100 # Checking Missing values
missing_values.sum()


# In[9]:


Bank_data[Bank_data.duplicated()].iloc[0:,:] #Checking Duplicate


# In[10]:


# Remove outliers using IQR
def IQR(df,x): 
    q1 = df[x].quantile(0.25)
    q3 = df[x].quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    cleaned_data = df.loc[(df[x] > fence_low) & (df[x] < fence_high)]
    return cleaned_data


# In[11]:


cleaned1 = IQR(Bank_data,"age")
cleaned1.shape


# In[12]:


cleaned1 = IQR(cleaned1, "campaign")
cleaned1.shape


# In[13]:


cleaned1 = IQR(cleaned1, "balance")
cleaned1.shape


# In[14]:


df = cleaned1.copy()
df
#Generating Dummy Variables for categorical data
df[cat_attr[:-1]] = df[cat_attr[:-1]].astype(str)

num_cattr = []
for i in df.columns:
    if i not in cat_attr:
        num_cattr.append(i)

num_cattr


# In[15]:


df[cat_attr[:-1]] = df[cat_attr[:-1]].astype(str)
df[num_cattr] = df[num_cattr].astype(float)

model_data = pd.get_dummies(df.iloc[:,:-1], drop_first=True).copy()
model_data.columns


# In[16]:


model_data = pd.concat([model_data,df.iloc[:,-1]],axis=1)
model_data.shape


# In[17]:


# Independent Variable
X = model_data.loc[:, ~model_data.columns.isin(['y'])]
#Dependent Variable
y = model_data.loc[:,'y'].copy()
# Dividing data in Training and testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape, X_test.shape


# # Task: 2 Random Forest

# In[18]:


rforest=RandomForestClassifier(n_estimators=100,  random_state=100)
rforest.fit(X_train, y_train)


# In[19]:


preds = rforest.predict(X_test)


# In[20]:


round(accuracy_score(y_test, preds), 2) # Accuracy


# In[21]:


print(classification_report(y_test, preds)) # classification report of random forest


# # Task 3: Get probabilities and make new dataframe

# In[22]:


probabilities = rforest.predict_proba(X_train)
probabilities[:5] # Getting probabilities for training data


# In[23]:


probabilities_test = rforest.predict_proba(X_test)
probabilities_test[:5] # Getting probabilities for testing data


# In[24]:


# Both probabilities are in array so we need to convert them dataframe so we can concat them
probab_df = pd.DataFrame(probabilities)
probab_test_df = pd.DataFrame(probabilities_test)
probab_df.head() 


# In[25]:


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


# In[26]:


X_train_new = full.iloc[:,:-1] # New tarining data
X_test_new = full_test.iloc[:,:-1] # New Testing data
X_train_new.head()


# In[27]:


y_train_new = full.y # Training target
y_test_new = full_test.y # Testing target
X_test_new.shape, y_test_new.shape # Shape of testing dta


# # Task 4: Decision Tree Classifier On new Data

# In[28]:


dt = DecisionTreeClassifier()
dt.fit(X_train_new, y_train_new) # Using new data


# In[29]:


dt_pred = dt.predict(X_test_new)
round(accuracy_score(y_test_new, dt_pred), 2) # Accuracy


# In[30]:


print(classification_report(y_test_new, dt_pred))


# In[31]:


fig = plt.figure(figsize=(20,10))
tree.plot_tree(dt, filled=True)
plt.title('Decision Tree Diagram');


# In[32]:


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


# In[33]:


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


# In[34]:


scores # List of our best parameter with best score


# In[35]:


preds_clfdt = clf_dt.predict(X_test_new)
round(accuracy_score(y_test_new, preds_clfdt), 2) # accuracy


# In[36]:


print(classification_report(y_test_new, preds_clfdt))


# # Task 5: Decision Tree with original data using best hyper parameter

# In[37]:


clf_dt.best_params_ # Now need to use these all parameter for build Decision tree


# In[38]:


dt_orig = DecisionTreeClassifier(criterion='gini',max_depth=50,max_features='sqrt',splitter='random')
dt_orig.fit(X_train,y_train) # original training and testing data


# In[39]:


dt_orig_pred = dt_orig.predict(X_test)
round(accuracy_score(y_test, dt_orig_pred), 2)


# In[40]:


print(classification_report(y_test,dt_orig_pred))


# # Task 6:

# In[43]:


print(f'Accuracy of Random foreston Original Data : {round(accuracy_score(y_test, preds), 2)}')
print(f'Accuracy of Decision Tree knowledge distillation from random forests : {round(accuracy_score(y_test_new, dt_pred), 2)}')
print(f'Accuracy of Decision Tree using Hyper parameter tunning :{round(accuracy_score(y_test_new, preds_clfdt), 2)}')
print(f'Accuracy of Decision Tree using Hyper parameter tunning  On original data :{round(accuracy_score(y_test_new, preds_clfdt), 2)}')


# In[ ]:





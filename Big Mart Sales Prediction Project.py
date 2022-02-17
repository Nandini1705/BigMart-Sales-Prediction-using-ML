#!/usr/bin/env python
# coding: utf-8

# ###### Importing the librabies

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ###### Warnings for clean notebook 

# In[49]:


import warnings # Warnings for clean notebook 
warnings.filterwarnings('ignore')


# ###### Reading the Data

# In[50]:


df=pd.read_csv('Downloads\Train.csv')


# In[51]:


df


# In[52]:


#First 5 rows of the dataframe
df.head() 


# In[53]:


# No.of data points and no. of features
df.shape 


# In[54]:


df.columns


# In[55]:


#Getting some information about the dataset
df.info() 


# ###### Missing Value Identification

# In[56]:


df.isnull().sum() #Checking for missing values


# ###### The observation here is Item_Weight which is numerical feature has 1463 missing values and Outlet_Size  which is Categorical Feature has 2410 missing values

# ###### Handling Missing Values for Numerical Feature

# In[57]:


#Mean --> average
#Mode --> more repeated value
# mean value of "Item_Weight" column
df['Item_Weight'].mean()


# In[58]:


# filling the missing values in "Item_weight column" with "Mean" value
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)


# ###### Handling Missing Values for Categorical Feature

# In[59]:


# mode of "Outlet_Size" column
df['Outlet_Size'].mode()


# In[60]:


# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))


# In[61]:


print(mode_of_Outlet_size)


# In[62]:


miss_values = df['Outlet_Size'].isnull()  


# In[63]:


print(miss_values)


# In[64]:


df.loc[miss_values, 'Outlet_Size'] = df.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])


# In[65]:


# checking for missing values
df.isnull().sum()


# ###### Data Analysis

# In[66]:


#Statistical measures about the data
df.describe() 


# ###### Distribution chart for Numerical Features

# In[67]:


# Item_Weight distribution
plt.figure(figsize=(6,6))
sns.distplot(df['Item_Weight'])
plt.show()


# ###### The observation here is maximum item weight is distributed between 10-15kg .So the mean value we got is 12.85 which is avg weight

# In[68]:


# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(df['Item_Visibility'])
plt.show()


# In[69]:


# Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(df['Item_MRP'])
plt.show()


# ###### The obsevation here is that more products range between 50-250$

# ###### Target Analysis

# In[70]:


# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(df['Item_Outlet_Sales'])
plt.show()


# ###### The observation here is one sided distribution which is not a noramal distribution and the sales are in the range of 2000-10000

# In[71]:


# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=df)
plt.show()


# ###### This tells us these are years where different outlet stores are established and more products are created in 1985 and observed similar for all rest years.

# ###### Bar chart for Categorical Features

# In[72]:


# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=df)
plt.show()


# ###### The observation here in Item_Fat_Content is we need to clean data because(lowfat,LF and reg) represents similar as Low fat and Regular. So we need to process this data

# In[73]:


# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=df)
plt.show()


# In[74]:


# Outlet_Size column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Size', data=df)
plt.show()


# ###### Data Pre-Processing

# In[75]:


df.head()


# In[76]:


df['Item_Fat_Content'].value_counts()


# In[77]:


df.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)


# In[78]:


df['Item_Fat_Content'].value_counts()


# ###### Transforming object data types into vector

# In[79]:


#Label Encoding
encoder = LabelEncoder() 


# In[80]:


df.select_dtypes(include=['object'])


# In[81]:


df['Item_Identifier'] = encoder.fit_transform(df['Item_Identifier'])

df['Item_Fat_Content'] = encoder.fit_transform(df['Item_Fat_Content'])

df['Item_Type'] = encoder.fit_transform(df['Item_Type'])

df['Outlet_Identifier'] = encoder.fit_transform(df['Outlet_Identifier'])

df['Outlet_Size'] = encoder.fit_transform(df['Outlet_Size'])

df['Outlet_Location_Type'] = encoder.fit_transform(df['Outlet_Location_Type'])

df['Outlet_Type'] = encoder.fit_transform(df['Outlet_Type'])


# In[82]:


df.head()


# ###### Splitting features and Target

# In[83]:


x=df.drop(['Item_Outlet_Sales'],axis=1) # independent variables
y=df['Item_Outlet_Sales']               # dependent variables or target column


# In[84]:


print(x)


# ###### Splitting the data into Training data & Testing Data

# In[462]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)


# In[463]:


print(x.shape, x_train.shape, x_test.shape)


# ###### Scaling the data

# In[464]:


scaler = StandardScaler()


# In[465]:


scaler.fit(x_train)


# In[466]:


x_train_standardized = scaler.transform(x_train)


# In[467]:


print(x_train_standardized)


# In[468]:


x_test_standardized = scaler.transform(x_test)


# In[469]:


print(x_train_standardized.std())


# In[470]:


print(x_test_standardized.std())


# ###### Machine Learning Model Training

# ###### LinearRegression Model

# In[471]:


# 1.LinearRegression Model
#df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1, inplace=True)
from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(x_train,y_train)

lr_pred = model.predict(x_test)
lr_pred
lr_accuracy = round(lr.score(x_train,y_train)*100)
lr_accuracy


# ###### L1 regularization model

# In[472]:


#2. Using L1 regularization model ( Lasso ) for a balanced fit 
from sklearn import linear_model
lasso=linear_model.Lasso(alpha=50)
lasso.fit(x_train,y_train)


# In[473]:


# prediction on test data
lasso.predict(x_test)
lasso_accuracy = round(lr.score(x_train,y_train)*100)
lasso_accuracy


# ###### L2 regularization model

# In[474]:


#2. Using L2 regularization model ( Ridge )

from sklearn.linear_model import Ridge
ridge=Ridge(alpha=50)
ridge.fit(x_train,y_train)


# In[475]:


# prediction on test data
ridge.predict(x_test)
Ridge_accuracy = round(lr.score(x_train,y_train)*100)
Ridge_accuracy


# ###### XGBoost Regressor

# In[476]:


from xgboost import XGBRegressor
from sklearn import metrics
regressor = XGBRegressor()
regressor.fit(x_train, y_train)


# In[480]:


# prediction on test data
test_data_prediction = regressor.predict(x_test)
# R squared Value
r2_test = metrics.r2_score(y_test, test_data_prediction)
print('R Squared value = ', r2_test)


# ###### DecisionTreeRegressor

# In[488]:


from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

tree.fit(x_train,y_train)
tree_pred = tree.predict(x_test)


# In[489]:


tree_pred


# In[490]:


tree_accuracy = round(tree.score(x_train,y_train)*100)
tree_accuracy


# ###### RandomForestRegressor

# In[492]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=400,max_depth=6,min_samples_leaf=100,n_jobs=4)

rf.fit(x_train,y_train)

rf_accuracy = round(rf.score(x_train,y_train)*100)

rf_accuracy


# In[ ]:


# Conclusion
# We see RandomForestRegressor and DecisionTreeRegressor proves to be beneficial which gives us 61% accuracy than 
#Simple linear regression model 
#L1 and L2 regularization and
#XGBoost Regressor


# In[ ]:





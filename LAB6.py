#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df = pd.read_csv(r"C:\Users\joaof\ironhack\labs\lab-customer-analysis-round-2\files_for_lab\csv_files\marketing_customer_analysis.csv")


# In[4]:


df = pd.read_csv(r"C:\Users\joaof\ironhack\labs\lab-customer-analysis-round-4\files_for_lab\csv_files\marketing_customer_analysis.csv")


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.head(5)


# In[8]:


cols = []


# In[9]:


for i in range(len(df.columns)):
    cols.append(df.columns[i].lower().replace(' ', '_'))


# In[10]:


df.columns = cols


# In[11]:


df


# In[12]:


numeric_columns = df.select_dtypes(include=['int', 'float'])


# In[13]:


numeric_columns


# In[14]:


categorical_columns = df.select_dtypes(include= ['object'])


# In[15]:


categorical_columns


# In[16]:


import seaborn as sns


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


sns.histplot(data=numeric_columns, x = 'income')
plt.ylabel('frequency')
plt.xlabel('income')
plt.show()


# In[19]:


sns.histplot(data=numeric_columns, x = 'monthly_premium_auto')
plt.ylabel('frequency')
plt.xlabel('monltlhy_premium_auto')
plt.show()


# In[20]:


sns.histplot(data=numeric_columns, x = 'income')
plt.ylabel('frequency')
plt.xlabel('monltlhy_premium_auto')
plt.show()


# In[21]:


sns.displot(data=numeric_columns, x = 'months_since_last_claim')
plt.ylabel('frequency')
plt.xlabel('Months Since Last Claim')
plt.show()


# In[22]:


sns.displot(data=numeric_columns, x = 'number_of_policies')
plt.ylabel('frequency')
plt.xlabel('Number of open complaints')
plt.show()


# In[23]:


sns.displot(data=numeric_columns, x = 'number_of_open_complaints')
plt.ylabel('frequency')
plt.xlabel('Number of open complaints')
plt.show()


# In[24]:


correlation_matrix =  numeric_columns.corr()


# In[25]:


correlation_matrix


# In[26]:


sns.heatmap(correlation_matrix, annot=True)
plt.show()
mask = np.zeros_like(correlation_matrix)
mask[np.triu_indices_from(mask)] = True 
fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(correlation_matrix, mask=mask, annot=True)
plt.show()


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


df_train, df_test = train_test_split(df, test_size=0.2, random_state=100)
print(df_train.shape, df_test.shape)


# In[29]:


#X-Y SPLIT
X = numeric_columns.drop('customer_lifetime_value', axis=1)
Y= numeric_columns['customer_lifetime_value']


# In[30]:


print("Features (X):")
print(X)
print("\nTarget (Y):")
print(Y)


# In[31]:


categorical_df = df.select_dtypes(include=['object', 'category'])


# In[32]:


print(categorical_df)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


dfx = pd.DataFrame(categorical_df)


# In[33]:


one_hot_encoded_df1 = pd.get_dummies(categorical_df.columns)


# In[37]:


print("Original Dataframe:")

print("\n0ne-Hot Encoded Dataframe:")
print(one_hot_encoded_df1)


# In[38]:


print(df)


# In[40]:


#concat dataframes
dataframe = pd.concat([numeric_columns, categorical_df], axis=1)


# In[41]:


dataframe


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


from sklearn.linear_model import LinearRegression


# In[53]:


from sklearn.metrics import mean_squared_error, r2_score


# In[54]:


df2 = pd.DataFrame(numeric_columns)


# In[55]:


import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[59]:


Y = numeric_columns['customer_lifetime_value']
X = numeric_columns.drop(['customer_lifetime_value'], axis=1)


# In[60]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[62]:


model = LinearRegression()


# In[63]:


model.fit(X_train, Y_train)


# In[66]:


Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Predictions:", Y_pred)


# 

# In[ ]:





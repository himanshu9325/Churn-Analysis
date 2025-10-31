#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# # connect to sql database 
# conn=mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='Himanshu@9325',
#     database='churn_db'
# )


# In[3]:


# query="SELECT * FROM customers;"
# df=pd.read_sql(query,conn)

df=pd.read_csv("Customers.csv")
# In[4]:

df.info()


# In[5]:


df.drop(columns=['customer_id'],inplace=True)


# In[6]:


df["gender"]=df["gender"].map({"Male":0,"Female":0})
df.info()


# In[36]:


# EDA HW

# df.info()

for col in df.columns:
    plt.Figure(figsize=(10,5))
    plt.hist(df[col])
    plt.title(col)
    plt.show()


# In[30]:


plt.Figure(figsize=(15,10))
sns.pairplot(df)
plt.show()


# In[7]:


# model Building
X=df.drop(columns="churn")
Y=df["churn"]


# In[8]:


# feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


# In[9]:


# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)


# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[12]:


y_pred=model.predict(X_test)


# In[13]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(Y_test,y_pred)
cr=classification_report(Y_test,y_pred)
accu=accuracy_score(Y_test,y_pred)


# In[14]:


print("Confusion Matrix:\n",cm)
print("\nClassification Report:\n",cr)
print("\nAccuracy Score:",accu)


# In[15]:


# save model
import pickle
with open("model.pkl","wb") as f:
    pickle.dump((model,scaler),f)
print("Model saved successfully.")


# In[ ]:





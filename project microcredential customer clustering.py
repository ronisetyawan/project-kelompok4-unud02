#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv( "food_delivery_datasets.csv" )
data


# In[3]:


data.drop(['random_id', 'status', 'resto_id'], axis=1, inplace=True)
data


# In[4]:


data.info()


# In[5]:


data = data.dropna()
data.shape


# In[6]:


data["cust_id"]= data["cust_id"].astype(str)


# In[7]:


# New Attribute : Monetary

data['Amount'] = data['delivery_fee']+data['food_price']
rfm_m = data.groupby('cust_id')['Amount'].sum()
rfm_m = rfm_m.reset_index()
rfm_m.head()


# In[8]:


# New Attribute : Frequency

rfm_f = data.groupby('cust_id')['order_id'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['cust_id', 'Frequency']
rfm_f.head()


# In[9]:


rfm = pd.merge(rfm_m, rfm_f, on='cust_id', how='inner')
rfm.head()


# In[10]:


# New Attribute : Recency

# Convert to datetime to proper datatype

data['date_time'] = pd.to_datetime(data['date_time'],format='%Y-%m-%dT%H:%M:%S')

# Compute the maximum date to know the last transaction date

max_date = max(data['date_time'])
max_date

# Compute the difference between max date and transaction date

data['Diff'] = max_date - data['date_time']
data.head()

# Compute last transaction date to get the recency of customers

rfm_p = data.groupby('cust_id')['Diff'].min()
rfm_p = rfm_p.reset_index()
rfm_p.head()


# In[11]:


# Extract number of days only

rfm_p['Diff'] = rfm_p['Diff'].dt.days
rfm_p.head()


# In[12]:


# Merge tha dataframes to get the final RFM dataframe

rfm = pd.merge(rfm,rfm_p, on='cust_id', how='inner')
rfm.columns = ['cust_id', 'Amount', 'Frequency', 'Recency']
rfm.head()


# In[13]:


# Outlier Analysis of Amount Frequency and Recency

attributes = ['Amount','Frequency','Recency']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')


# In[14]:


# Removing (statistical) outliers for Amount
Q1 = rfm.Amount.quantile(0.05)
Q3 = rfm.Amount.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Amount >= Q1 - 1.5*IQR) & (rfm.Amount <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]


# In[15]:


# Rescaling the attributes

rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[16]:


rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
rfm_df_scaled.head()


# In[17]:


#building model
#Hierarchical Clustering
# Complete linkage

mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[18]:


# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


# In[19]:


# Assign cluster labels

rfm['Cluster_Labels'] = cluster_labels
rfm.head()


# In[20]:


sns.pairplot(rfm, hue='Cluster_Labels')


# Hierarchical Clustering with 3 Cluster Labels
# 
# Customers with Cluster_Labels 2 are the customers with high amount of transactions as compared to other customers. Customers with Cluster_Labels 2 are frequent buyers. Customers with Cluster_Labels 1 are not recent buyers and hence least of importance from business point of view.

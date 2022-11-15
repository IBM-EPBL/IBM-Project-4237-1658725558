#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import calendar


# In[2]:


data=pd.read_csv("autos.csv",encoding = "latin-1")


# In[3]:


data.head()


# In[4]:


data.describe()


# In[5]:


data.corr()


# In[6]:


#dropping unwanted columns
print(data.seller.unique())
print(data.offerType.unique())
print(data.abtest.unique())
print(data.nrOfPictures.unique())


# In[7]:


data.drop(["name","seller","offerType","nrOfPictures","lastSeen","dateCreated","postalCode","dateCrawled"],axis="columns",inplace=True)


# In[8]:


data


# # Data Cleaning

# In[9]:


data_car = data.copy()

# Filter bad data
data_car = data_car[
    (data_car["yearOfRegistration"].between(1945, 2017, inclusive=True)) &
    (data_car["powerPS"].between(100, 500, inclusive=True)) &
    (data_car["price"].between(100, 200000, inclusive=True))] 


# In[10]:


#checking for null values
data_car.isnull().any()


# In[11]:


data_car.isnull().sum()


# In[12]:


data_car["gearbox"].value_counts()


# In[13]:


data_car['gearbox'].fillna(value='manuell', inplace=True)


# In[14]:


data_car["gearbox"].isnull().sum()


# In[15]:


data_car["notRepairedDamage"].value_counts()


# In[16]:


data_car["notRepairedDamage"].isnull().sum()


# In[17]:


data_car["notRepairedDamage"].fillna("nein",inplace = True)


# In[18]:


data_car["notRepairedDamage"].isnull().sum()


# In[19]:


data_car["fuelType"].value_counts()


# In[20]:


data_car["fuelType"].fillna("benzin",inplace = True)


# In[21]:


data_car.isnull().sum()


# In[22]:


data_car["vehicleType"].value_counts()


# In[23]:


#we can fill according to fueltype values
data_car.groupby("fuelType")["vehicleType"].value_counts()


# In[24]:


data_car["vehicleType"].fillna("blank",inplace = True)


# In[25]:


data_car["vehicleType"].isnull().sum()


# In[26]:


len(data_car["model"].unique())


# In[27]:


data_car["model"].value_counts()


# In[28]:


data_car["model"].fillna("3er",inplace =True)


# In[29]:


data_car.head()


# In[30]:


data_car.describe()


# In[31]:



#checking for null values again in our data
data_car.isnull().sum()


# # Data Vizualization

# In[32]:


cat_val = ["abtest","vehicleType", "brand", "model", "gearbox","fuelType", "notRepairedDamage"]

for i,col in enumerate(cat_val):
    v=data_car[col].unique()
    g = data_car.groupby(by=col)[col].count().sort_values(ascending=False)
    r=range(min(len(v),5))
    print(g.head())
    plt.figure(figsize=(5,3))
    plt.bar(r,g.head())
    plt.xticks(r, g.index)
    plt.show()


# In[33]:


from pandas.plotting import scatter_matrix
num_attributes = ["price", "yearOfRegistration", "powerPS", "kilometer"]

pd.plotting.scatter_matrix(data_car[num_attributes], figsize = (12,8), alpha = 0.1)


# In[34]:


data_car["price"].hist(bins = 50, log = True)


# In[35]:



data_car.corr()


# In[36]:


def plot_correlation_map( df ):
    corr = data_car.corr()
    _ , ax = plt.subplots( figsize =( 20 , 10 ) )
    #cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = "YlGnBu",
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )
plot_correlation_map(cat_val)


# In[37]:


plt.scatter(data_car['yearOfRegistration'],data_car['price'])


# # Encoding categorical values

# In[38]:


data_car.head()

# ['vehicleType']  :  [nan 'coupe' 'suv' 'kleinwagen' 'limousine' 'cabrio' 'bus' 'kombi'
#  'andere']
# ['brand']  :  ['volkswagen' 'audi' 'jeep' 'skoda' 'bmw' 'peugeot' 'ford' 'mazda'
#  'nissan' 'renault' 'mercedes_benz' 'opel' 'seat' 'citroen' 'honda' 'fiat'
#  'mini' 'smart' 'hyundai' 'sonstige_autos' 'alfa_romeo' 'subaru' 'volvo'
#  'mitsubishi' 'kia' 'suzuki' 'lancia' 'porsche' 'toyota' 'chevrolet'
#  'dacia' 'daihatsu' 'trabant' 'saab' 'chrysler' 'jaguar' 'daewoo' 'rover'
#  'land_rover' 'lada']
#     ['gearbox']  :  ['manuell' 'automatik' nan]
# ['fuelType']  :  ['benzin' 'diesel' nan 'lpg' 'andere' 'hybrid' 'cng' 'elektro']
# ['notRepairedDamage']  :  [nan 'ja' 'nein']
# # In[39]:


data_car.dtypes


# In[40]:


from sklearn.preprocessing import LabelEncoder
category_col=["abtest","vehicleType","fuelType","gearbox","notRepairedDamage","brand","model"]
mapping_dict ={} 
labelEncoder=LabelEncoder()
for col in category_col: 
    data_car[col] = labelEncoder.fit_transform(data_car[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
print(mapping_dict) 


# In[41]:


data_car.head()


# # model building

# In[42]:


#splitting dependent and independent varibles
x=data_car.iloc[:,1:].values
x


# In[43]:


y=data_car.iloc[:,0:1].values
y


# In[44]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x)
x


# In[45]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


# In[46]:


from sklearn.linear_model import LinearRegression
mr=LinearRegression()
mr.fit(x_train,y_train)


# In[47]:


y_pred_mr=mr.predict(x_test)
y_pred_mr


# In[48]:


y_test


# In[49]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred_mr)


# In[50]:


y_test[10:15]


# In[51]:


y_pred_mr[10:15]


# In[60]:


from sklearn.tree import DecisionTreeRegressor
dt= DecisionTreeRegressor(criterion='mse',random_state=0) #mse is mean square error
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)

tree_rmse = np.sqrt(y_pred_dt)
tree_rmse
r2_score(y_test,y_pred_dt)


# In[53]:


y_test[5:8]


# In[54]:


y_pred_dt[5:8]


# In[55]:


#import library for random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(random_state=42, n_jobs =-1, max_depth = 30)
#n_estimators no of decision trees
rf.fit(x_train,y_train)
y_pred_rf=dt.predict(x_test)
r2_score(y_test,y_pred_rf)


# In[58]:


from joblib import dump
dump(rf,"rf.save")


# In[ ]:





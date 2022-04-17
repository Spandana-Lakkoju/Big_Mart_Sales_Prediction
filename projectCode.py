import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

train_set=pd.read_csv("Train.csv")
test_set=pd.read_csv("Test.csv")




print("####Dimension Details ##########")
print("Dimension of Train data:",train_set.shape) 
print("Dimension of Test data:",test_set.shape) 
print("##############################")    
k=train_set.head(5)
print(k)
kk=train_set.describe()
print(train_set.describe())
print("Training Set Information")
print(train_set.info())
item_count=train_set['Item_Identifier'].value_counts().sort_values()
print("###############Information Training Data#################")
# reading train and test data
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.head())
print("###No. of Null values##################")
print(test_data.apply(lambda x: sum(x.isnull())))
print("Unique values of Item fat content column#####")
print(test_data['Item_Fat_Content'].unique())
print("# combining Item_Fat_Content misspelled")
train_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
print("# creating new column num_years")
train_data['num_years'] = train_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x) 
test_data['num_years'] = test_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)     

print("######Print Unique Item Type ##########################")
print(train_data['Item_Type'].unique())
#Generate Graph
import seaborn as sns
sns.factorplot('Outlet_Location_Type',data = train_data,hue ='Outlet_Size' ,kind='count')

sns.factorplot('Outlet_Location_Type',data = test_data,hue ='Outlet_Size' ,kind='count')

sns.factorplot('Outlet_Type','num_years',data = test_data,hue='Outlet_Size' )

sns.factorplot('Outlet_Type','num_years',data = train_data,hue='Outlet_Size' )

full_data = [train_data, test_data]
# filling null values
for data in full_data:
    data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace = True)
    data['Outlet_Size'].fillna('Medium',inplace = True)

col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
# handling catagorical variables
train_datar = pd.get_dummies(train_data, columns = col, drop_first = True)
test_datar = pd.get_dummies(test_data, columns = col,drop_first = True)


feat_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'num_years',
       'Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Size_Medium', 'Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']

X = train_datar[feat_cols]
y = train_datar['Item_Outlet_Sales']

# splitting data as X_train and X_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)


# linear regression model 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test)
# SVM model 
from sklearn.svm import SVR
reg = SVR() 
reg.fit(X_train, y_train) 
y_pred = reg.predict(X_test)


# calculating RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Square Error:",rmse)

#creating results .csv file
result = pd.DataFrame()
result['Item_Identifier'] = test_datar['Item_Identifier']
result['Outlet_Identifier'] = test_datar['Outlet_Identifier']

# predicting on actual test data
X_t = test_datar[feat_cols]
y_result = reg.predict(X_t)

result["Item_Outlet_Sales"] = y_result
result = result.sort_index()
result.to_csv('Bigmart_XGBoost3.csv',index = False)





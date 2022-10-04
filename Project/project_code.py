import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import os 
print(os.getcwd())

"""Step:1 Problem Statemen 
        To Apply Charges on Smoker"""


# # Data Gathering
df = pd.read_csv("D:/API_creation2/Project/medical_insurance.csv")
print(df.head())

## EDA ##

print(df.isna().sum())

## feature Engineering
# SEX#
print(df['sex'].unique())

df["sex"].replace({"female":0,"male":1},inplace=True)
print(df.info)

df["smoker"].replace({"yes":1,"no":0},inplace=True)


print(df["region"].unique())
print()

df=pd.get_dummies(df,columns=["region"])

print(df)
print(df.sample())


# Feature Selection


# Model Training

x =df.drop("charges",axis=1)  # 2d
y =df["charges"] # 1d

x_train,x_test,y_train,y_test =train_test_split(x,y,train_size=0.8,random_state=40)

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)


## Evalution ##

#On Testing Data

y_pred_test = lr_model.predict(x_test)

MSE = mean_squared_error(y_test, y_pred_test)
print("MSE =>>>>",MSE)
RMSE =np.sqrt(MSE)
MAE = mean_absolute_error(y_test, y_pred_test)
print(MAE)

# Testing User input 
age =19
sex ='male'
bmi=28
children = 4
smoker = 'no'
region ='southwest'

project_data ={'sex':{"female":0,"male":1},"smoker":{'yes':1,'no':0},"columns":list(x.columns)}


columns_name =list(x.columns)
# create test array

test_array = np.zeros(len(x.columns))
test_array[0]=age
test_array[1]=project_data['sex'][sex]
test_array[2]=bmi
test_array[3]= children
test_array[4]= project_data['smoker'][smoker]
region = "region_"+region
region_index= np.where(columns_name==region)
test_array[region_index] = 1
print(test_array)
lr_model.predict([test_array])


## create json file##
with open ('project_data.json','w') as file:
        json.dump(project_data,file)

## create pickle file 
with open ("Linear_model.pkl","wb") as file:
        pickle.dump(lr_model, file)

print("Your project successfully Done")



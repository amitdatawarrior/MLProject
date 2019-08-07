import stepwiseSelection as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

xls = pd.ExcelFile('casestudy.xlsx')
df1 = pd.read_excel(xls, 'TurbineMasterData')
df2 = pd.read_excel(xls, 'SiteConditions')
df3 = pd.read_excel(xls, 'MaterialCost')
df3=df3.rename(columns = {'DturbineID':'TurbineNumber'})
temp = pd.merge(df1, df2, on='TurbineNumber', how='inner')
df = pd.merge(temp, df3, on='TurbineNumber', how='inner')

X = df.drop(columns= "MaterialCost")
y = df.MaterialCost

final_vars, iterations_logs = ss.backwardSelection(X,y, model_type="linear")

iterations_file = open("Iterations_logs.txt","w+") 
iterations_file.write(iterations_logs)
iterations_file.close()

df =  df[[ 'CurrentAge_Years', 'WindSpeed Stdev', 'Country', 'TurbineType','TurbineVersion', 'MaterialCost' ]]
df = pd.get_dummies(df, columns=['Country', 'TurbineType','TurbineVersion' ], drop_first=True)
df1 = df.drop(columns="MaterialCost")
X = df1.iloc[:,:].values
y = df.iloc[:, 2].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
print("r_square score: ", r2_score(y_test,y_pred))




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False

homestay = pd.read_csv(r'AB_NYC_2019.csv')

homestay.drop(['id','host_name','last_review'], axis=1, inplace=True) # Drop the 'host_name' not only because it is insignificant for analysis but also for ethical reasons. 
homestay.fillna({'reviews_per_month':0},inplace=True)
homestay = homestay[homestay["price"]<=400]
homestay.loc[(homestay.minimum_nights >30),'minimum_nights'] = 30
homestay.drop(['host_id',"latitude",'longitude'], axis=1, inplace=True)

homestay1 = pd.get_dummies(homestay,columns = ['neighbourhood_group','room_type'],drop_first=True)
homestay1.drop(["neighbourhood"], axis=1, inplace=True)
homestay1.drop(['name'], axis=1, inplace=True)
X = homestay1.loc[:,homestay1.columns != 'price']
y = homestay1['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)
linreg = LinearRegression().fit(X_train, y_train)



def LOG(X_test):
    y_pred = linreg.predict(X_test)
    return y_pred
y_pred = linreg.predict(X_test)

df = pd.DataFrame(X_test)

df.to_csv('X_test.csv', index=False)

print(X_test.info())
print(y_pred[0])
print ('RMSE = %.3f'%np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
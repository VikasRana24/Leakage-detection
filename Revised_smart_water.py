#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#import seaborn as sns
#import matplotlib.pyplot as plt
import pickle
data=pd.read_csv('test-new-data.csv')
#data.head()
data.isnull().sum()
data=data.drop(['subtopic'],axis=1)
data.name.value_counts()
new=data['name'].str.split(':',n = 1,expand=True)
new=new[1]
data=data.drop(['name'],axis=1)
data['name']=new
data.publisher.value_counts()
data_n=data.loc[data['name']=='flow-rate'] 
data_tvc=data.loc[data['name']=='total-volume-consumption'] 
a=data_n.value
a1=pd.DataFrame(a)
f_r=a1.rename(columns={'value':'flow-rate'})
b=data_tvc.value
b1=pd.DataFrame(b)
b1=b1.reset_index()
b1=b1.drop(['index'],axis=1)
tvc=b1.rename(columns={'value':'total-volume-consumption'})
new_data=pd.concat([f_r,tvc],axis=1)
new_data=new_data.dropna()
#new_data.to_csv('new_data.csv')
new_data['flow-rate'].value_counts()
date=data['time']
date = date.str.split('+',n=1,expand=True)
new_date=date[0].str.split('T',n=1,expand=True)
data_new=new_date[0]+' '+new_date[1]
import datetime 
# In[6
new_date=pd.to_datetime(data_new, errors='coerce')
new_date=pd.DataFrame(data_new)
new_date=new_date.rename(columns = {0:"time"})
new_date1=new_date[:7086]
final_data=pd.concat([new_data,new_date1],axis=1)
#final_data.shape
final_data['time']=pd.to_datetime(final_data['time'],format='%Y-%m-%d %H:%M:%S')
final_data['time'].dtypes
final_data['hour']=final_data['time'].dt.hour
final_data['day_of_month']=final_data['time'].dt.day
final_data.set_index('time',inplace=True)
final_data.index
final_data.shape
fr=final_data.iloc[:,0:1]
tvc=final_data.iloc[:,0:2]
fr['flow-rate'].value_counts()
tvc=tvc.drop(['flow-rate'],axis=1)
thing_id=pd.DataFrame(data.iloc[:,3])
thing_id=thing_id.rename(columns={'publisher':'thing_id'})
thing_id=thing_id.iloc[:7086]
#thing_id.shape
final_data=final_data.drop(['hour','day_of_month'],axis=1)
final_data=final_data.reset_index()
final_data=pd.concat([final_data,thing_id],axis=1)
final_data['site-id']='3041e48e-bd24-4865-b83b-4a4a827e92e9'
final_data['pipe-dia']=15
final_data['hour']=final_data.time.dt.hour
final_data['day']=final_data.time.dt.day
final_data['abs_val_diff']=final_data['total-volume-consumption']-final_data['total-volume-consumption'].shift(1)
final_data['time_duration(mins)']=((final_data['time']-final_data['time'].shift(1)).dt.seconds)/60
final_data['weekday']=final_data['time'].dt.dayofweek
#Segregating the timings and labeling them.
final_data['Time_slots'] = pd.cut(final_data['hour'], [0,4,8,12,15,18,24], labels=["Late Night","Early Mornings","Morning", "After Noon","Evening","Late Evenings"],right=False)
sum_tvc=final_data.groupby('Time_slots')['abs_val_diff'].sum()
weekday_tvc=final_data.groupby('weekday')['abs_val_diff'].sum()
leakage=pd.read_csv('new_leakage.csv')
leakage=leakage.drop(['Unnamed: 0'],axis=1)
leakage=leakage[leakage['leakage']==1]
leakage=leakage.rename(columns={'flow_rate':'flow-rate','total_volume_consumption':'total-volume-consumption'})
leakage['abs_val_diff']=leakage['total-volume-consumption']-leakage['total-volume-consumption'].shift(1)
leakage=leakage.fillna(0)
final_data1=final_data[['flow-rate','total-volume-consumption','abs_val_diff']]
final_data1.head()
final_data1=final_data1.fillna(0)
final_data1['leakage']=0
f_l=pd.concat([final_data1,leakage],axis=0)
#leakage Distribution
#distribution = (f_l['leakage'].value_counts()*100.0 /len(f_l.index)).plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )                                                                           
#distribution.set_ylabel('leakage',fontsize = 12)
#distribution.set_title('leakage Distribution', fontsize = 12)
# ## Split the data into test and train sets
from sklearn.model_selection import train_test_split 
# Putting feature variable to X
X = f_l.drop(['leakage'], axis=1)
# Putting response variable to y
y = f_l['leakage']
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)
# Scaling
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
model_scaled=standard_scaler.fit(X_train)
X_train_scaled=model_scaled.transform(X_train)
X_test_scaled=model_scaled.transform(X_test)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0))) 
# ## Smote Analysis
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train_scaled, y_train.ravel()) 
dataa=pd.DataFrame(y_train_res)
dataa['leakage']=dataa[0]
dataa=dataa.drop([0],axis=1)
dataa.head()
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))   
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 
#leakage Distribution
#distribution = (dataa['leakage'].value_counts()*100.0 /len(dataa.index)).plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), fontsize = 12 )                                                                           
#distribution.set_ylabel('leakage',fontsize = 12)
#distribution.set_title('leakage Distribution', fontsize = 12)

# ## Model building
from sklearn.svm import SVC
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import confusion_matrix, classification_report 
# Model building
# instantiate an object of class SVC()
# note that we are using cost C=1
model_new = SVC(C=1.0,kernel='rbf', random_state = 0,break_ties=True)
# fit
model_new.fit(X_train_res, y_train_res.ravel())
# predict
y_pred = model_new.predict(X_test_scaled)
#print(classification_report(y_test, y_pred)) 
#from sklearn import metrics
#metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
y_pred_train=model_new.predict(X_train_scaled)
#print(y_pred)
#print(classification_report(y_train, y_pred_train)) 
#from sklearn import metrics
#metrics.confusion_matrix(y_true=y_train, y_pred=y_pred_train)
pickle.dump(model_new, open('model_new.pkl','wb'))
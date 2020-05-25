# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:41:58 2020

@author: techolution
"""

import numpy as np
import pandas as pd
from flask import Flask,request #,jsonify
import pickle
#import flasgger
from flasgger import Swagger
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()


app = Flask(__name__)
Swagger(app)
#load the data

new_data=pd.read_csv('C:/Users/techolution/Desktop/Smart City/X.csv')
new_data=new_data.drop(['Unnamed: 0'],axis=1)
scaled=standard_scaler.fit(new_data)

#Load the model 

model_new=pickle.load(open('model_new.pkl','rb'))

#add the logging feature
#logger=logging.getLogger('logger')
#logging.basicConfig(filename="water_leak.log", level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s", filemode="a+")


@app.route('/')
def first_route():
#       app.logger.info('find get route')
       return 'find get route'

@app.route('/get_route',methods=['GET'])
def get_route():
      if request.method=='GET':
#             app.logger.info('created first get route')
             return 'Created first get route'

@app.route('/predictLeak1',methods=['GET'])
def predict_leak():
    """Lets Predict Leak on the fly
    Leakage Prediction 
    ---
    paramters:
        - name :flow_rate
          in: query
          type: number
          required: true
          
        - name:total_volume_consumption
          in: query
          type: number
          required: true
       
    responses:
        200:
            description: The Leakage is  
    """
    flow_rate = request.args.get('flow_rate')
    total_volume_consumption = request.args.get('total_volume_consumption')
    df_arr=np.array([[flow_rate,total_volume_consumption]]).astype(np.float64)
    df_test=pd.DataFrame(df_arr)
    df_test['abs_val_diff']=df_test['total_volume_consumption']-df_test['total_volume_consumption'].shift(1)
    #time=df_test['time'][1:]
    #time=time.reset_index()
    #    time=time.drop(['index'],axis=1)
    #    df_test=df_test.drop(['time'],axis=1)
    #    df_test=df_test[1:]
    df_scaled=scaled.transform(df_test)
    #print(df_scaled.head())
    prediction=model_new.predict(df_scaled)
    return str(list(prediction))


if __name__ == '__main__':
       app.run(port=5000)
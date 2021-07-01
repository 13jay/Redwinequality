from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# import pandas as pd
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def Home():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
     ''''fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
     'pH', 'sulphates', 'alcohol'''
     
     fixed_acidity = float(request.form['Fixed_acidity'])
     volatile_acidity = float(request.form['Volatile_acidity'])
     citric_acid = float(request.form['Citric_acid'])
     residual_sugar = float(request.form['Residual_sugar'])
     chlorides = float(request.form['Chlorides'])
     free_sulphur_dioxide  = float(request.form['Free_sulphur_dioxide'])
     total_sulphur_dioxide = float(request.form['Total_sulphur_dioxide'])
     density = float(request.form['Density'])
     pH = float(request.form['pH'])
     sulphates = float(request.form['Sulphates'])
     alcohol = float(request.form['Alcohol'])

     data = np.array([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulphur_dioxide,total_sulphur_dioxide,density,pH,sulphates,alcohol])

#Object that scales the data before predicting, data cleaning ibject from ML notebook
     print(data,'***************************** Data recevied')
     sc = open('scaler_new.pickle','rb')
     sc = pickle.load(sc)
     data = sc.transform(data.reshape(1, -1))
     print(data,'Scaled data output')


     rfc = open('redwine.pickle','rb')
     rfc = pickle.load(rfc)
 
     prediction = rfc.predict(data)
     # output = round(prediction[0],2)
     output = prediction
     
     return render_template('index.html',prediction_text="the quality of the wine is {}".format('GOOD' if output[0] != 0 else 'BAD'))
     

if __name__=="__main__":
    app.run(debug=True)


from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os
import sklearn


app=Flask(__name__)
if os.path.getsize("LinearRegressionModel.pkl") > 0:
    with open("LinearRegressionModel.pkl", "rb") as file:
        unpickler = pickle.Unpickler(file)
        model = unpickler.load()

car = pd.read_csv("Cleaned_Car_data.csv")

@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0,"Select Company")
    return render_template('index.html', companies=companies, car_models=car_model, years=year, fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([car_model,company,year,driven,fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)
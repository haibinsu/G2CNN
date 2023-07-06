"""
Application that predicts heart disease percentage in the population of a town
based on the number of bikers and smokers. 

Trained on the data set of percentage of people biking 
to work each day, the percentage of people smoking, and the percentage of 
people with heart disease in an imaginary sample of 500 towns.

"""

import numpy as np
from flask import Flask, request, render_template, flash
import pickle
import pandas as pd

#Create an app object using the Flask class. 
app = Flask(__name__)
#app.secret_key = "cqygfxgfsthlhg"

#Load the trained model. (Pickle file)
#model = pickle.load(open('models/model.pkl', 'rb'))

vispils=pd.read_csv("vispils.csv")


#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, home function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 
#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')


#You can use the methods argument of the route() decorator to handle different HTTP methods.
#GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to the decorator to allow for form submission. 
#Redirect to /predict page with the output

#@app.route('/predict',methods=['POST'])
#def predict():
#
#    int_features = [float(x) for x in request.form.values()] #Convert string inputs to float.
#    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
#    prediction = model.predict(features)  # features Must be in the form [[a, b]]
#
#    output = round(prediction[0], 2)
#
#    return render_template('index.html', prediction_text='Percent with heart disease is {}'.format(output))


@app.route('/predict',methods=['POST'])
def predict():
    tol=0.15

    int_smis=request.form['SMILES_input']
    int_temp=float(request.form['Temperature_input'])

    row=vispils[(vispils['Iso SMILES']==int_smis) & (vispils['Temperature'].between(int_temp-tol, int_temp+tol))]

    properties=row[['Iso SMILES', 'Log viscosity', 'Temperature', 'DataSource', 'Reliability']]
    output = properties.values[0]
    output[1]=f'{10**output[1]:.2f}'

    pred_vars={"output": output,
               "prediction_text":'η = {:.2f} mPas at {} K'.format(float(output[1]), output[2]),
               }
    
    return render_template('index.html', **pred_vars)

if __name__ == "__main__":
    app.run(debug=True)
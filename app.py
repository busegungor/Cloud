from flask import Flask, render_template, request, jsonify
from flask_restful import Api, Resource
import pickle
import numpy as np
import sklearn
import sys
import os 


model = pickle.load(open('fish.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3 ]])
    pred = model.predict(arr)
    return render_template('after.html', data = pred)

if __name__ == "__main__":
    app.run(debug = True)
    
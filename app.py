import pickle
from flask import Flask, request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            G2=int(request.form.get('G2')),
            G1=int(request.form.get('G1')),
            Medu=int(request.form.get('Medu')),
            higher=request.form.get('higher'),
            paid=request.form.get('paid'),
            studytime=int(request.form.get('studytime')),
            Fedu=int(request.form.get('Fedu')),
            internet=request.form.get('internet'),
            goout=int(request.form.get('goout')),
            traveltime=int(request.form.get('traveltime')),
            romantic=request.form.get('romantic'),
            age=int(request.form.get('age')),
            failures=int(request.form.get('failures'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=round((results[0] / 20) * 100) )
        
if __name__ == "__main__":
    app.run(port=5001, debug = True)
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Reasons_diseases=int(request.form.get('Reasons_diseases')),
            Reasons_pregnancy=int(request.form.get('Reasons_pregnancy')),
            Reasons_health_symptomps=int(request.form.get('Reasons_health_symptomps')),
            Reasons_light=int(request.form.get('Reasons_light')),
            Month_Value=int(request.form.get('Month_Value')),
            Day_of_the_week=int(request.form.get('Day_of_the_week')),
            Transportation_Expense=float(request.form.get('Transportation_Expense')),
            Distance_to_Work=float(request.form.get('Distance_to_Work')),
            Age=int(request.form.get('Age')),
            Daily_Work_Load_Average=float(request.form.get('Daily_Work_Load_Average')),
            Body_Mass_Index=float(request.form.get('Body_Mass_Index')),
            Education=int(request.form.get('Education')),
            Children=int(request.form.get('Children')),
            Pets=int(request.form.get('Pets'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

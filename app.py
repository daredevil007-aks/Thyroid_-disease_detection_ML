from crypt import methods
from flask import Flask, request,render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods['GET', 'POST'])

def predict_datapoint():
    if request.method=="GET":
        return render_template('form.html')


    else:
        data=CustomData(
            
        )
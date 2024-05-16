import logging
from webbrowser import get
from flask import Flask, request,render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict_datapoint():
    if request.method=="GET":
        return render_template('form.html')

    else:
        data=CustomData(
           age = float(request.form.get('age')),
           sex = request.form.get('sex'),
           on_thyroxine = request.form.get('on_thyroxine'),
           query_on_thyroxine = request.form.get('query_on_thyroxine'),
           on_antithyroid_medication = request.form.get('on_antithyroid_medication'),
           sick = request.form.get('sick'),
           pregnant = request.form.get('pregnant'),
           thyroid_surgery = request.form.get('thyroid_surgery'),
           I131_treatment = request.form.get('I131_treatment'),
           query_hypothyroid = request.form.get('query_hypothyroid'),
           query_hyperthyroid = request.form.get('query_hyperthyroid'),
           lithium = request.form.get('lithium'),
           goitre = request.form.get('goitre'),
           tumor = request.form.get('tumor'),
           hypopituitary = request.form.get('hypopituitary'),
           psych = request.form.get('psych'),
           TSH_measured = request.form.get('TSH_measured'),
           TSH = float(request.form.get('TSH')),
           T3_measured = request.form.get('T3_measured'),
           T3 = float(request.form.get('T3')),
           TT4_measured = request.form.get('TT4_measured'),
           TT4 = float(request.form.get('TT4')),
           T4U_measured = request.form.get('T4U_measured'),
           T4U = float(request.form.get('T4U')),
           FTI_measured = request.form.get('FTI_measured'),
           FTI = float(request.form.get('FTI')),
           TBG_measured = request.form.get('TBG_measured'),
           referral_source = request.form.get('referral_source')

        )
        final_new_data = data.get_data_as_dataframe()

        logging.info(f'Test Dataframe Head: \n{final_new_data.head().to_string()}')
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = pred

        return render_template('result.html',final_result=results)


if __name__=='__main__':
    app.run(host='0.0.0.0')
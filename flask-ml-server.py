import os
from flask import Flask
from flask import request, render_template
from flask import send_from_directory

from joblib import load
import numpy as np

app = Flask(__name__)

#TODO: load iris classfier from file
#      use try-except, print message and exit if there is a problem
try:
    iris_classifier = load('iris_logreg.joblib')
    print(iris_classifier)
except:
    print("model failed to load")
@app.route('/')
def index():
    return render_template('prediction_input.html')

#GET REQUEST
@app.route('/iris_prediction')
def get_iris_prediction():
    
    values_ok = True
    pred_str = 'None'
    pred_proba = 0.0
    
    #TODO: Get feature values as float from request.values dictionary
    #      Set values_ok to False if any conversion produces an error.
    try:
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))
    except:
        values_ok = False

    
    #TODO: call predict() on the loaded classifier using the feature values
    #      and retrieve the predicted iris flower string
    #      assign string to pred_str

    # Not necessary if predict_proba() is being used. See below. 
    # NOTE: added class "target_names" to model from joblib dump in jupyter notebook. This makes the original target names accessible and therefore adds flexibility. If a new 'type' were to be added to the original model, this new type would carry over.

    #TODO: call predict_proba() on the loaded classifier
    #      assign probablity to pred_proba
    if values_ok == True:
        probabilities = iris_classifier.predict_proba([[sepal_length, sepal_width, petal_length, petal_width]])
        
        pred_proba = max(probabilities[0])
        pred_str = iris_classifier.target_names[np.argmax(probabilities, axis=1)][0].title()

    return render_template('prediction_response.html',
                           request_dict=request.values,
                          pred_str=pred_str,
                           pred_proba='{:.3f}'.format(pred_proba),
                          values_ok=values_ok) 

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0")

import pickle
from flask import Flask, request, app, jsonify, render_template
import numpy as np 
import pandas as pd 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')


@app.route('/predict_app', methods=['POST'])
def predict_api():
    # data is a dictionary  
    """
    {
        "data": {
            "Frequency":9,
            "Angle of attack" : 1,
            "Chord length" : 1,
            "Free-stream velocity": 1,
            "Suction side":1
        }
    }
    """
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)



@app.route('/predict',methods=['POST'])
def predict():

    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

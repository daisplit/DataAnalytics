from flask import Flask, request
import os
from backend.app import Modelling

from flask_cors import CORS

import numpy as np
import pandas as pd
OBJECT_MODEL = Modelling()

app = Flask(__name__)
app.secret_key = 'XXX'
CORS(app)
API_BASE = "/api"
@app.route(API_BASE + "/forecast", methods=['GET', 'POST'])
def forecast():
    model_info = request.get_json()
    raw_data = model_info["data"]
    series = pd.Series(eval(raw_data.split()[0]))
    data = pd.DataFrame([series]).T.reset_index()
    print(data)
    data.columns = ['index','expenses']
    print(data)
    output = OBJECT_MODEL.forecast(data)
    result_json = output.to_json(orient='records')
    return result_json

if __name__ == '__main__':
    app.run(debug=False)

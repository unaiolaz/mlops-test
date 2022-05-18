import json
import numpy as np
import joblib
import pandas as pd
from training import prep_data
def init():
    global model
    model_path = "../models/random_forest_prueba.pkl"
    model = joblib.load(model_path)
    
def run(raw_data, request_headers):
    data = prep_data(raw_data)
    data = np.array(data)
    result = model.predict(data)
    return {"result": result.tolist()}
init()
test_row = pd.read_csv("../data/test.csv")
request_header = {}
prediction = run(test_row, {})
print("Test result: ", prediction)
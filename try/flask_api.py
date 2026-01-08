from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("logreg.pkl","rb")
model=pickle.load(pickle_in)


@app.route('/predict',methods=["Get"])
def predict_class():
    age=int(request.args.get("age"))
    new_user=int(request.args.get("new_user"))
    total_pages_visited=int(request.args.get("total_pages_visited"))
    prediction=model.predict([[age,new_user,total_pages_visited]])
    print(prediction[0])
    return "Model prediction is"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def prediction_test_file():

    df_test=pd.read_csv(request.files.get("file"))
    prediction=model.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0:5000')
    
    
# http://localhost:5000/apidocs/
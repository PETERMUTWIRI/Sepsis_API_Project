# from fastapi import FastAPI,Form, Body,Path
# from typing import Annotated
# from pydantic import BaseModel, Field
# import joblib
# import pandas as pd
# import numpy as np
# import uvicorn
# from fastapi.responses import JSONResponse


# app = FastAPI()

# # Load the numerical imputer, scaler, and model
# num_imputer_filepath = "joblib_files/numerical_imputer.joblib"
# scaler_filepath = "joblib_files/scaler.joblib"
# model_filepath = "joblib_files/lr_model.joblib"

# num_imputer = joblib.load(num_imputer_filepath)
# scaler = joblib.load(scaler_filepath)
# model = joblib.load(model_filepath)

# class PatientData(BaseModel):
#     PRG: float 
#     PL: float
#     PR: float
#     SK: float
#     TS: float
#     M11: float
#     BD2: float
#     Age: float
#     Insurance: int

# def preprocess_input_data(user_input):
#     input_data_df = pd.DataFrame([user_input])
#     num_columns = [col for col in input_data_df.columns if input_data_df[col].dtype != 'object']
#     input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
#     input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)
#     return input_scaled_df

# @app.get("/")
# def read_root():
#         return "Sepsis Prediction App"
# @app.post("/sepsis/predict")
# def get_data_from_user(data:PatientData):
#     user_input = data.dict()
#     input_scaled_df = preprocess_input_data(user_input)
#     probabilities = model.predict_proba(input_scaled_df)[0]
#     prediction = np.argmax(probabilities)

#     sepsis_status = "Positive" if prediction == 1 else "Negative"
#     probability = probabilities[1] if prediction == 1 else probabilities[0]

#     if prediction == 1:
#         sepsis_explanation = "A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention."
#     else:
#         sepsis_explanation = "A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

#     statement = f"The patient's sepsis status is {sepsis_status} with a probability of {probability:.2f}. {sepsis_explanation}"

#     user_input_statement = "user-inputted data: "
#     output_df = pd.DataFrame([user_input])

#     result = {'predicted_sepsis': sepsis_status, 'statement': statement, 'user_input_statement': user_input_statement, 'input_data_df': output_df.to_dict('records')}
#     return result

# from fastapi import FastAPI, Form
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import numpy as np
# import uvicorn
# from fastapi.responses import JSONResponse

# app = FastAPI()

# # Load the entire pipeline
# pipeline_filepath = "pipeline.joblib"
# pipeline = joblib.load(pipeline_filepath)

# class PatientData(BaseModel):
#     PRG: float
#     PL: float
#     PR: float
#     SK: float
#     TS: float
#     M11: float
#     BD2: float
#     Age: float
#     Insurance: int

# @app.get("/")
# def read_root():
#     explanation = {
#         'message': "Welcome to the Sepsis Prediction App",
#         'description': "This API allows you to predict sepsis based on patient data.",
#         'usage': "Submit a POST request to /predict with patient data to make predictions.",
#         'input_fields': {
#             'PRG': 'Plasma_glucose',
#             'PL': 'Blood_Work_Result_1',
#             'PR': 'Blood_Pressure',
#             'SK': 'Blood_Work_Result_2',
#             'TS': 'Blood_Work_Result_3',
#             'M11': 'Body_mass_index',
#             'BD2': 'Blood_Work_Result_4',
#             'Insurance': 'Sepsis (Positive = 1, Negative = 0)'
#         }
#     }
#     return explanation


# @app.post("/predict")
# def get_data_from_user(data: PatientData):
#     user_input = data.model_dump()

    
#     input_df = pd.DataFrame([user_input])
#      # Make predictions using the loaded pipeline
#     # Make predictions using the loaded pipeline
#     predictions = pipeline.predict(user_input)
#     probabilities = pipeline.decision_function(user_input)

#     # Assuming the pipeline uses a Logistic Regression model
#     probability_of_positive_class = probabilities[0]

#      # Calculate the prediction
#     prediction = 1 if probability_of_positive_class >= 0.5 else 0

#     sepsis_status = "Positive" if prediction == 1 else "Negative"
#     sepsis_explanation = "A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention." if prediction == 1 else "A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

   
#     if prediction == 1:
#         sepsis_status = "Positive"
#         sepsis_explanation = "A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention."
#     else:
#         sepsis_status = "Negative"
#         sepsis_explanation = "A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

#     result = {
#         'predicted_sepsis': sepsis_status,
#         'sepsis_explanation': sepsis_explanation
#     }
#     return result

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Load the entire pipeline
pipeline_filepath = "pipeline.joblib"
pipeline = joblib.load(pipeline_filepath)

class PatientData(BaseModel):
    Plasma_glucose : float
    Blood_Work_Result_1: float
    Blood_Pressure : float
    Blood_Work_Result_2 : float
    Blood_Work_Result_3 : float
    Body_mass_index  : float
    Blood_Work_Result_4: float
    Age: float
    Insurance: int

@app.get("/")
def read_root():
    explanation = {
        'message': "Welcome to the Sepsis Prediction App",
        'description': "This API allows you to predict sepsis based on patient data.",
        'usage': "Submit a POST request to /predict with patient data to make predictions.",
        
    }
    return explanation

@app.post("/predict")
def get_data_from_user(data: PatientData):
    user_input = data.dict()

    input_df = pd.DataFrame([user_input])

    # Make predictions using the loaded pipeline
    prediction = pipeline.predict(input_df)
    probabilities = pipeline.predict_proba(input_df)

    
    probability_of_positive_class = probabilities[0][1]

    # Calculate the prediction
    sepsis_status = "Positive" if prediction[0] == 1 else "Negative"
    sepsis_explanation = "A positive prediction suggests that the patient might be exhibiting sepsis symptoms and requires immediate medical attention." if prediction[0] == 1 else "A negative prediction suggests that the patient is not currently exhibiting sepsis symptoms."

    result = {
        'predicted_sepsis': sepsis_status,
        'probability': probability_of_positive_class,
        'sepsis_explanation': sepsis_explanation
    }
    return result

import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)


with open('pipeline.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions

#test missing observation_id and data
def check_request(request):
    try:
        request['observation_id']
    except:
        error = "No observation_id found"
        return False, error 
    try:
        request["data"]
    except:
        error = 'No data found in observation'
    return True, ""

#test missing columns and extra columns
def check_columns(observation):
    keys = set(observation.keys())
    if len(columns) - len(keys) > 0: 
        missing = set(columns) - keys
        error = "Missing columns: {}".format(missing)
        return False, error

    if len(keys) - len(columns) > 0: 
        missing = keys - set(columns)
        error = "Not expected columns: {}".format(missing)
        return False, error

    return True, ""
    
#categories: A list of potential values for column
def get_valid_categories(df, column):
    categories = df[column].unique().tolist()    

    return categories

#test invalid values for categorical features 
def check_categorical_features(observation, df, categorical_features):
    
    for cat in categorical_features:
        valid_cats = get_valid_categories(df, cat)
        if observation[cat] not in valid_cats:
            error ="{} is not a valid value for {} category".format(observation[cat],cat)
            return False, error
    return True, ""

#test invalid values for numerical features 
def check_numerical_features(observation, df, numerical_features):
    for num_cat in numerical_features:
        valid_cats = get_valid_categories(df, num_cat)
        if observation[num_cat] not in valid_cats:
            error ="{} is not a valid value for {} category".format(observation[num_cat], num_cat)
            return False, error
    return True, ""
    

# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)
df = pd.read_csv("bank.csv")
numerical_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = [ 'workclass', 'education', 'marital-status', 'race', 'sex']


@app.route('/predict', methods=['POST'])
def predict():
    
    obs_dict = request.get_json()

    #check observations
    request_ok, error = check_request(obs_dict)
    if not request_ok:
        response = {'error': error}
        return jsonify(response)
    _id = obs_dict['observation_id']
    observation = obs_dict['data']

    columns_ok, error = check_columns(observation)
    if not columns_ok:
        response = {'error': error}
        return jsonify(response)

    categories_ok, error = check_categorical_features(observation, valid_cats, categorical_features)
    if not categories_ok:
        response = {'error': error}
        return jsonify(response)

    numerical_features_ok, error = check_numerical_features(observation, valid_cats, numerical_features)
    if not numerical_features_ok:
        response = {'error': error}
        return jsonify(response)

    df_request = pd.DataFrame([observation], columns = columns).astype(dtypes) 
    proba = pipeline.predict_proba(df_request)[0, 1]
    prediction = pipeline.predict(df_request)[0]
    
    response = {'prediction': bool(prediction), 'proba': proba}

    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)

    
    
if __name__ == "__main__":
    app.run()

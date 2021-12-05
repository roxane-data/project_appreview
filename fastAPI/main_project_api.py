"""

Q1. Install FastAPI.

Q2. Create a project folder (as a module containing an empty file called __init__.py) named california_housing_api, your api code will be contained inside main.py.

Q3. Create a post method for the URL newcaliforniareg for creating a new trained
regression model. On the parameters sent, indicate the proportion of data to be used
for the training and the alpha for Ridge to be used. Remember you have to save
this model using pickle.

Q4. Create a post method to predict a data entry for a "California Housing" by
specifying which trained model to use.
[Optional] - Going further

You can refer to the last, optional part of today's lesson in order to complete the following questions.

Let's add some functionnalities to our API :

    We can update the name of a trained model.
    We can delete a trained model.
    We'll be using Pymongo to store our data.

Q5. Create a get method to list all trained models created.

Q6. Create a get method to have information of one trained model using its ID.

Q7. Create a patch method to update the name of an existing trained model. This means we will update the MongoDB collection using our interface.

Q8. Create a post method to predict a data entry for a "California Housing" by specifying which trained model to use.


"""
import pickle
import uuid
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import spacy
nlp = spacy.load("en_core_web_md")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

MODELS = {
    "lr": {
        "model": LogisticRegression,
        "name": "Logistic Regression",
        "api_model_code": "lr",
        "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    }
}

app = FastAPI()

class ModelTrainIn(BaseModel):
    name: str
    api_model_code: str
    documentation: str = None

class ModelTrainOut(BaseModel):
    train_id: str = None
    api_model_code: str = None
    trained_model_name: str = None
    acc: float = None

class ModelPredictIn(BaseModel):
    train_id: str
    trained_model_name: str = None
    feedback: str

class ModelPredictOut(BaseModel):
    predicted_rating: int = None
    predicted_comment_cat: str = None
    train_id: str


@app.post("/model/train/", response_model=ModelTrainOut) # here we're defining the data model for our response
async def train_model(model_train: ModelTrainIn): # this defines the data model for our request
    # we transform our data model object into a dictionary
    model_train_dict = model_train.dict()

    # we initialize our ML model
    model = MODELS[model_train_dict['api_model_code']]['model']() # we initialize our ML model

    # load and split data
    data = data
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

    """ vectorize after or before the split and separate df between positive, neutral or negative comments """

    # fit model and get a prediction and a score
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # we define a unique ID
    unique_id = uuid.uuid4().hex

    # saving the model locally
    filename = f"trained_models/{unique_id}.sav"
    pickle.dump(model, open(filename, 'wb'))
    model_train_dict.update({"train_id": unique_id, "accuracy": accuracy})

    # return our response that has the same data structure as ModelTrainOut
    return model_train_dict
    """
    model_train_dict = model_train.dict()
    test_size = model_train_dict['proportion_data']
    model = MODELS[model_train_dict['api_model_code']]['model'](alpha = alpha)
    data = fetch_california_housing()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    unique_id = uuid.uuid4().hex
    filename = f"trained_models/{unique_id}.sav"
    pickle.dump(model, open(filename, 'wb'))
    model_train_dict.update({"train_id": unique_id, "mse": mse})
    return model_train_dict
"""
@app.post("/model/predict/", response_model=ModelPredictOut)
async def predict_rating(review_data: ModelPredictIn):

    review_data_dict = review_data.dict()
    train_id = review_data_dict['train_id']
    model = pickle.load(open(f"trained_models/{train_id}.sav", 'rb'))
    """
    data = [[
        price_data_dict['MedInc'],
        price_data_dict['HouseAge'],
        price_data_dict['AveRooms'],
        price_data_dict['AveBedrms'],
        price_data_dict['Population'],
        price_data_dict['AveOccup'],
        price_data_dict['Latitude'],
        price_data_dict['Longitude']
    ]]
    """
    pred = model.predict(data)

    return {
        "predicted_rating": pred,
        "train_id": train_id
    }

import pickle
from flask import Flask, render_template, request
import requests

app = Flask(__name__)


@app.route('/', methods=['GET'])
def rating_index():
    return render_template('index.html')

@app.route('/dataviz/', methods=['GET'])
def viz_index():
    return render_template('lda1.html')

@app.route('/predict/', methods=['POST'])
def result():
    review = str(request.form['review'])


    r = requests.post('http://127.0.0.1:8000/model/predict/', json={
    """ train_id to modify """

        "train_id": "2f80d339e120436b8f5b9640fd750578",
        "review": review
    })

    prediction_response = r.json()    #model = pickle.load(open(f"model.sav", 'rb'))


    return render_template("prediction.html", rating=prediction_response['rating'])


if __name__ == '__main__':
    app.debug = True
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True)

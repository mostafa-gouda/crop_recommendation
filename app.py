from textwrap import indent
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv('Crop_recommendation.csv')
app = Flask(__name__)

mean = [
    50.551818181818184, 53.36272727272727, 48.14909090909091,
    25.616243851779533, 71.48177921778648, 6.469480065256369,
    103.46365541576832,
]

std = [
    36.91733383375668, 32.985882738587144, 50.64793054666006, 5.063748599958843,
    22.263811589761115, 0.7739376880298721, 54.95838852487811,
]


@app.route('/', methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route('/', methods=["POST"])
def predict():
    feature_values = []
    for i in range(1, 8):
        feature_value = float(request.form[f"feature{i}"])
        scaled_feature_value = (feature_value-mean[i-1])/std[i-1]
        feature_values.append(scaled_feature_value)

    feature_values = tf.constant([feature_values])
    model = tf.keras.models.load_model("my_model/")
    predictions = model.predict(feature_values)
    answer = df['label'].unique()[np.argmax(predictions[0])]
    return render_template("index.html", prediction=answer)


if __name__ == '__main__':
    app.run(port=4000, debug=True)

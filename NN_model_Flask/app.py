import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request


app = Flask(__name__, template_folder='template')


@app.route('/')
def main():
    return render_template('main.html')


def predict_value(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 11)
    loaded_model = tf.keras.models.load_model('NN_model_Flask/saved_model.h5')
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predict',methods = ['POST'])
def submit_prediction():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = predict_value(to_predict_list)
        prediction = str(result)
        return render_template('predict.html',prediction=prediction)
    

if __name__ == '__main__':
    app.run(debug=True)
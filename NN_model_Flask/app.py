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


#def mn_prediction(params):
    #model = tf.keras.models.load_model('NN_model_Flask/saved_model.h5')
    #pred = model.predict([params])
    #return pred


@app.route('/predict',methods = ['POST'])
def submit_prediction():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = predict_value(to_predict_list)
        prediction = str(result)
        return render_template('predict.html',prediction=prediction)
    

#def mn_predict():
    #message = ''
    #if request.method == 'POST':
        #param_list = ('plot', 'mup', 'ko', 'seg', 'tv', 'pp', 'mup', 'pr', 'ps', 'yn', 'shn', 'pln')
        #params = []
        #for i in param_list:
            #param = request.form.get(i)
            #params.append(param)
        #params = [float(i.replace(',', '.')) for i in params]

        #message = f'Спрогнозированное Соотношение матрица-наполнитель для введенных параметров: {mn_prediction(params)}'
    #return render_template('mn.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__) 
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():  

    petal_length = request.form['petal_length']
    sepal_length = request.form['sepal_length']
    petal_width = request.form['petal_width']
    sepal_width = request.form['sepal_width']

    input_data = [sepal_length,sepal_width,petal_length,petal_width]
    data = [float(i) for i in input_data]
    model_input = np.array(data).reshape(1,-1)
    pred = model.predict(model_input)[0]
    
    idx = np.argmax(pred)
    if idx == 0:
        return (f"Iris setosa with probability {100*np.round(np.amax(pred),3)}%")
    elif idx == 1:
        return (f"Iris virginica with probability {100*np.round(np.amax(pred),3)}%")
    else:
        return (f"Iris versicolor with probability {100*np.round(np.amax(pred),3)}%")

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
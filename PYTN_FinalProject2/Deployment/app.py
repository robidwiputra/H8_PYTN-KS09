from flask import Flask, render_template, request
import numpy as np
import pickle

# model = pickle.load(open('model_svm.pkl', 'rb'))
model = pickle.load(open('model/model_logreg.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)


    output = {0.0:'not rain tomorrow', 1.0:'rain tomorrow'}

    return render_template('index.html', prediction_text='It will {}'.format(output[prediction[0]]))

if __name__ == '__main__':
    app.run(debug=True)


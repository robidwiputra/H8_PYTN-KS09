from distutils.log import debug
import flask
import numpy as np
import pickle

app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/rf_kmeans.pkl', 'rb'))

@app.route('/')
def main():
    return(flask.render_template('main.html'))
    
@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in flask.request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)


    output = {0.0:'Medium Spenders', 1.0:'Pay Later Purchasers', 2.0:'Big Spenders', 3.0:'Frugal Consumers'}
    
    return flask.render_template('main.html', prediction_text='{} Segmentation'.format(output[prediction[0]]))

if __name__ == '__main__':
    app.run(debug=True)
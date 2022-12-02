from distutils.log import debug
import flask
import numpy as np
import pickle

app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/model_rfs.pkl', 'rb'))

@app.route('/')
def main():
    return(flask.render_template('main.html'))
    
@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in flask.request.form.values()]
    final_features = [np.array(float_features)]
    print(final_features)
    prediction = model.predict(final_features)
    if(prediction == 1):
        return flask.render_template('main.html', prediction_text='Your probability to get heart attack are high.')
    else:
        return flask.render_template('main.html', prediction_text='Your probability to get heart attack are low.')

if __name__ == "__main__":
    app.run(debug=True)
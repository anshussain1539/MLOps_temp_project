from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('iris_model.joblib')
scaler = joblib.load('iris_scaler.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Normalize input features
        input_features = scaler.transform(
            [[sepal_length, sepal_width, petal_length, petal_width]])

        # Make prediction
        prediction = model.predict(input_features)[0]

        return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

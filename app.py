# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.joblib')

# Define the feature names
feature_names = ['shop_id', 'item_id',
                 'date_block_num', 'item_category_id', 'Month', 'Year']


@app.route('/')
def index():
    # Pass the feature names to the template
    return render_template('index.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # # Assuming the form input names match the feature names
        # features = [float(request.form[name]) for name in feature_names]
        # input_data = pd.DataFrame([features], columns=feature_names)
        # prediction = model.predict(input_data)

        shop_id = int(request.form['shop_id'])
        item_id = int(request.form['item_id'])
        date_block_num = int(request.form['date_block_num'])
        item_category_id = int(request.form['item_category_id'])
        Month = int(request.form['Month'])
        Year = int(request.form['Year'])
        input_features = np.array([
            shop_id, item_id, date_block_num, item_category_id, Month, Year])
        # Make prediction
        input_features = input_features.reshape(1,-1)
        prediction = model.predict(input_features)
        return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

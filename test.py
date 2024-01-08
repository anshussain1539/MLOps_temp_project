# import pandas as pd
# from sklearn.metrics import accuracy_score
# import joblib

# # Load the trained model and scaler
# model = joblib.load('iris_model.joblib')
# scaler = joblib.load('iris_scaler.joblib')

# # Load the test data
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
# df_test = pd.read_csv(url, names=column_names)

# # Separate features and labels
# X_test = df_test.drop("class", axis=1)
# y_test = df_test["class"]

# # Normalize test features
# X_test_scaled = scaler.transform(X_test)

# # Make predictions
# predictions = model.predict(X_test_scaled)

# # Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
def test1():
    assert True


def test2():
    assert True

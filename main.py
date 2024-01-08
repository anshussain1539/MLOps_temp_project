import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width",
                "petal_length", "petal_width", "class"]

df = pd.read_csv(url, names=column_names)

# Split data into features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Split the data into training and testing sets
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple Support Vector Machine (SVM) model
model = SVC()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'iris_model.joblib')
joblib.dump(scaler, 'iris_scaler.joblib')

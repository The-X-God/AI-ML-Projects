
# 🌸 Iris Flower Classification
# Created by The-X-God

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("🌸 IRIS FLOWER CLASSIFICATION")
print("--------------------------------")
print("Model Training Completed!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction[0]])

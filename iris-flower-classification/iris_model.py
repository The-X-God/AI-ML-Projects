
# 🌸 Iris Flower Classification V2
# Created by The-X-God

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n🌸 IRIS FLOWER CLASSIFICATION")
print("--------------------------------")
print("Model Training Completed!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\n📊 CLASSIFICATION REPORT")
print(classification_report(
    y_test, predictions, target_names=iris.target_names
))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nCONFUSION MATRIX")
print(cm)

# Confusion Matrix Visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
display.plot(cmap="Blues")
plt.title("Iris Flower Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Accuracy Visualization
plt.figure(figsize=(6, 4))
plt.bar(["Random Forest"], [accuracy * 100])
plt.ylabel("Accuracy (%)")
plt.title("Iris Model Accuracy")
plt.ylim(0, 100)
plt.savefig("accuracy_graph.png")
plt.show()

# Predict New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("\n🌺 NEW FLOWER PREDICTION")
print("Predicted Flower:", iris.target_names[prediction[0]])


# 🔢 Handwritten Digit Recognition
# Created by The-X-God

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load Dataset
digits = load_digits()

X = digits.data
y = digits.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Make Predictions
predictions = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n🔢 HANDWRITTEN DIGIT RECOGNITION")
print("--------------------------------")
print("Model Training Completed!")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\n📊 CLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

# Predict Multiple Sample Digits

print("\n🔢 MULTIPLE DIGIT PREDICTIONS")
print("--------------------------------")

for i in range(5):

    sample = X_test[i].reshape(1, -1)

    prediction = model.predict(sample)

    print("Sample", i + 1)
    print("Predicted Digit:", prediction[0])
    print("Actual Digit:", y_test[i])

    if prediction[0] == y_test[i]:
        print("Result: Correct ✅")
    else:
        print("Result: Incorrect ❌")

    print("--------------------------------")

plt.title(f"Predicted Digit: {prediction[0]}")
plt.axis("off")
plt.savefig("digit_prediction.png")
plt.show()

# Accuracy Graph
plt.figure(figsize=(6, 4))
plt.bar(["Random Forest"], [accuracy * 100])
plt.ylabel("Accuracy (%)")
plt.title("Digit Recognition Accuracy")
plt.ylim(0, 100)

plt.savefig("accuracy_graph.png")
plt.show()

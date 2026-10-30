# AI-Based-Voltage-Prediction-System
AI-based system to monitor voltage readings and detect normal, overvoltage, and undervoltage conditions using Machine Learning.
# AI-Based Overvoltage and Undervoltage Prediction System

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Generate Dataset
np.random.seed(42)

normal = np.random.uniform(210, 250, 500)
overvoltage = np.random.uniform(254, 280, 250)
undervoltage = np.random.uniform(150, 206, 250)

voltage = np.concatenate([normal, overvoltage, undervoltage])

condition = (
    ["Normal"] * 500 +
    ["Overvoltage"] * 250 +
    ["Undervoltage"] * 250
)

data = pd.DataFrame({
    "Voltage": voltage,
    "Condition": condition
})

# 2. Data Preprocessing
X = data[["Voltage"]]
y = data["Condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train AI Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print(" AI VOLTAGE PREDICTION SYSTEM")
print("====================================")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, y_pred))

# 5. Voltage Prediction
while True:
    try:
        voltage_input = float(
            input("\nEnter Voltage (or 0 to Exit): ")
        )

        if voltage_input == 0:
            print("System Stopped!")
            break

        if voltage_input < 0:
            print("Invalid Voltage!")
            continue

        if voltage_input < 207:
            result = "Undervoltage"
        elif voltage_input > 253:
            result = "Overvoltage"
        else:
            result = "Normal"

        prediction = model.predict(
            pd.DataFrame({"Voltage": [voltage_input]})
        )[0]

        print("\n---------- RESULT ----------")
        print("Input Voltage:", voltage_input, "V")
        print("Predicted Condition:", prediction)

        if result == "Normal":
            print("Status: NORMAL VOLTAGE")
        elif result == "Overvoltage":
            print("ALERT: OVERVOLTAGE DETECTED!")
        else:
            print("ALERT: UNDERVOLTAGE DETECTED!")

    except ValueError:
        print("Please enter a valid number.")

# 6. Voltage Graph
plt.figure(figsize=(8, 5))

plt.scatter(
    data["Voltage"],
    range(len(data)),
    c=data["Voltage"],
    cmap="viridis"
)

plt.xlabel("Voltage (V)")
plt.ylabel("Data Samples")
plt.title("AI-Based Voltage Monitoring System")
plt.grid(True)
plt.show()

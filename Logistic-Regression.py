import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# 1. Machine Learning: Advanced ANN Student Performance Predictor
# ----------------------------------------------------
print("Training the Artificial Neural Network (ANN) model...")

# Generate synthetic dataset for Study Hours & Attendance Rate
X_raw, y_raw = make_classification(
    n_samples=600,
    n_features=2,
    n_redundant=0,
    n_classes=2,
    weights=[0.5, 0.5],
    random_state=42
)

# Convert to Pandas DataFrame with explicit feature labels
X = pd.DataFrame(X_raw, columns=["Study Hours", "Attendance Rate"])
y = pd.Series(y_raw)

# Split data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an Advanced Artificial Neural Network (Multi-Layer Perceptron Classifier)
# Hidden layer with (10, 5) neurons handles non-linear relationships better than basic Logistic Regression
model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Calculate model accuracy
y_pred = model.predict(X_test_scaled)
model_accuracy = accuracy_score(y_test, y_pred) * 100

print(f"Training completed successfully! ANN Model Accuracy: {model_accuracy:.2f}%")

# ----------------------------------------------------
# 2. Tkinter Graphical User Interface (GUI)
# ----------------------------------------------------
root = tk.Tk()
root.title("Student Performance Predictor - ANN Powered")
root.geometry("500x480")
root.config(bg="#f8fafc")

# Main Title
title_label = tk.Label(
    root,
    text="Student Success Prediction (Neural Network)",
    font=("Arial", 13, "bold"),
    bg="#f8fafc",
    fg="#0f172a"
)
title_label.pack(pady=15)

# Display Model Accuracy
acc_label = tk.Label(
    root,
    text=f"Model Accuracy (ANN): {model_accuracy:.2f}%",
    font=("Arial", 10),
    bg="#e2e8f0",
    fg="#1e293b",
    padx=10, pady=5
)
acc_label.pack(pady=5)

# Input Fields Frame
frame = tk.Frame(root, bg="#f8fafc")
frame.pack(pady=15)

# Feature 1: Study Hours
tk.Label(frame, text="Weekly Study Hours:", font=("Arial", 10, "bold"), bg="#f8fafc", fg="#334155").grid(row=0, column=0, sticky="w", pady=8)
entry_f1 = tk.Entry(frame, font=("Arial", 11), width=15, relief="solid", bd=1)
entry_f1.grid(row=0, column=1, pady=8, padx=10)
entry_f1.insert(0, "5.5")

# Feature 2: Attendance Rate
tk.Label(frame, text="Attendance Rate (%):", font=("Arial", 10, "bold"), bg="#f8fafc", fg="#334155").grid(row=1, column=0, sticky="w", pady=8)
entry_f2 = tk.Entry(frame, font=("Arial", 11), width=15, relief="solid", bd=1)
entry_f2.grid(row=1, column=1, pady=8, padx=10)
entry_f2.insert(0, "85.0")

# Prediction Core Logic Function
def make_prediction(event=None):
    try:
        # Read inputs
        study_hours = float(entry_f1.get())
        attendance = float(entry_f2.get())

        # Wrap inputs into a matching DataFrame
        user_input_df = pd.DataFrame([[study_hours, attendance]], columns=["Study Hours", "Attendance Rate"])

        # Scale inputs
        user_scaled = scaler.transform(user_input_df)

        # Predict using the Neural Network model
        prediction = model.predict(user_scaled)[0]
        probabilities = model.predict_proba(user_scaled)[0]
        confidence = probabilities[prediction] * 100

        # Update Result Display
        if prediction == 1:
            result_text = f"Predicted Outcome: PASS\nConfidence: {confidence:.2f}%"
            color = "#16a34a"  # Green
        else:
            result_text = f"Predicted Outcome: FAIL\nConfidence: {confidence:.2f}%"
            color = "#dc2626"  # Red

        result_display.config(text=result_text, fg=color)

    except ValueError:
        messagebox.showerror("Input Error", "Please ensure you enter valid numeric values!")

# Main Predict Button
predict_btn = tk.Button(
    root,
    text="Predict Outcome",
    font=("Arial", 11, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    padx=20, pady=8,
    relief="raised",
    command=make_prediction
)
predict_btn.pack(pady=15)

# Bind ENTER key globally across inputs to trigger predictions seamlessly
entry_f1.bind("<Return>", make_prediction)
entry_f2.bind("<Return>", make_prediction)

# Results Display Label Frame
result_display = tk.Label(
    root,
    text="Enter details above or press Enter to predict",
    font=("Arial", 11, "bold"),
    bg="#ffffff",
    fg="#475569",
    width=38, height=3,
    relief="solid", bd=1
)
result_display.pack(pady=10)

# Run Application Loop
root.mainloop()
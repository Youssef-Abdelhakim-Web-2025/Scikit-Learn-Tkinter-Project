# Student Performance Predictor (ANN-Powered Desktop App)

A desktop application built with Python that combines **Artificial Neural Networks (ANN)** from `scikit-learn` with an interactive **Tkinter** graphical user interface. The application trains a multi-layer perceptron model in the background to forecast student pass/fail results based on study habits and attendance rates.

## 🚀 Key Features
* **Advanced Neural Network:** Uses `MLPClassifier` (Multi-Layer Perceptron) with hidden layers to capture non-linear patterns more effectively than standard linear/logistic models.
* **Keyboard Shortcuts:** Fully integrated keyboard event bindings (`<Return>`) allowing users to press **Enter** inside any text field to trigger instant analysis.
* **Interactive UI (Tkinter):** Clean dashboard featuring input fields, dynamic color-coded output (Green for Pass, Red for Fail), and confidence metrics.
* **Data Scaling:** Automatically normalizes custom inputs using `StandardScaler` to ensure prediction integrity.
* **Robust Error Handling:** Validates field inputs to prevent runtime crashes.

## 🛠️ Requirements
* Python 3.x
* Required libraries:
  ```bash
  pip install numpy pandas scikit-learn

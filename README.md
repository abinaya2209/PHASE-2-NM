import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import gradio as gr

# Example of a synthetic patient data set
# For simplicity, I’ll generate a dataset with common features: Age, Gender, Blood Pressure, Cholesterol, etc.
np.random.seed(42)

# Simulating a small patient dataset
data = {
    'Age': np.random.randint(20, 80, 1000),
    'Gender': np.random.choice(['Male', 'Female'], 1000),
    'Blood Pressure': np.random.randint(100, 180, 1000),
    'Cholesterol': np.random.randint(150, 300, 1000),
    'BMI': np.random.uniform(18, 35, 1000),
    'Smoking': np.random.choice(['Yes', 'No'], 1000),
    'Diabetes': np.random.choice(['Yes', 'No'], 1000),
    'Heart Disease': np.random.choice([0, 1], 1000)  # Target variable (1=Disease, 0=No Disease)
}

df = pd.DataFrame(data)

# Encode categorical variables (Gender, Smoking, Diabetes)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0})
df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})

# Splitting the dataset into features and target variable
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model building: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Feature importance visualization
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance for Heart Disease Prediction')
plt.show()

# Define prediction function for Gradio interface
def predict_disease(age, gender, blood_pressure, cholesterol, bmi, smoking, diabetes):
    # Preprocessing input
    input_data = np.array([[age, gender, blood_pressure, cholesterol, bmi, smoking, diabetes]])
    input_data_scaled = scaler.transform(input_data)

    # Predict disease risk
    prediction = model.predict(input_data_scaled)

    # Return prediction result
    if prediction[0] == 1:
        return "The patient is at risk for heart disease."
    else:
        return "The patient is not at risk for heart disease."

# Gradio interface for clinicians and patients to interact with the model
iface = gr.Interface(
    fn=predict_disease,
    inputs=[
        gr.inputs.Slider(minimum=20, maximum=80, default=50, label="Age"),
        gr.inputs.Radio(["Male", "Female"], label="Gender"),
        gr.inputs.Slider(minimum=100, maximum=180, default=120, label="Blood Pressure"),
        gr.inputs.Slider(minimum=150, maximum=300, default=200, label="Cholesterol"),
        gr.inputs.Slider(minimum=18, maximum=35, default=25, label="BMI"),
        gr.inputs.Radio(["Yes", "No"], label="Smoking"),
        gr.inputs.Radio(["Yes", "No"], label="Diabetes")
    ],
    outputs="text",
    live=True
)

# Launch Gradio interface
iface.launch()

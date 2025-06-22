from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import re

app = FastAPI()

clf = joblib.load("ml_model_suggestion_rf.pkl")
le = joblib.load("label_encoder.pkl")

class CodeInput(BaseModel):
    code: str
# Template mapping
TEMPLATES = {
    "Linear Regression (scikit-learn)": """from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
""",
    "XGBoost": """import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
""",
    "TensorFlow": """import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)
""",
    "Keras": """from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=10))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=10)
""",
    "PyTorch": """import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = Net()
""",
    "Scikit-learn (general)": """from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
""",
    "Other / Unknown": "No code suggestion available."
}

def extract_features(snippet):
    lines = snippet.strip().split('\n')
    num_lines = len(lines)
    num_chars = len(snippet)
    avg_line_length = sum(len(line) for line in lines) / num_lines if num_lines > 0 else 0

    num_keywords = len(re.findall(r"\b(def|class|fit|predict|compile|train|model|transform)\b", snippet))
    num_functions = snippet.count("def")
    num_classes = snippet.count("class")

    uses_linear_regression = "LinearRegression" in snippet
    uses_xgboost = "xgboost" in snippet
    uses_tensorflow = "tensorflow" in snippet
    uses_keras = "keras" in snippet
    uses_torch = any(token in snippet for token in [
        "import torch", "torch.nn", "torch.optim", "torch.Tensor", "nn.Module", "F.relu"
    ])
    uses_sklearn = "sklearn" in snippet or "from sklearn" in snippet
    uses_pandas = "pandas" in snippet or "pd.DataFrame" in snippet

    return pd.DataFrame([{
        "num_lines": num_lines,
        "num_chars": num_chars,
        "avg_line_length": avg_line_length,
        "num_keywords": num_keywords,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "uses_linear_regression": uses_linear_regression,
        "uses_xgboost": uses_xgboost,
        "uses_tensorflow": uses_tensorflow,
        "uses_keras": uses_keras,
        "uses_torch": uses_torch,
        "uses_sklearn": uses_sklearn,
        "uses_pandas": uses_pandas
    }])

@app.post("/suggest")
async def suggest_code(data: CodeInput):
    features = extract_features(data.code)
    pred = clf.predict(features)
    label = le.inverse_transform(pred)[0]
    suggestion = TEMPLATES.get(label, "No suggestion available.")
    return {
        "label": label,
        "suggestion": suggestion
    }


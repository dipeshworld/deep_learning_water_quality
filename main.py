# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 2. LOAD DATA
df = pd.read_csv("dataset/water_quality.csv")

# 3. DATA PREPROCESSING
# Drop non-numeric / irrelevant columns
drop_cols = ["Well_ID", "State", "District", "Block", "Village"]
df = df.drop(columns=drop_cols, errors='ignore')

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Separate targets
target_reg = "WQI"
target_clf = "Water Quality Classification"

# Encode classification target
le = LabelEncoder()
df[target_clf] = le.fit_transform(df[target_clf])

# Features
X = df.drop(columns=[target_reg, target_clf])
y_reg = df[target_reg]
y_clf = df[target_clf]

# Train-test split
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. REGRESSION MODEL (WQI)
reg_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

reg_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(patience=10, restore_best_weights=True)

history_reg = reg_model.fit(
    X_train, y_reg_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
y_reg_pred = reg_model.predict(X_test).flatten()

# Evaluation
r2 = r2_score(y_reg_test, y_reg_pred)

print("\n===== REGRESSION RESULTS =====")
print("R² Score:", r2)

# 5. CLASSIFICATION MODEL
num_classes = len(np.unique(y_clf))

clf_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

clf_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_clf = clf_model.fit(
    X_train, y_clf_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
y_clf_pred_probs = clf_model.predict(X_test)
y_clf_pred = np.argmax(y_clf_pred_probs, axis=1)

# Evaluation
accuracy = accuracy_score(y_clf_test, y_clf_pred)
f1 = f1_score(y_clf_test, y_clf_pred, average='weighted')

print("\n===== CLASSIFICATION RESULTS =====")
print("Accuracy:", accuracy)
print("F1 Score:", f1)

# 6. SAVE MODELS
reg_model.save("wqi_regression_model.h5")
clf_model.save("water_quality_classifier.h5")

print("\nModels saved successfully!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from tensorflow import keras
import tensorflow as tf
from sklearn.feature_selection import RFE

# Load dataset
df = pd.read_csv("D:/CCP/Dataset/Customer_churn_dataset.csv")
df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

df = df[df['TotalCharges'] < df['TotalCharges'].quantile(0.99)]

df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
binary_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", 
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
               "StreamingMovies", "PaperlessBilling", "Churn"]
df[binary_cols] = df[binary_cols].replace({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1})
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['TenureGroups'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72], labels=[1, 2, 3, 4, 5])
df = pd.get_dummies(df, columns=['TenureGroups'], drop_first=True)

scaler = MinMaxScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth']] = scaler.fit_transform(
    df[['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargesPerMonth']])

# Handle class imbalance using SMOTEENN
X = df.drop('Churn', axis=1)
y = df['Churn']
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=5)

# Feature selection with RFE
selector = RFE(estimator=RandomForestClassifier(n_estimators=500, random_state=42), n_features_to_select=15)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_selected, y_train)
log_pred = log_model.predict(X_test_selected)
print("Logistic Regression Classification Report:\n", classification_report(y_test, log_pred))

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=10, random_state=42)
rf_model.fit(X_train_selected, y_train)
rf_pred = rf_model.predict(X_test_selected)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# XGBoost GridSearchCV
params = {
    "n_estimators": [300, 500],
    "max_depth": [6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid=params, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_selected, y_train)
best_xgb_params = grid_search.best_params_

xgb_model = XGBClassifier(**best_xgb_params, random_state=42)
xgb_model.fit(X_train_selected, y_train)
xgb_pred = xgb_model.predict(X_test_selected)
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# Create and Train ANN Model
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(X_train_selected.shape[1],), kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

ann_model = create_model()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = ann_model.fit(X_train_selected, y_train, epochs=300, batch_size=32, validation_data=(X_test_selected, y_test), callbacks=[early_stopping, lr_scheduler])

y_pred_ann = (ann_model.predict(X_test_selected) > 0.5).astype(int)
print("ANN Classification Report:\n", classification_report(y_test, y_pred_ann))

# Ensemble Model
ensemble_model = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('xgb', xgb_model)
], voting='soft')

ensemble_model.fit(X_train_selected, y_train)
ensemble_pred = ensemble_model.predict(X_test_selected)
print("Ensemble Model Classification Report:\n", classification_report(y_test, ensemble_pred))

# Combine confusion matrices for all models into a single DataFrame
models = {'Logistic Regression': log_pred, 'Random Forest': rf_pred, 'XGBoost': xgb_pred, 'ANN': y_pred_ann.flatten()}
cm_list = []

for name, pred in models.items():
    cm = confusion_matrix(y_test, pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    cm_df['Model'] = name  # Add model name to the dataframe
    cm_list.append(cm_df)

# Concatenate all confusion matrices into a single DataFrame
cm_combined = pd.concat(cm_list, axis=0, ignore_index=False)

# Visualize the combined confusion matrices in a table format
plt.figure(figsize=(10, 6))
sns.heatmap(cm_combined.pivot(index='Model', columns='Predicted 0', values='Actual 0'), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for All Models')
plt.show()

#Heart_disease_prediction.py using machine learning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#load database
df = pd.read_csv("heart.csv")

#display 5 rows
# print(df.head())

# #show basic info
# print("\nDatabase info:")
# print(df.info())

# #describe numerical stats
# print("\nBasic statistic:")
# print(df.describe())

# #check for  missing value
# print("\nMissing value:")
# print(df.isnull().sum())

# print("\nShape:")
# print(df.shape)

# print(df['HeartDisease'].value_counts())

# # Set style
# sns.set(style="whitegrid")

# # Plot HeartDisease count
# plt.figure(figsize=(6, 4))
# sns.countplot(x='HeartDisease', data=df, palette='viridis')
# plt.title("Heart Disease Distribution")
# plt.show()

# # Age distribution by Heart Disease
# plt.figure(figsize=(8, 5))
# sns.histplot(data=df, x='Age', hue='HeartDisease', kde=True, palette='magma', bins=30)
# plt.title("Age Distribution by Heart Disease")
# plt.show()

# # Sex vs Heart Disease
# plt.figure(figsize=(6, 4))
# sns.countplot(x='Sex', hue='HeartDisease', data=df, palette='Set2')
# plt.title("Heart Disease by Sex")
# plt.show()


# First, check the column names
print(df.columns)

# List of categorical columns to encode
cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Show final shapes
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# # Train the model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# Predict
# y_pred = model.predict(X_test)

# # Evaluate
# accuracy = accuracy_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print(f"✅ Accuracy: {accuracy:.4f}")
# print("\n📊 Confusion Matrix:\n", cm)
# print("\n📋 Classification Report:\n",report)


# Initialize and train
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)

print("✅ Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\n📋 Classification Report:\n", classification_report(y_test,y_pred_rf))




# # Initialize and train
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb_model.fit(X_train, y_train)

# # Predict and evaluate
# y_pred_xgb = xgb_model.predict(X_test)

# print("✅ XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
# print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
# print("\n📋 Classification Report:\n", classification_report(y_test,y_pred_xgb))



import joblib

# Save the trained Random Forest model
joblib.dump(rf_model,'heart_disease_model.pkl')
print("✅ Model saved as 'heart_disease_model.pkl'")


rf_model = joblib.load('heart_disease_model.pkl')
print("✅ Model loaded successfully")

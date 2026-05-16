import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Load
df = pd.read_csv("train.csv")

print(df.head())

# Data Understanding
print(df.info())

print(df.isnull().sum())

# Fill Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove Unnecessary Columns
df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

# Convert Categorical Data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

df['Embarked'] = le.fit_transform(df['Embarked'])

# Features and Target
X = df.drop('Survived', axis=1)

y = df['Survived']

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


import joblib
joblib.dump(model,"titanic_model.pkl")
print("model saved successfully")
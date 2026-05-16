import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("train.csv")

# Sirf 4 features use karo
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].copy()

# Missing values handle karo
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Sex ko number mein convert
df['Sex'] = (df['Sex'] == 'male').astype(int)

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "titanic_model.pkl")

print("✅ Model trained successfully!")
print("Features used:", X.columns.tolist())
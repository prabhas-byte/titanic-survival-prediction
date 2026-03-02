import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("TITANIC SURVIVAL PREDICTION")
print("Loading data...")


try:
    df = pd.read_csv('titanic.csv')
    print(f"Dataset loaded – shape: {df.shape}")
except FileNotFoundError:
    print("ERROR: titanic.csv not found in the current folder.")
    print("Please place the file 'titanic.csv' in the same directory as this script.")
    exit()
except Exception as e:
    print("ERROR loading file:", str(e))
    exit()

print("\n1. Basic information")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

print("\n2. Quick visualizations")

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df, palette='viridis')
plt.title('Survived vs Not Survived')

plt.subplot(2, 2, 2)
sns.barplot(x='Sex', y='Survived', data=df, palette='coolwarm')
plt.title('Survival Rate by Gender')

plt.subplot(2, 2, 3)
sns.barplot(x='Pclass', y='Survived', data=df, palette='magma')
plt.title('Survival Rate by Passenger Class')

plt.subplot(2, 2, 4)
sns.histplot(df['Age'].dropna(), kde=True, bins=30, color='teal')
plt.title('Age Distribution')

plt.tight_layout()
plt.show()

print("\n3. Preprocessing...")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print("Data after preprocessing:")
print(df.head())

print("\n4. Preparing data for modeling")

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

print("\n5. Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("\n6. Results")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "═"*60)
print(f"FINAL ACCURACY: {accuracy:.4f}  →  {accuracy*100:.2f}%")
print("═"*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)

plt.figure(figsize=(10,6))
importances.plot(kind='barh', color='skyblue')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

print("\n" + "═"*60)
print("Done. Model training and evaluation completed.")
print("Typical good accuracy for this task: 78–84%")
print("═"*60)
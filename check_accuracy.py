import pandas as pd
import joblib
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data and model
df = pd.read_csv('RT_IOT2022.csv')
rf_model = joblib.load('random_forest_model.pkl')

# Preprocess
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = pd.get_dummies(df, columns=['proto', 'service'], drop_first=True)

le = LabelEncoder()
df['Attack_type'] = le.fit_transform(df['Attack_type'])

X = df.drop(columns=['Attack_type'])
y = df['Attack_type']

# Generate a random seed for this run
seed = random.randint(1, 100000)

# Split with random shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Predict
y_pred = rf_model.predict(X_test)

# Metrics
print("=" * 40)
print("MODEL ACCURACY METRICS")
print("=" * 40)
print(f"Random Seed: {seed}")
print("-" * 40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
print("=" * 40)

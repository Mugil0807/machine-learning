import pandas as pd
import numpy as np
import chardet
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, r2_score, silhouette_score,
    calinski_harabasz_score, davies_bouldin_score
)
import matplotlib.pyplot as plt
import warnings
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Data Loading and Preprocessing ---
print("--- 1. Data Loading and Preprocessing ---")

# Try to load the CSV file
try:
    print("Loading CSV file...")
    # Create a clean version of the file first
    clean_lines = []
    with open('lab/original.csv', 'r', encoding='cp1252', errors='replace') as file:
        # Read header
        header = file.readline().strip()
        clean_lines.append(header)
        
        # Read remaining lines and clean them
        for line in file:
            # Replace any problematic characters
            clean_line = line.strip().replace('\x00', '')
            if clean_line:  # Only add non-empty lines
                clean_lines.append(clean_line)
    
    # Join lines and create a string buffer
    clean_content = '\n'.join(clean_lines)
    
    # Read the cleaned content with pandas
    df = pd.read_csv(StringIO(clean_content),
                    engine='python',
                    sep=',',
                    skipinitialspace=True,
                    quoting=1,  # QUOTE_ALL
                    quotechar='"',
                    on_bad_lines='skip')
    
    print("Successfully loaded the file!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("Successfully loaded the file!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading file: {str(e)}")
    print("Current working directory:", os.getcwd())
    raise

# Define values to be treated as NaN
na_values = ['NA', 'É........', 'É....(NA)', 'É.......(NA)', '...']
df.replace(na_values, np.nan, inplace=True)

# Drop rows with missing values in crucial columns
df.dropna(subset=['teacher', 'student', 'fluency'], inplace=True)
print(f"Shape after dropping NaNs: {df.shape}")

# Combine teacher and student text into a single feature
df['text'] = df['teacher'].astype(str) + ' ' + df['student'].astype(str)

# Encode the target variable 'fluency' into numerical format
fluency_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['fluency_encoded'] = df['fluency'].map(fluency_mapping)

# Drop rows where mapping resulted in NaN
df.dropna(subset=['fluency_encoded'], inplace=True)
df['fluency_encoded'] = df['fluency_encoded'].astype(int)

print("Target variable 'fluency' encoded into numerical values (0: Low, 1: Medium, 2: High).")

# Feature Engineering: TF-IDF
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray()
y = df['fluency_encoded'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("-" * 50)

# --- Helper function ---
def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\nMetrics for {dataset_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    print(f"  R2 Score: {r2:.4f}")
    return mse, rmse, mape, r2

# --- A1 & A2 ---
print("\n--- A1 & A2: Linear Regression with Single Feature ---")
X_train_single = X_train[:, 0].reshape(-1, 1)
X_test_single = X_test[:, 0].reshape(-1, 1)

reg_single = LinearRegression().fit(X_train_single, y_train)
y_train_pred_single = reg_single.predict(X_train_single)
calculate_metrics(y_train, y_train_pred_single, "Train (Single Feature)")
y_test_pred_single = reg_single.predict(X_test_single)
calculate_metrics(y_test, y_test_pred_single, "Test (Single Feature)")

# --- A3 ---
print("\n--- A3: Linear Regression with All Features ---")
reg_all = LinearRegression().fit(X_train, y_train)
y_train_pred_all = reg_all.predict(X_train)
calculate_metrics(y_train, y_train_pred_all, "Train (All Features)")
y_test_pred_all = reg_all.predict(X_test)
calculate_metrics(y_test, y_test_pred_all, "Test (All Features)")

# --- A4 & A5 ---
print("\n--- A4 & A5: K-Means Clustering (k=2) ---")
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_train)
silhouette_2 = silhouette_score(X_train, kmeans_2.labels_)
ch_score_2 = calinski_harabasz_score(X_train, kmeans_2.labels_)
db_score_2 = davies_bouldin_score(X_train, kmeans_2.labels_)
print(f"Silhouette: {silhouette_2:.4f}, CH: {ch_score_2:.4f}, DB: {db_score_2:.4f}")

# --- A6 ---
print("\n--- A6: Optimal K Evaluation ---")
k_values = range(2, 11)
silhouette_scores, ch_scores, db_scores = [], [], []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_train)
    labels = km.labels_
    silhouette_scores.append(silhouette_score(X_train, labels))
    ch_scores.append(calinski_harabasz_score(X_train, labels))
    db_scores.append(davies_bouldin_score(X_train, labels))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
ax1.plot(k_values, silhouette_scores, 'bo-')
ax1.set_ylabel('Silhouette')
ax2.plot(k_values, ch_scores, 'go-')
ax2.set_ylabel('Calinski-Harabasz')
ax3.plot(k_values, db_scores, 'ro-')
ax3.set_xlabel('k')
ax3.set_ylabel('Davies-Bouldin')
plt.show()

# --- A7 ---
print("\n--- A7: Elbow Method ---")
distortions = []
for k in range(2, 20):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_train)
    distortions.append(km.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 20), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


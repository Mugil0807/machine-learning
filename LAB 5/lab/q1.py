import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Data Loading and Preprocessing ---
print("--- 1. Data Loading and Preprocessing ---")

# Load the dataset
try:
    df = pd.read_csv('original.csv')
    print("Dataset loaded successfully.")
    print(f"Original shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'original.csv' not found. Please make sure the file is in the correct directory.")
    exit()

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

# Drop rows where mapping resulted in NaN (if any unexpected 'fluency' values existed)
df.dropna(subset=['fluency_encoded'], inplace=True)
df['fluency_encoded'] = df['fluency_encoded'].astype(int)

print("Target variable 'fluency' encoded into numerical values (0: Low, 1: Medium, 2: High).")

# Feature Engineering: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray()
y = df['fluency_encoded'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("-" * 50)


# --- Helper function for metrics ---
def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculates and prints regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Add a small epsilon to avoid division by zero in MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\nMetrics for {dataset_name}:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
    print(f"  R-squared (R2) Score: {r2:.4f}")
    return mse, rmse, mape, r2

# --- A1 & A2: Linear Regression with One Attribute ---
print("\n--- A1 & A2: Linear Regression with a Single Feature ---")

# We select the first feature from the TF-IDF matrix as our single attribute
X_train_single = X_train[:, 0].reshape(-1, 1)
X_test_single = X_test[:, 0].reshape(-1, 1)

# A1: Train the model
reg_single = LinearRegression().fit(X_train_single, y_train)
print("Linear Regression model trained on a single feature.")

# A2: Predictions and Metrics
# On Training Data
y_train_pred_single = reg_single.predict(X_train_single)
calculate_metrics(y_train, y_train_pred_single, "Train Set (Single Feature)")

# On Test Data
y_test_pred_single = reg_single.predict(X_test_single)
calculate_metrics(y_test, y_test_pred_single, "Test Set (Single Feature)")
print("-" * 50)


# --- A3: Linear Regression with All Attributes ---
print("\n--- A3: Linear Regression with All Features ---")

# Train the model using all features
reg_all = LinearRegression().fit(X_train, y_train)
print("Linear Regression model trained on all features.")

# Predictions and Metrics
# On Training Data
y_train_pred_all = reg_all.predict(X_train)
calculate_metrics(y_train, y_train_pred_all, "Train Set (All Features)")

# On Test Data
y_test_pred_all = reg_all.predict(X_test)
calculate_metrics(y_test, y_test_pred_all, "Test Set (All Features)")
print("-" * 50)


# --- A4 & A5: K-Means Clustering (k=2) and Evaluation ---
print("\n--- A4 & A5: K-Means Clustering (k=2) ---")

# A4: Perform k-means clustering
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_train)
print("K-Means clustering performed with k=2.")
print(f"Cluster labels for the first 10 data points: {kmeans_2.labels_[:10]}")
# print(f"Cluster centers shape: {kmeans_2.cluster_centers_.shape}")

# A5: Calculate clustering scores
silhouette_2 = silhouette_score(X_train, kmeans_2.labels_)
ch_score_2 = calinski_harabasz_score(X_train, kmeans_2.labels_)
db_score_2 = davies_bouldin_score(X_train, kmeans_2.labels_)

print("\nClustering Evaluation Metrics for k=2:")
print(f"  Silhouette Score: {silhouette_2:.4f} (Higher is better)")
print(f"  Calinski-Harabasz Score: {ch_score_2:.4f} (Higher is better)")
print(f"  Davies-Bouldin Index: {db_score_2:.4f} (Lower is better)")
print("-" * 50)


# --- A6: Finding Optimal K using Evaluation Scores ---
print("\n--- A6: Finding Optimal K using Evaluation Scores ---")
k_values = range(2, 11)
silhouette_scores = []
ch_scores = []
db_scores = []

print("Calculating scores for k from 2 to 10...")
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_train)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_train, labels))
    ch_scores.append(calinski_harabasz_score(X_train, labels))
    db_scores.append(davies_bouldin_score(X_train, labels))

# Plotting the scores
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
fig.suptitle('Clustering Evaluation Scores vs. Number of Clusters (k)', fontsize=16)

# Silhouette Score Plot
ax1.plot(k_values, silhouette_scores, 'bo-')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score (Higher is better)')
ax1.grid(True)

# Calinski-Harabasz Score Plot
ax2.plot(k_values, ch_scores, 'go-')
ax2.set_ylabel('Calinski-Harabasz Score')
ax2.set_title('Calinski-Harabasz Score (Higher is better)')
ax2.grid(True)

# Davies-Bouldin Index Plot
ax3.plot(k_values, db_scores, 'ro-')
ax3.set_xlabel('Number of Clusters (k)')
ax3.set_ylabel('Davies-Bouldin Index')
ax3.set_title('Davies-Bouldin Index (Lower is better)')
ax3.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
print("Displaying plot for clustering evaluation scores...")
plt.show()
print("-" * 50)


# --- A7: Finding Optimal K using Elbow Method ---
print("\n--- A7: Finding Optimal K using the Elbow Method ---")
distortions = []
k_range = range(2, 20)

print("Calculating inertia for k from 2 to 19...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_train)
    distortions.append(kmeans.inertia_) # inertia_ is the sum of squared distances

# Plotting the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, distortions, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Distortion)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
print("Displaying elbow plot...")
plt.show()
print("-" * 50)

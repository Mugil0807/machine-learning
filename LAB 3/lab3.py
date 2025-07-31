    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc, classification_report
)
from scipy.spatial.distance import minkowski
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

print("=== FLUENCY ANALYTICS USING k-NN CLASSIFICATION ===\n")

# --- Load & Preprocess Data ---
print("🔄 Loading data file...")
try:
    try:
        print("   Attempting to load original.csv...")
        df = pd.read_csv("original.csv")
        print("✓ Data loaded from original.csv")
    except Exception as csv_error:
        print(f"   CSV failed: {csv_error}")
        print("   Attempting to load original.xlsx...")
        df = pd.read_excel("original.xlsx")
        print("✓ Data loaded from original.xlsx")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

print(f"✓ Original dataset shape: {df.shape}")

# Clean column names
df.columns = df.columns.str.strip().str.lower()
print(f"✓ Columns: {list(df.columns)}")

# === CRITICAL: Clean the fluency column to fix the sorting error ===
print("\n🔄 Cleaning fluency data...")

# Check what's in the fluency column
print("Raw fluency values sample:", df['fluency'].head(10).tolist())
print("Fluency column dtypes:", df['fluency'].dtype)

# Remove rows with missing values
original_size = len(df)
df = df.dropna(subset=['teacher', 'student', 'fluency'])
print(f"✓ Removed {original_size - len(df)} rows with missing values")

# Convert everything to string and clean
df['fluency'] = df['fluency'].astype(str).str.strip().str.lower()

# Fix common typos in fluency labels
fluency_corrections = {
    'hgih': 'high',      # Fix typo
    'mediu': 'medium',   # Fix typo  
    'na': 'medium',      # Convert 'na' to medium (or remove if preferred)
    'med': 'medium',     # Handle abbreviations
    'hi': 'high',
    'lo': 'low'
}

print("🔄 Fixing fluency label typos...")
original_counts = df['fluency'].value_counts()
print("Before corrections:", dict(original_counts))

# Apply corrections
df['fluency'] = df['fluency'].replace(fluency_corrections)

# Remove problematic entries
df = df[df['fluency'] != 'nan']
df = df[df['fluency'] != 'none'] 
df = df[df['fluency'] != '']
df = df[~df['fluency'].isin(['', ' ', 'null'])]

# Keep only main classes with enough samples
valid_classes = ['high', 'medium', 'low']
df = df[df['fluency'].isin(valid_classes)]

print("After corrections:", dict(df['fluency'].value_counts()))

print(f"✓ Final dataset size: {df.shape[0]} samples")

# Check unique fluency values
unique_fluency = df['fluency'].value_counts()
print(f"✓ Fluency distribution:\n{unique_fluency}")

if len(df) < 50:
    print("❌ Too few samples after cleaning. Check your data.")
    exit()

# Combine teacher and student text
df['combined_text'] = df['teacher'].astype(str) + " " + df['student'].astype(str)

# Display sample data
print("\n=== SAMPLE DATA ===")
for i in range(min(3, len(df))):
    print(f"Sample {i+1}:")
    print(f"  Teacher: {str(df['teacher'].iloc[i])[:60]}...")
    print(f"  Student: {str(df['student'].iloc[i])[:60]}...")
    print(f"  Fluency: '{df['fluency'].iloc[i]}'")
    print()

X_text = df['combined_text']
y = df['fluency']

# --- TF-IDF Vectorization ---
print("\n=== TF-IDF VECTORIZATION ===")
print("🔄 Processing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=200, stop_words='english', lowercase=True, min_df=2)
X = vectorizer.fit_transform(X_text).toarray()
print(f"✓ TF-IDF matrix shape: {X.shape}")

# --- Label Encoding (SAFE SORTING) ---
print("\n=== LABEL ENCODING ===")
# Now safe to sort since all values are cleaned strings
classes = sorted(y.unique())
y_encoded = pd.Categorical(y, categories=classes).codes

print(f"✓ Classes found: {classes}")
print(f"✓ Number of classes: {len(classes)}")
print(f"✓ Encoded labels range: {np.unique(y_encoded)}")

# Class distribution
for i, cls in enumerate(classes):
    count = np.sum(y_encoded == i)
    print(f"   Class '{cls}': {count} samples")

# Check if we have enough samples per class for stratified split
min_samples_per_class = min([np.sum(y_encoded == i) for i in np.unique(y_encoded)])
print(f"✓ Minimum samples per class: {min_samples_per_class}")

if min_samples_per_class < 2:
    print("⚠️  Using random split (not stratified) due to small class sizes")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )
else:
    print("✓ Using stratified split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

print("\n" + "="*60)
print("ANSWERING LAB QUESTIONS")
print("="*60)

# === A1: Intraclass Spread & Interclass Distance ===
print("\n=== A1: INTRACLASS SPREAD & INTERCLASS DISTANCE ===")

centroids = []
spreads = []

for cls_idx in np.unique(y_encoded):
    class_vecs = X[y_encoded == cls_idx]
    centroid = np.mean(class_vecs, axis=0)
    spread = np.mean(np.std(class_vecs, axis=0))
    
    centroids.append(centroid)
    spreads.append(spread)
    
    class_name = classes[cls_idx]
    print(f"Class '{class_name}': {len(class_vecs)} samples, Avg spread: {spread:.4f}")

# Calculate interclass distances
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        distance = np.linalg.norm(centroids[i] - centroids[j])
        print(f"Distance '{classes[i]}' ↔ '{classes[j]}': {distance:.4f}")

# === A2: Histogram, Mean, Variance of Feature ===
print("\n=== A2: FEATURE ANALYSIS ===")

feature_index = 0
feature_values = X[:, feature_index]
feature_name = vectorizer.get_feature_names_out()[feature_index]

print(f"Analyzing feature: '{feature_name}'")
print(f"Mean: {np.mean(feature_values):.6f}")
print(f"Variance: {np.var(feature_values):.6f}")

plt.figure(figsize=(10, 6))
plt.hist(feature_values, bins=20, edgecolor='black', alpha=0.7)
plt.title(f"Histogram of TF-IDF Feature: '{feature_name}'")
plt.xlabel("TF-IDF Value")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.show()

# === A3: Minkowski Distance ===
print("\n=== A3: MINKOWSKI DISTANCE ANALYSIS ===")

vec1 = X_train[0]
vec2 = X_train[1]

minkowski_distances = []
for r in range(1, 11):
    distance = minkowski(vec1, vec2, p=r)
    minkowski_distances.append(distance)
    print(f"Minkowski distance (r={r}): {distance:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), minkowski_distances, marker='o', linewidth=2)
plt.title("Minkowski Distance vs r Parameter")
plt.xlabel("r")
plt.ylabel("Distance")
plt.grid(True, alpha=0.3)
plt.show()

# === A4-A5: Train kNN ===
print("\n=== A4-A5: k-NN TRAINING ===")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("✓ k-NN classifier trained (k=3)")

# === A6: Accuracy ===
print("\n=== A6: ACCURACY ===")

train_acc = knn.score(X_train, y_train)
test_acc = knn.score(X_test, y_test)

print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# === A7: Sample Predictions ===
print("\n=== A7: SAMPLE PREDICTIONS ===")

predictions = knn.predict(X_test[:5])
for i in range(5):
    actual = classes[y_test[i]]
    predicted = classes[predictions[i]]
    print(f"Sample {i+1}: Actual='{actual}', Predicted='{predicted}'")

# === A8: Accuracy vs k ===
print("\n=== A8: ACCURACY vs k ===")

k_values = list(range(1, 11))
test_accuracies = []

print("🔄 Testing different k values...")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    test_accuracies.append(acc)
    print(f"k={k}: {acc:.4f}")

optimal_k = k_values[np.argmax(test_accuracies)]
print(f"\n✓ Optimal k: {optimal_k} (Accuracy: {max(test_accuracies):.4f})")

plt.figure(figsize=(10, 6))
plt.plot(k_values, test_accuracies, 'o-', linewidth=2)
plt.title("Test Accuracy vs k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.show()

# === A9: Confusion Matrix ===
print("\n=== A9: CLASSIFICATION METRICS ===")

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"Classes: {classes}")

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nMacro Averages:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=classes))

print("\n" + "="*60)
print("OPTIONAL QUESTIONS")
print("="*60)

# === O1: Normal Distribution ===
print("\n=== O1: NORMAL DISTRIBUTION COMPARISON ===")

feature_mean = np.mean(feature_values)
feature_std = np.std(feature_values)
normal_data = np.random.normal(feature_mean, feature_std, len(feature_values))

plt.figure(figsize=(10, 6))
plt.hist(feature_values, bins=20, alpha=0.7, label='TF-IDF Feature', density=True)
plt.hist(normal_data, bins=20, alpha=0.7, label='Normal Distribution', density=True)
plt.title("Feature Distribution vs Normal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# === O2: Distance Metrics ===
print("\n=== O2: DISTANCE METRICS ===")

metrics = ['euclidean', 'manhattan', 'cosine']
for metric in metrics:
    knn_metric = KNeighborsClassifier(n_neighbors=optimal_k, metric=metric)
    knn_metric.fit(X_train, y_train)
    acc = knn_metric.score(X_test, y_test)
    print(f"{metric.capitalize()}: {acc:.4f}")

# === O3: ROC Curve ===
print("\n=== O3: ROC ANALYSIS ===")

if len(classes) == 2:
    print("Binary classification - generating ROC curve...")
    y_proba = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"AUC: {roc_auc:.4f}")
else:
    print(f"Multi-class problem ({len(classes)} classes) - ROC for binary only")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Dataset: {len(df)} samples")
print(f"✓ Classes: {classes}")
print(f"✓ Features: {X.shape[1]} TF-IDF features")
print(f"✓ Best k: {optimal_k}")
print(f"✓ Best accuracy: {max(test_accuracies):.4f}")
print("✓ All lab questions completed!")
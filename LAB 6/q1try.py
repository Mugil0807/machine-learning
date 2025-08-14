import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# -------------------- A1: Entropy Function --------------------
def calculate_entropy(y):
    """Calculates the entropy of a set of labels."""
    if len(y) == 0:
        return 0
    value_counts = Counter(y)
    total_samples = len(y)
    entropy = 0
    for count in value_counts.values():
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy

# -------------------- A2: Gini Index Function --------------------
def calculate_gini_index(y):
    """Calculates the Gini impurity of a set of labels."""
    if len(y) == 0:
        return 0
    value_counts = Counter(y)
    total_samples = len(y)
    gini = 1.0
    for count in value_counts.values():
        probability = count / total_samples
        gini -= probability ** 2
    return gini

# -------------------- A3: Information Gain Function --------------------
def calculate_information_gain(X, y, feature_index):
    """Calculates the information gain for a specific feature."""
    parent_entropy = calculate_entropy(y)
    feature_values = np.unique(X[:, feature_index])
    weighted_entropy = 0
    total_samples = len(y)
    for value in feature_values:
        mask = X[:, feature_index] == value
        subset_y = y[mask]
        if len(subset_y) > 0:
            subset_entropy = calculate_entropy(subset_y)
            weight = len(subset_y) / total_samples
            weighted_entropy += weight * subset_entropy
    return parent_entropy - weighted_entropy

def rank_information_gain(X_dense, y, feature_names, top_n=10):
    """Ranks all features by their information gain."""
    gains = []
    for i in range(X_dense.shape[1]):
        gain = calculate_information_gain(X_dense, y, i)
        gains.append((feature_names[i], gain))
    gains_sorted = sorted(gains, key=lambda x: x[1], reverse=True)
    return gains_sorted

# -------------------- A5: Decision Tree Build Function --------------------
def build_decision_tree(X, y, max_depth=4):
    """Builds and trains a Decision Tree classifier."""
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf

# -------------------- A6: Decision Tree Plot Function --------------------
def plot_decision_tree_sklearn(clf, feature_names, class_names):
    """Visualizes the trained Decision Tree."""
    plt.figure(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Visualization")
    plt.show()

# -------------------- A7: PCA-based Decision Boundary Function --------------------
def plot_decision_boundary_pca(X_tfidf, y, class_names):
    """Reduces data to 2D using PCA and plots the decision boundary."""
    # --- PCA: Dimensionality Reduction ---
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_tfidf.toarray())
    
    # --- Train a new tree on the 2D data ---
    clf_2d = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    clf_2d.fit(X_2d, y)
    
    # --- Create a mesh grid to plot the boundary ---
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # --- Predict on the grid and plot ---
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=[class_names[i] for i in y],
                    palette="deep", edgecolor='k', s=80)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Decision Boundary (PCA-reduced TF-IDF space)")
    plt.show()

# -------------------- MAIN EXECUTION BLOCK --------------------
if __name__ == "__main__":
    # Step 1: Load and clean the dataset
    try:
        df = pd.read_csv('original.csv', encoding='latin-1', engine='python')
    except FileNotFoundError:
        print("Error: 'original.csv' not found. Make sure it's in the same directory.")
        exit()

    # Step 2: Preprocess the data
    df.rename(columns={'student': 'text', 'fluency': 'target'}, inplace=True)
    df = df.dropna(subset=['text', 'target'])
    df['target'] = df['target'].astype(str).str.strip().str.lower()
    df = df[~df['target'].isin(["na", "nan", "none", "n/a"])] # Remove invalid targets

    # Step 3: Vectorize the text data using TF-IDF
    # This converts the text into a numerical matrix
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()
    X_dense = X_tfidf.toarray() # Convert to a dense array for calculations

    # Step 4: Encode the target labels
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['target'])

    # ---------------- A1 & A2: Measure Impurity ----------------
    print("\n--- A1 & A2: Impurity Measures ---")
    print(f"Entropy of target: {calculate_entropy(y):.4f}")
    print(f"Gini index of target: {calculate_gini_index(y):.4f}")

    # ---------------- A3: Rank Features by Information Gain ----------------
    print("\n--- A3: Top 10 Features by Information Gain ---")
    gains_sorted = rank_information_gain(X_dense, y, feature_names, top_n=10)
    for feature, gain in gains_sorted[:10]:
        print(f"{feature:20} {gain:.4f}")

    # ---------------- A5: Build, Train, and Evaluate the Tree ----------------
    print("\n--- A5: Decision Tree Evaluation ---")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    clf = build_decision_tree(X_train, y_train, max_depth=4)
    y_pred = clf.predict(X_test)
    
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    
    # Map numeric predictions back to original labels
    y_test_labels = le_target.inverse_transform(y_test)
    y_pred_labels = le_target.inverse_transform(y_pred)
    
    # Get unique classes that actually appear in the test set and predictions
    unique_classes = sorted(list(set(y_test_labels) | set(y_pred_labels)))
    print(classification_report(y_test_labels, y_pred_labels, target_names=unique_classes, zero_division=0))

    # --- Plot Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ---------------- A6: Visualize the Full Tree ----------------
    print("\n--- A6: Generating Decision Tree Visualization ---")
    plot_decision_tree_sklearn(clf, feature_names, le_target.classes_)

    # ---------------- A7: Visualize the Decision Boundary ----------------
    print("\n--- A7: Generating Decision Boundary Plot ---")
    plot_decision_boundary_pca(X_tfidf, y, le_target.classes_)


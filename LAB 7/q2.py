import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import catboost as cb
import warnings
import pandas as pd

df = pd.read_csv("original.csv", 
                encoding='latin-1',
                on_bad_lines='skip',  # Skip problematic rows
                engine='python',      # Use Python parser which is more flexible
                sep=None)            # Automatically detect separator
print(df.head())

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =======================
# 1. Load and Clean Data
# ======================
    

# Keep only student + fluency columns
df = df[["student", "fluency"]]

# Drop missing values
df.dropna(subset=["student", "fluency"], inplace=True)

# Remove rows where fluency is "NA"
df = df[df["fluency"].str.upper() != "NA"]

# Clean student text (remove newlines/extra spaces)
df["student"] = df["student"].str.replace(r"\s+", " ", regex=True).str.strip()

# Normalize fluency labels (title case: High, Low, Medium)
df["fluency"] = df["fluency"].str.strip().str.title()

# Check class distribution and remove rare classes
class_counts = df["fluency"].value_counts()
print("\nClass distribution before filtering:")
print(class_counts)

# Remove classes with less than 2 samples
min_samples = 2
valid_classes = class_counts[class_counts >= min_samples].index
df = df[df["fluency"].isin(valid_classes)]

print("\nClass distribution after filtering:")
print(df["fluency"].value_counts())

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["fluency"])

# Define features (X) and target (y)
X = df["student"]

# =======================
# 2. Train/Test Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =======================
# 3. TF-IDF Vectorization
# =======================
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =======================
# 4. Define Models + Grids
# =======================
models = {
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="mlogloss"),
    "CatBoost": cb.CatBoostClassifier(random_state=42, verbose=0, thread_count=-1, allow_writing_files=False),
    "Naïve-Bayes": MultinomialNB(),
    "MLP Classifier": MLPClassifier(
        random_state=42,
        max_iter=1000,      # Increased maximum iterations
        early_stopping=True, # Enable early stopping
        learning_rate='adaptive',  # Use adaptive learning rate
        validation_fraction=0.1,   # Use 10% of training data for early stopping
        n_iter_no_change=10       # Number of iterations with no improvement to wait before early stopping
    ),
}

param_dist = {
    "Support Vector Machine": {
        "C": [1, 10],
        "kernel": ["rbf", "linear"],
    },
    "Decision Tree": {
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    },
    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [10, 20],
    },
    "AdaBoost": {
        "n_estimators": [50],
        "learning_rate": [0.1, 1],
    },
    "XGBoost": {
        "n_estimators": [100],
        "learning_rate": [0.1],
        "max_depth": [3],
    },
    "CatBoost": {
        "iterations": [100],
        "learning_rate": [0.1],
        "depth": [3],
    },
    "Naïve-Bayes": {
        "alpha": [0.1, 1.0],
    },
    "MLP Classifier": {
        "hidden_layer_sizes": [(50,)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.0001],
    },
}

# =======================
# 5. Train + Evaluate
# =======================
results = []
for name, model in models.items():
    print(f"Tuning and evaluating {name}...")

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist[name],
        n_iter=3,  # Reduced number of iterations
        cv=3,      # Reduced number of cross-validation folds
        scoring="accuracy",
        n_jobs=2,  # Limit parallel jobs
        random_state=42,
        error_score="raise",
    )

    # Fit model
    random_search.fit(X_train_tfidf, y_train)

    best_model = random_search.best_estimator_

    # Predictions
    y_train_pred = best_model.predict(X_train_tfidf)
    y_test_pred = best_model.predict(X_test_tfidf)

    # Metrics
    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Train Precision": precision_score(y_train, y_train_pred, average="weighted"),
        "Test Precision": precision_score(y_test, y_test_pred, average="weighted"),
        "Train Recall": recall_score(y_train, y_train_pred, average="weighted"),
        "Test Recall": recall_score(y_test, y_test_pred, average="weighted"),
        "Train F1-Score": f1_score(y_train, y_train_pred, average="weighted"),
        "Test F1-Score": f1_score(y_test, y_test_pred, average="weighted"),
    })

# =======================
# 6. Results Table
# =======================
results_df = pd.DataFrame(results).set_index("Model")
print("\n--- Performance Metrics Comparison ---")
print(results_df)
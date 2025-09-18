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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Load and Preprocess Data
try:
    df = pd.read_csv('original.csv')
except FileNotFoundError:
    print("Error: 'original.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Drop rows with missing text or fluency data
df.dropna(subset=['teacher', 'student', 'fluency'], inplace=True)
# Filter out rows where fluency is 'NA'
df = df[df['fluency'] != 'NA']

# Combine teacher and student text into a single feature
df['text'] = df['teacher'].astype(str) + ' ' + df['student'].astype(str)

# Define features (X) and target (y)
X = df['text']
y = df['fluency']

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)

# 2. Vectorize Text Data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 3. Define Models and Hyperparameter Grids for RandomizedSearchCV
models = {
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    "CatBoost": cb.CatBoostClassifier(random_state=42, verbose=0),
    "Naïve-Bayes": MultinomialNB(),
    "MLP Classifier": MLPClassifier(random_state=42, max_iter=500)
}

param_dist = {
    "Support Vector Machine": {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    },
    "Decision Tree": {
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Random Forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    "CatBoost": {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7]
    },
    "Naïve-Bayes": {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
    },
    "MLP Classifier": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }
}


# 4. Perform Hyperparameter Tuning and Evaluate Models
results = []
for name, model in models.items():
    print(f"Tuning and evaluating {name}...")

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist[name],
        n_iter=10,  # Number of parameter settings that are sampled
        cv=5,       # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        error_score='raise'
    )

    # Fit the model
    random_search.fit(X_train_tfidf, y_train)

    # Get the best model
    best_model = random_search.best_estimator_

    # Make predictions on training and testing data
    y_train_pred = best_model.predict(X_train_tfidf)
    y_test_pred = best_model.predict(X_test_tfidf)

    # Calculate performance metrics
    # Using 'weighted' average for multiclass classification to handle class imbalance
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')

    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Store results
    results.append({
        "Model": name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Precision": train_precision,
        "Test Precision": test_precision,
        "Train Recall": train_recall,
        "Test Recall": test_recall,
        "Train F1-Score": train_f1,
        "Test F1-Score": test_f1
    })

# 5. Tabulate and Display Results
results_df = pd.DataFrame(results)
results_df.set_index("Model", inplace=True)

print("\n--- Performance Metrics Comparison ---")
print(results_df)
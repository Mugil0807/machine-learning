# 1. Import Necessary Libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load and Prepare Your Data
# For this example, we use the Iris dataset.
# Replace this with your own project's data loading and preprocessing steps.
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Model and Hyperparameter Grid
# Instantiate the classifier you want to tune.
# This could be any of the models mentioned in your lab (SVM, XGBoost, etc.)
model = RandomForestClassifier(random_state=42)

# Define the hyperparameter distribution to search over.
# These are the settings that RandomizedSearchCV will randomly sample from.
param_distributions = {
    'n_estimators': np.arange(10, 201, 10),  # Number of trees in the forest
    'max_features': ['sqrt', 'log2', None],      # Number of features to consider at every split
    'max_depth': list(np.arange(10, 111, 10)) + [None], # Maximum depth of the tree
    'min_samples_split': np.arange(2, 11),       # Minimum number of samples required to split a node
    'min_samples_leaf': np.arange(1, 5),         # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                   # Method of sampling data points (with or without replacement)
}

# 4. Set Up and Run RandomizedSearchCV
# [cite_start]This will search for the best hyperparameters using cross-validation. [cite: 23]
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=100,  # Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    cv=5,        # 5-fold cross-validation
    verbose=2,   # Prints updates as it runs
    random_state=42, # For reproducibility
    n_jobs=-1    # Use all available CPU cores
)

print("Starting hyperparameter tuning with RandomizedSearchCV...")
# Fit the random search object to the training data
random_search.fit(X_train, y_train)
print("Tuning finished.")

# 5. Display the Results and Evaluate the Best Model
print("\n--- Hyperparameter Tuning Results ---")

# Print the best combination of hyperparameters found
print("Best Parameters Found:")
print(random_search.best_params_)

# Print the cross-validation score of the best estimator
print("\nBest Cross-Validation Score:")
print(f"{random_search.best_score_:.4f}")

# The best model is automatically refit on the entire training data, so we can use it directly
best_model = random_search.best_estimator_

# Evaluate the best model on the test set
print("\n--- Performance on Test Set ---")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy:.4f}")

print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))
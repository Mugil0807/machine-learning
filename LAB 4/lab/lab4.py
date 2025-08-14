# Import necessary libraries
import pandas as pd  # For data manipulation and analysis (e.g., DataFrames)
import numpy as np  # For numerical operations, especially with arrays
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier  # The K-Nearest Neighbors classification algorithm
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score  # For evaluating model performance
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert text data into numerical vectors
from sklearn.linear_model import LinearRegression  # The Linear Regression algorithm
import matplotlib.pyplot as plt  # For creating plots and visualizations

# === A1: Load and process original.csv for KNN Classification ===
# Safely load the CSV, handling potential encoding issues and bad lines.
df = pd.read_csv("original.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")
# Clean up column names by removing leading/trailing whitespace and converting to lowercase.
df.columns = df.columns.str.strip().str.lower()
# Remove rows where essential columns ('teacher', 'student', 'fluency') have missing values.
df.dropna(subset=['teacher', 'student', 'fluency'], inplace=True)

# Check if the DataFrame is empty after dropping NA values.
if df.empty:
    print(" Warning: The dataframe from original.csv is empty after cleaning. Skipping A1.")
else:
    # Standardize the 'fluency' column for consistency.
    df['fluency'] = df['fluency'].astype(str).str.strip().str.lower()
    # Create a dictionary to correct common typos and abbreviations in the 'fluency' column.
    fluency_corrections = {
        'hgih': 'high', 'mediu': 'medium', 'na': 'medium',
        'med': 'medium', 'hi': 'high', 'lo': 'low'
    }
    # Apply the corrections.
    df['fluency'] = df['fluency'].replace(fluency_corrections)
    # Filter the DataFrame to only include rows with the three target fluency levels.
    df = df[df['fluency'].isin(['high', 'medium', 'low'])]

    # Proceed only if there's a sufficient amount of data (at least 5 rows).

    # Proceed only if there's a sufficient amount of data (at least 5 rows).
    if not df.empty and len(df) >= 5:
        # Combine 'teacher' and 'student' text columns into a single feature for vectorization.
        df['combined'] = df['teacher'].astype(str) + " " + df['student'].astype(str)
        
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english', lowercase=True)
        
        ## FIX: Rename variables to distinguish them from the regression task
        X_class = vectorizer.fit_transform(df['combined']).toarray()
        y_class = pd.Categorical(df['fluency'], categories=['high', 'medium', 'low'], ordered=True).codes

        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.3, stratify=y_class, random_state=42
        )
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_class, y_train_class)

        y_pred_train = knn.predict(X_train_class)
        y_pred_test = knn.predict(X_test_class)

        print("Train Confusion Matrix:\n", confusion_matrix(y_train_class, y_pred_train))
        print("Test Confusion Matrix:\n", confusion_matrix(y_test_class, y_pred_test))
        print("\nTrain Classification Report:\n", classification_report(y_train_class, y_pred_train))
        print("Test Classification Report:\n", classification_report(y_test_class, y_pred_test))
    else:
        print(" Not enough valid rows in original.csv for A1.")

# === A2: Regression Metrics ===
df_stock = pd.read_csv("IRCTC Stock Price.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")

def parse_volume(val):
    if isinstance(val, str):
        val = val.replace(',', '')
        if 'M' in val.upper():
            return float(val.upper().replace('M', '')) * 1e6
        elif 'K' in val.upper():
            return float(val.upper().replace('K', '')) * 1e3
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

if df_stock.empty:
    print(" Warning: IRCTC Stock Price.csv is empty. Skipping A2.")
else:
    ## Standardize column names to remove whitespace and convert to lowercase.
    ## This is the most important fix to prevent columns from being missed.
    df_stock.columns = df_stock.columns.str.strip().str.lower()

    # Define the columns we need to work with using their new lowercase names.
    price_cols = ['price', 'open', 'high', 'low']
    
    ## Make numeric conversion more robust by first removing commas.
    for col in price_cols:
        if col in df_stock.columns:
            # First, remove any commas from the string representation of the column.
            df_stock[col] = df_stock[col].astype(str).str.replace(',', '')
            # Then, convert to a numeric type, turning errors into NaN.
            df_stock[col] = pd.to_numeric(df_stock[col], errors='coerce')

    if 'volume' in df_stock.columns:
        df_stock['volume'] = df_stock['volume'].apply(parse_volume)

    # Now, drop rows with NaN in any of the essential columns.
    required_cols = ['price', 'open', 'high', 'low', 'volume']
    df_stock.dropna(subset=required_cols, inplace=True)

    if not df_stock.empty and len(df_stock) >= 5:
        ## FIX: Use the new lowercase column names for defining X and y.
        X = df_stock[['open', 'high', 'low', 'volume']]
        y = df_stock['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Linear Regression Performance Metrics:")
        print(f"MSE  = {mse:.2f}")
        print(f"RMSE = {rmse:.2f}")
        print(f"MAPE = {mape:.4f}")
        print(f"R²   = {r2:.4f}")

        coeff_df = pd.DataFrame(model.coef_, index=X.columns, columns=["Coefficient"])
        print("\n Feature Coefficients:")
        print(coeff_df)

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, edgecolors='k', alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title("Actual vs Predicted Prices")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        # This warning will now only show if the file is truly empty or lacks valid data.
        print(" Not enough valid rows in IRCTC Stock Price.csv for A2 after cleaning.")

# === A3: Generate and Visualize Synthetic Training Data ===
np.random.seed(0)  # Set a seed for reproducibility of random numbers.
# Create a synthetic dataset of 20 points, each with 2 features, ranging from 1 to 10.
X_train_synthetic = np.random.uniform(1, 10, (20, 2))
# Create a target array with two classes (10 points in class 0, 10 in class 1).
y_train_synthetic = np.array([0]*10 + [1]*10)

# Create a scatter plot of the synthetic data.
# 'c=y_train_synthetic' colors the points based on their class.
# 'cmap='bwr'' uses a blue-white-red colormap.
plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='bwr', s=60, edgecolors='k')
plt.title("A3: Synthetic Training Data (20 points)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

# === A4: Visualize KNN Class Boundaries for k=3 ===
# Create a grid of points that covers the entire feature space to visualize the decision boundary.
x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_vals, y_vals)
# Flatten the grid into a list of points to be fed into the classifier.
test_points = np.c_[xx.ravel(), yy.ravel()]

# Train a KNN classifier with k=3 on the synthetic data.
knn_synthetic = KNeighborsClassifier(n_neighbors=3)
knn_synthetic.fit(X_train_synthetic, y_train_synthetic)
# Predict the class for every point in the mesh grid.
Z = knn_synthetic.predict(test_points).reshape(xx.shape)

# Plot the decision boundary using a filled contour plot.
plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
# Overlay the original training points.
plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='bwr', edgecolors='k')
plt.title("A4: Class Boundaries using k=3")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

# === A5: Repeat Boundary Visualization for Different k Values ===
# Loop through different values of k to see how it affects the decision boundary.
for k in [1, 3, 5, 7]:
    # Train a new KNN model for the current value of k.
    knn_varied = KNeighborsClassifier(n_neighbors=k)
    knn_varied.fit(X_train_synthetic, y_train_synthetic)
    # Predict on the same mesh grid.
    Z = knn_varied.predict(test_points).reshape(xx.shape)

    # Plot the decision boundary and the training points.
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='bwr', edgecolors='k')
    plt.title(f"A5: Class Boundaries using k={k}")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()
    # Note: Smaller k (e.g., k=1) leads to a more complex boundary (overfitting).
    # Larger k (e.g., k=7) leads to a smoother, more generalized boundary (underfitting).

# === A6: Visualize Project Data in 2D (or Fallback to Synthetic) ===
# Use a try-except block to handle cases where the project data from A1 is not available.
try:
    # Check if the dataframe from A1 has enough data.
    if not df.empty and len(df) >= 5:
        # Select the first two TF-IDF features for a 2D plot.
        X_proj_2d = X[:, :2]
        # Ensure y is a NumPy array for consistent plotting.
        y_proj = y

        # Create a scatter plot of the first two features from the project data.
        plt.scatter(X_proj_2d[:, 0], X_proj_2d[:, 1], c=y_proj, cmap='coolwarm', edgecolors='k')
        plt.title("A6: Project Data using Two Features")
        plt.xlabel("TF-IDF Feature 1")
        plt.ylabel("TF-IDF Feature 2")
        plt.grid(True)
        plt.show()
    else:
        # If not enough data, raise an error to trigger the 'except' block.
        raise ValueError("Fallback to synthetic data")
except:
    # If the 'try' block fails, plot the synthetic data as a fallback example.
    plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='coolwarm', edgecolors='k')
    plt.title("A6: Synthetic Data Fallback")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


# === A7: Hyperparameter Tuning ===
params = {'n_neighbors': list(range(1, 11))}
grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
if not df.empty and len(df) >= 5:
    ## FIX: Use the classification data (_class variables) to fit the grid search.
    grid.fit(X_train_class, y_train_class)
    
    print(f"\nBest k: {grid.best_params_['n_neighbors']}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
else:
    print(" Skipping A7 because not enough project data.")

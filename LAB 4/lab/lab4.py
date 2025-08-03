import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Load original.csv safely ===
df = pd.read_csv("original.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")
df.columns = df.columns.str.strip().str.lower()
df.dropna(subset=['teacher', 'student', 'fluency'], inplace=True)

if df.empty:
    print("⚠️ Warning: The dataframe from original.csv is empty after cleaning. Skipping A1.")
else:
    df['fluency'] = df['fluency'].astype(str).str.strip().str.lower()
    fluency_corrections = {
        'hgih': 'high', 'mediu': 'medium', 'na': 'medium',
        'med': 'medium', 'hi': 'high', 'lo': 'low'
    }
    df['fluency'] = df['fluency'].replace(fluency_corrections)
    df = df[df['fluency'].isin(['high', 'medium', 'low'])]

    if not df.empty and len(df) >= 5:
        df['combined'] = df['teacher'].astype(str) + " " + df['student'].astype(str)
        vectorizer = TfidfVectorizer(max_features=200, stop_words='english', lowercase=True)
        X = vectorizer.fit_transform(df['combined']).toarray()
        y = pd.Categorical(df['fluency'], categories=['high', 'medium', 'low']).codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)

        y_pred_train = knn.predict(X_train)
        y_pred_test = knn.predict(X_test)

        print("Train Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
        print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
        print("\nTrain Classification Report:\n", classification_report(y_train, y_pred_train))
        print("Test Classification Report:\n", classification_report(y_test, y_pred_test))
    else:
        print("⚠️ Not enough valid rows in original.csv for A1.")

# === A2: Regression Metrics ===
df_stock = pd.read_csv("IRCTC Stock Price.csv", encoding="ISO-8859-1", engine="python", on_bad_lines="skip")

def parse_volume(val):
    if isinstance(val, str):
        val = val.replace(',', '')
        if 'M' in val:
            return float(val.replace('M', '')) * 1e6
        elif 'K' in val:
            return float(val.replace('K', '')) * 1e3
    try:
        return float(val)
    except:
        return np.nan

if df_stock.empty:
    print("⚠️ Warning: IRCTC Stock Price.csv is empty. Skipping A2.")
else:
    df_stock['Price'] = pd.to_numeric(df_stock['Price'], errors='coerce')
    df_stock['Open'] = pd.to_numeric(df_stock['Open'], errors='coerce')
    df_stock['High'] = pd.to_numeric(df_stock['High'], errors='coerce')
    df_stock['Low'] = pd.to_numeric(df_stock['Low'], errors='coerce')
    df_stock['Volume'] = df_stock['Volume'].apply(parse_volume)

    df_stock.dropna(subset=['Price', 'Open', 'High', 'Low', 'Volume'], inplace=True)

    if not df_stock.empty and len(df_stock) >= 5:
        X = df_stock[['Open', 'High', 'Low', 'Volume']]
        y = df_stock['Price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("✅ Linear Regression Performance Metrics:")
        print(f"MSE  = {mse:.2f}")
        print(f"RMSE = {rmse:.2f}")
        print(f"MAPE = {mape:.4f}")
        print(f"R²   = {r2:.4f}")

        coeff_df = pd.DataFrame(model.coef_, index=X.columns, columns=["Coefficient"])
        print("\n🔍 Feature Coefficients:")
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
        print("⚠️ Not enough valid rows in IRCTC Stock Price.csv for A2.")

# === A3: Synthetic Training Data ===
np.random.seed(0)
X_train_synthetic = np.random.uniform(1, 10, (20, 2))
y_train_synthetic = np.array([0]*10 + [1]*10)

plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='bwr', s=60, edgecolors='k')
plt.title("A3: Synthetic Training Data (20 points)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

# === A4: Class Boundaries using k=3 ===
x_vals = np.arange(0, 10.1, 0.1)
y_vals = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_vals, y_vals)
test_points = np.c_[xx.ravel(), yy.ravel()]

knn_synthetic = KNeighborsClassifier(n_neighbors=3)
knn_synthetic.fit(X_train_synthetic, y_train_synthetic)
Z = knn_synthetic.predict(test_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='bwr', edgecolors='k')
plt.title("A4: Class Boundaries using k=3")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.grid(True)
plt.show()

# === A5: Repeat with different k values ===
for k in [1, 3, 5, 7]:
    knn_varied = KNeighborsClassifier(n_neighbors=k)
    knn_varied.fit(X_train_synthetic, y_train_synthetic)
    Z = knn_varied.predict(test_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    plt.scatter(X_train_synthetic[:, 0], X_train_synthetic[:, 1], c=y_train_synthetic, cmap='bwr', edgecolors='k')
    plt.title(f"A5: Class Boundaries using k={k}")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()

# === A6: Use 2 Features from Project Data (fallback to synthetic if not enough rows) ===
try:
    if not df.empty and len(df) >= 5:
        X_proj_2d = X.values[:, :2]
        y_proj = y.values if hasattr(y, 'values') else y

        plt.scatter(X_proj_2d[:, 0], X_proj_2d[:, 1], c=y_proj, cmap='coolwarm', edgecolors='k')
        plt.title("A6: Project Data using Two Features")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()
    else:
        raise ValueError("Fallback to synthetic")
except:
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
    grid.fit(X_train, y_train)
    print(f"Best k: {grid.best_params_['n_neighbors']}")
    print(f"Best cross-validation accuracy: {grid.best_score_:.4f}")
else:
    print("⚠️ Skipping A7 because not enough project data.")

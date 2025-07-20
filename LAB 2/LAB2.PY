

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
import statistics
import warnings
warnings.filterwarnings('ignore')

excel_file = 'Lab Session Data.xlsx'
xls = pd.ExcelFile("Lab Session Data.xlsx")


# A1: Linear Algebra Analysis
def analyze_linear_system():
    df = pd.read_excel(excel_file, sheet_name='Purchase data')
    print(xls.sheet_names)
    print(df.columns.tolist())
    A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    C = df['Payment (Rs)'].values
    dimensionality = A.shape[1]
    num_vectors = A.shape[0]
    rank_A = np.linalg.matrix_rank(A)
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C
    return A, C, X, dimensionality, num_vectors, rank_A

# A2: Customer Classification
def classify_customers():
    df = pd.read_excel(excel_file, sheet_name='Purchase data')
    df['Class'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    y = df['Class'].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = LogisticRegression()
    model.fit(X, y_encoded)
    acc = accuracy_score(y_encoded, model.predict(X))
    return acc, df['Class'].value_counts()

# A3: Stock Price Analysis
def analyze_stock_data():
    df = pd.read_excel(excel_file, sheet_name='IRCTC Stock Price')
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.day_name()
    mean_price = statistics.mean(df['Price'])
    var_price = statistics.variance(df['Price'])
    wed = df[df['DayOfWeek'] == 'Wednesday']
    apr = df[df['Date'].dt.month == 4]
    wed_mean = statistics.mean(wed['Price'])
    apr_mean = statistics.mean(apr['Price'])
    prob_loss = (df['Chg%'] < 0).mean()
    prob_profit_wed = (wed['Chg%'] > 0).mean()
    return df, mean_price, var_price, wed_mean, apr_mean, prob_loss, prob_profit_wed

# A4: Data Exploration
def explore_thyroid_data():
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    types = {col: 'Categorical' if df[col].dtype == 'object' else 'Numerical' for col in df.columns}
    missing = df.isnull().sum()
    outliers = {}
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    stats = df.describe()
    return df, types, missing, outliers, stats

# A5: Similarity Measures
def calculate_similarity_measures():
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    bin_df = df.select_dtypes(include='number').fillna(0).astype(int)
    v1 = bin_df.iloc[0].astype(bool).astype(int).values
    v2 = bin_df.iloc[1].astype(bool).astype(int).values
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) else 0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    return jc, smc

# A6: Cosine Similarity
def calculate_cosine_similarity():
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    num_df = df.select_dtypes(include='number').fillna(0)
    v1 = num_df.iloc[0].values
    v2 = num_df.iloc[1].values
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_sim_scipy = 1 - cosine(v1, v2)
    return cos_sim, cos_sim_scipy

# A7: Heatmap Plot
def create_similarity_heatmap():
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    bin_df = df.select_dtypes(include='number').fillna(0).astype(bool).astype(int)
    data = bin_df.iloc[:20].values
    n = data.shape[0]
    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))
    cos_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            v1, v2 = data[i], data[j]
            f11 = np.sum((v1 == 1) & (v2 == 1))
            f00 = np.sum((v1 == 0) & (v2 == 0))
            f01 = np.sum((v1 == 0) & (v2 == 1))
            f10 = np.sum((v1 == 1) & (v2 == 0))
            jc_matrix[i, j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) else 0
            smc_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11)
            cos_matrix[i, j] = 1 - cosine(v1, v2)

    return jc_matrix, smc_matrix, cos_matrix

# A8: Data Imputation
def impute_missing_values():
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    for col in df.select_dtypes(include='number').columns:
        if df[col].isnull().sum() > 0:
            if df[col].skew() < 1:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# A9: Data Normalization
def normalize_data():
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    num_df = df.select_dtypes(include='number').fillna(0)
    scaler_std = StandardScaler()
    scaler_mm = MinMaxScaler()
    std_scaled = pd.DataFrame(scaler_std.fit_transform(num_df), columns=num_df.columns)
    mm_scaled = pd.DataFrame(scaler_mm.fit_transform(num_df), columns=num_df.columns)
    return std_scaled, mm_scaled


# Optional O1: Use two square matrices and compare X values
def optional_square_matrix_experiments():
    df = pd.read_excel(excel_file, sheet_name='Purchase data')
    full_A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    C = df['Payment (Rs)'].values

    A1 = full_A[:3, :]
    C1 = C[:3]
    X1 = np.linalg.pinv(A1) @ C1

    A2 = full_A[-3:, :]
    C2 = C[-3:]
    X2 = np.linalg.pinv(A2) @ C2

    X_full = np.linalg.pinv(full_A) @ C
    return X1, X2, X_full

# Optional O2: Random sample analysis for A5-A6

def optional_random_sampling_analysis(sample_size=20, random_state=42):
    df = pd.read_excel(excel_file, sheet_name='thyroid0387_UCI')
    sampled_df = df.sample(n=sample_size, random_state=random_state)
    bin_df = sampled_df.select_dtypes(include='number').fillna(0).astype(bool).astype(int)
    num_df = sampled_df.select_dtypes(include='number').fillna(0)

    v1, v2 = bin_df.iloc[0], bin_df.iloc[1]
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) else 0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)

    v1_full = num_df.iloc[0].values
    v2_full = num_df.iloc[1].values
    cosine_sim = 1 - cosine(v1_full, v2_full)

    return {"JC": jc, "SMC": smc, "Cosine": cosine_sim}

# Optional O3: Apply thyroid-style analysis to marketing_campaign

def optional_marketing_campaign_analysis():
    df = pd.read_excel(excel_file, sheet_name='marketing_campaign')
    data_types = {col: 'Categorical' if df[col].dtype == 'object' else 'Numerical' for col in df.columns}
    missing = df.isnull().sum()
    stats = df.describe(include='all')
    return {"data_types": data_types, "missing_values": missing, "statistics": stats}

# Main execution
if __name__ == "__main__":
    print("Lab Session 02 - Data Science and Linear Algebra")
    print("="*60)

    # A1: Linear Algebra
    A, C, X, dim, num_vecs, rank_A = analyze_linear_system()
    print(f"\nA1. Linear Algebra")
    print(f"Dimensionality: {dim}")
    print(f"Number of vectors: {num_vecs}")
    print(f"Rank of matrix A: {rank_A}")
    print(f"Estimated product costs (Candy, Mango, Milk): {X.round(2)}")

    # A2: Classification
    acc, class_dist = classify_customers()
    print("\nA2. Classification Results")
    print(f"Accuracy: {acc:.2f}")
    print("Class Distribution:")
    print(class_dist)

    # A3: Stock Price Analysis
    stock_df, mp, vp, wm, am, pl, ppw = analyze_stock_data()
    print("\nA3. Stock Price Analysis")
    print(f"Mean Price: {mp:.2f}, Variance: {vp:.2f}")
    print(f"Wednesday Mean Price: {wm:.2f}, April Mean Price: {am:.2f}")
    print(f"Probability of Loss: {pl:.3f}, Profit on Wednesday: {ppw:.3f}")
    
    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stock_df, x='DayOfWeek', y='Chg%', alpha=0.6)
    plt.xticks(rotation=45)
    plt.title("Chg% vs Day of Week")
    plt.tight_layout()
    plt.show()

    # A4: Data Exploration
    thyroid_df, types, missing, outliers, stats = explore_thyroid_data()
    print("\nA4. Data Exploration")
    print("Data Types:", types)
    print("Missing Values:\n", missing)
    print("Outliers Detected:\n", outliers)
    print("Statistical Summary:\n", stats)

    # A5: Similarity Measures
    jc, smc = calculate_similarity_measures()
    print("\nA5. Similarity Measures")
    print(f"Jaccard Coefficient: {jc:.3f}")
    print(f"Simple Matching Coefficient: {smc:.3f}")

    # A6: Cosine Similarity
    cos_manual, cos_scipy = calculate_cosine_similarity()
    print("\nA6. Cosine Similarity")
    print(f"Cosine Similarity (Manual): {cos_manual:.3f}")
    print(f"Cosine Similarity (Scipy): {cos_scipy:.3f}")

    # A7: Heatmap Plot
    print("\nA7. Similarity Heatmaps")
    jc_mat, smc_mat, cos_mat = create_similarity_heatmap()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(jc_mat, annot=True, cmap='Blues', ax=axes[0])
    axes[0].set_title("Jaccard Coefficient")
    sns.heatmap(smc_mat, annot=True, cmap='Greens', ax=axes[1])
    axes[1].set_title("Simple Matching Coefficient")
    sns.heatmap(cos_mat, annot=True, cmap='Oranges', ax=axes[2])
    axes[2].set_title("Cosine Similarity")
    plt.tight_layout()
    plt.show()

    # A8: Data Imputation
    imputed_df = impute_missing_values()
    print("\nA8. Missing Values After Imputation:")
    print(imputed_df.isnull().sum())

    # A9: Normalization
    std_scaled, mm_scaled = normalize_data()
    print("\nA9. Normalized Data Statistics")
    print("Standard Scaling:\n", std_scaled.describe())
    print("MinMax Scaling:\n", mm_scaled.describe())
    
    #optional:1
    X1, X2, X_full = optional_square_matrix_experiments()
    print("\nO1. Square Matrix Comparison")
    print("X1 (first 3 rows):", X1.round(2))
    print("X2 (last 3 rows):", X2.round(2))
    print("X_full (all rows):", X_full.round(2))
    
    #optional:2
    o2_result = optional_random_sampling_analysis()
    print("\nO2. Random Sample Similarity")
    print(o2_result)

    #optional:3
    o3_result = optional_marketing_campaign_analysis()
    print("\nO3. Marketing Campaign Data Analysis")
    print("Data Types:\n", o3_result['data_types'])
    print("Missing Values:\n", o3_result['missing_values'])
    print("Statistics:\n", o3_result['statistics'])


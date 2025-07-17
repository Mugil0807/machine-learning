import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import statistics
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Load the data
def load_purchase_data():
    """Load and prepare the purchase data"""
    # Load Purchase Data worksheet
    df = pd.read_csv('Lab Session Data.csv', sheet_name='Purchase Data')
    return df

# A1. Linear Algebra Analysis
def analyze_linear_system():
    """Analyze the purchase data using linear algebra concepts"""
    df = load_purchase_data()
    
    # Create matrices A and C for AX = C
    A = df[['Candies', 'Mangoes', 'Milk_Packets']].values
    C = df['Payment'].values
    
    print("A1. Linear Algebra Analysis")
    print("="*50)
    
    # Dimensionality of vector space
    dimensionality = A.shape[1]
    print(f"Dimensionality of vector space: {dimensionality}")
    
    # Number of vectors
    num_vectors = A.shape[0]
    print(f"Number of vectors in vector space: {num_vectors}")
    
    # Rank of matrix A
    rank_A = np.linalg.matrix_rank(A)
    print(f"Rank of Matrix A: {rank_A}")
    
    # Using pseudo-inverse to find cost of each product
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C
    
    print(f"\nCost of each product:")
    print(f"Candy cost: Rs. {X[0]:.2f}")
    print(f"Mango cost: Rs. {X[1]:.2f}")
    print(f"Milk packet cost: Rs. {X[2]:.2f}")
    
    return A, C, X

# A2. Classification Model
def classify_customers():
    """Classify customers as RICH or POOR based on payment"""
    df = load_purchase_data()
    
    print("\nA2. Customer Classification")
    print("="*50)
    
    # Create labels: RICH (>200) vs POOR (<=200)
    df['Class'] = df['Payment'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    
    # Prepare features and target
    X = df[['Candies', 'Mangoes', 'Milk_Packets']].values
    y = df['Class'].values
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X, y_encoded)
    
    # Make predictions
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_encoded, y_pred)
    
    print(f"Classification Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Class distribution:")
    print(df['Class'].value_counts())
    
    return model, le

# A3. Stock Price Analysis
def analyze_stock_data():
    """Analyze stock price data"""
    # Load IRCTC Stock Price worksheet
    stock_data = pd.read_csv('IRCTC Stock Price.csv', sheet_name='IRCTC Stock Price')
    
    # Convert Date column to datetime if needed
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['DayOfWeek'] = stock_data['Date'].dt.day_name()
    
    print("\nA3. Stock Price Analysis")
    print("="*50)
    
    # Calculate mean and variance
    mean_price = statistics.mean(stock_data['Price'])
    var_price = statistics.variance(stock_data['Price'])
    print(f"Population mean: {mean_price:.2f}")
    print(f"Population variance: {var_price:.2f}")
    
    # Wednesday analysis
    wed_data = stock_data[stock_data['DayOfWeek'] == 'Wednesday']
    if len(wed_data) > 0:
        wed_mean = statistics.mean(wed_data['Price'])
        print(f"Wednesday sample mean: {wed_mean:.2f}")
        print(f"Difference from population mean: {wed_mean - mean_price:.2f}")
    
    # Probability of loss
    losses = stock_data['Chg%'].apply(lambda x: x < 0)
    prob_loss = losses.mean()
    print(f"Probability of making a loss: {prob_loss:.3f}")
    
    # Probability of profit on Wednesday
    if len(wed_data) > 0:
        wed_profits = wed_data['Chg%'].apply(lambda x: x > 0)
        prob_profit_wed = wed_profits.mean()
        print(f"Probability of profit on Wednesday: {prob_profit_wed:.3f}")
    
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(stock_data['DayOfWeek'], stock_data['Chg%'], alpha=0.6)
    plt.xlabel('Day of Week')
    plt.ylabel('Change %')
    plt.title('Stock Change % vs Day of Week')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return stock_data

# A4. Data Exploration
def explore_thyroid_data():
    """Explore thyroid dataset"""
    # Load thyroid0387_UCI worksheet
    thyroid_data = pd.read_csv('thyroid0387_UCI.csv', sheet_name='thyroid0387_UCI')
    
    print("\nA4. Data Exploration")
    print("="*50)
    
    print("Dataset Info:")
    print(thyroid_data.info())
    print("\nData Types:")
    for col in thyroid_data.columns:
        if thyroid_data[col].dtype == 'object':
            print(f"{col}: Categorical (Nominal)")
        else:
            print(f"{col}: Numerical")
    
    print("\nMissing Values:")
    print(thyroid_data.isnull().sum())
    
    print("\nNumerical Variables Summary:")
    print(thyroid_data.describe())
    
    # Outlier detection using IQR
    print("\nOutlier Detection (IQR method):")
    for col in thyroid_data.select_dtypes(include=[np.number]).columns:
        Q1 = thyroid_data[col].quantile(0.25)
        Q3 = thyroid_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = thyroid_data[(thyroid_data[col] < lower_bound) | (thyroid_data[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers")
    
    return thyroid_data

# A5. Similarity Measures
def calculate_similarity_measures():
    """Calculate Jaccard and Simple Matching Coefficients"""
    thyroid_data = pd.read_csv('thyroid0387_UCI.csv', sheet_name='thyroid0387_UCI')

    # Create binary vectors for demonstration
    vector1 = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    vector2 = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    
    print("\nA5. Similarity Measures")
    print("="*50)
    
    # Calculate frequency counts
    f11 = np.sum((vector1 == 1) & (vector2 == 1))  # Both 1
    f00 = np.sum((vector1 == 0) & (vector2 == 0))  # Both 0
    f01 = np.sum((vector1 == 0) & (vector2 == 1))  # First 0, Second 1
    f10 = np.sum((vector1 == 1) & (vector2 == 0))  # First 1, Second 0
    
    print(f"f11 (both 1): {f11}")
    print(f"f00 (both 0): {f00}")
    print(f"f01 (0,1): {f01}")
    print(f"f10 (1,0): {f10}")
    
    # Jaccard Coefficient
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0
    
    # Simple Matching Coefficient
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
    
    print(f"\nJaccard Coefficient (JC): {jc:.3f}")
    print(f"Simple Matching Coefficient (SMC): {smc:.3f}")
    
    print(f"\nAnalysis:")
    print(f"JC focuses on positive matches, ignoring negative matches")
    print(f"SMC considers both positive and negative matches")
    
    return jc, smc

# A6. Cosine Similarity
def calculate_cosine_similarity():
    """Calculate cosine similarity between two vectors"""
    thyroid_data = pd.read_csv('thyroid0387_UCI.csv', sheet_name='thyroid0387_UCI')

    # Example vectors
    vector1 = np.array([1, 2, 3, 4, 5])
    vector2 = np.array([2, 3, 4, 5, 6])
    
    print("\nA6. Cosine Similarity")
    print("="*50)
    
    # Calculate cosine similarity
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    cos_sim = dot_product / (norm1 * norm2)
    
    print(f"Vector 1: {vector1}")
    print(f"Vector 2: {vector2}")
    print(f"Cosine Similarity: {cos_sim:.3f}")
    
    # Alternative using scipy
    cos_sim_scipy = 1 - cosine(vector1, vector2)
    print(f"Cosine Similarity (scipy): {cos_sim_scipy:.3f}")
    
    return cos_sim

# A7. Heatmap Plot
def create_similarity_heatmap():
    """Create heatmap of similarity measures"""
    thyroid_data = pd.read_csv('thyroid0387_UCI.csv', sheet_name='thyroid0387_UCI')

    # Generate sample data
    np.random.seed(42)
    n_vectors = 20
    n_features = 8
    
    # Create binary vectors
    data = np.random.randint(0, 2, (n_vectors, n_features))
    
    print("\nA7. Heatmap Plot")
    print("="*50)
    
    # Calculate similarity matrices
    jc_matrix = np.zeros((n_vectors, n_vectors))
    smc_matrix = np.zeros((n_vectors, n_vectors))
    cos_matrix = np.zeros((n_vectors, n_vectors))
    
    for i in range(n_vectors):
        for j in range(n_vectors):
            if i == j:
                jc_matrix[i, j] = 1.0
                smc_matrix[i, j] = 1.0
                cos_matrix[i, j] = 1.0
            else:
                # Jaccard and SMC for binary data
                v1, v2 = data[i], data[j]
                f11 = np.sum((v1 == 1) & (v2 == 1))
                f00 = np.sum((v1 == 0) & (v2 == 0))
                f01 = np.sum((v1 == 0) & (v2 == 1))
                f10 = np.sum((v1 == 1) & (v2 == 0))
                
                jc_matrix[i, j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0
                smc_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11)
                
                # Cosine similarity
                cos_matrix[i, j] = 1 - cosine(v1, v2) if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0 else 0
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.heatmap(jc_matrix, annot=True, fmt='.2f', cmap='viridis', ax=axes[0])
    axes[0].set_title('Jaccard Coefficient')
    
    sns.heatmap(smc_matrix, annot=True, fmt='.2f', cmap='viridis', ax=axes[1])
    axes[1].set_title('Simple Matching Coefficient')
    
    sns.heatmap(cos_matrix, annot=True, fmt='.2f', cmap='viridis', ax=axes[2])
    axes[2].set_title('Cosine Similarity')
    
    plt.tight_layout()
    plt.show()
    
    return jc_matrix, smc_matrix, cos_matrix

# Optional: Load marketing campaign data for O3
def load_marketing_data():
    """Load marketing campaign data"""
    thyroid_data = pd.read_csv('thyroid0387_UCI.csv', sheet_name='thyroid0387_UCI')

    # Load marketing_campaign worksheet
    marketing_data = pd.read_csv('Lab Session Data.xlsx', sheet_name='marketing_campaign')
    return marketing_data
    """Impute missing values using appropriate measures"""
    # Create sample data with missing values
    np.random.seed(42)
    data = pd.DataFrame({
        'numeric_normal': np.random.normal(50, 10, 100),
        'numeric_with_outliers': np.concatenate([np.random.normal(30, 5, 90), [100, 120, 150, 200, 250, 300, 350, 400, 450, 500]]),
        'categorical': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Introduce missing values
    data.loc[np.random.choice(100, 10, replace=False), 'numeric_normal'] = np.nan
    data.loc[np.random.choice(100, 15, replace=False), 'numeric_with_outliers'] = np.nan
    data.loc[np.random.choice(100, 8, replace=False), 'categorical'] = np.nan
    
    print("\nA8. Data Imputation")
    print("="*50)
    
    print("Before imputation:")
    print(data.isnull().sum())
    
    # Impute using appropriate measures
    # Mean for normal numeric data
    data['numeric_normal'].fillna(data['numeric_normal'].mean(), inplace=True)
    
    # Median for numeric data with outliers
    data['numeric_with_outliers'].fillna(data['numeric_with_outliers'].median(), inplace=True)
    
    # Mode for categorical data
    data['categorical'].fillna(data['categorical'].mode()[0], inplace=True)
    
    print("\nAfter imputation:")
    print(data.isnull().sum())
    
    return data

# A9. Data Normalization
def normalize_data():
    """Normalize data using appropriate techniques"""
    thyroid_data = pd.read_csv('thyroid0387_UCI.csv', sheet_name='thyroid0387_UCI')

    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 50, 100),
        'feature2': np.random.exponential(2, 100),
        'feature3': np.random.uniform(0, 1000, 100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    print("\nA9. Data Normalization")
    print("="*50)
    
    print("Original data statistics:")
    print(data.describe())
    
    # Identify numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # StandardScaler (Z-score normalization)
    scaler_standard = StandardScaler()
    data_standard = data.copy()
    data_standard[numeric_cols] = scaler_standard.fit_transform(data[numeric_cols])
    
    # MinMaxScaler (0-1 normalization)
    scaler_minmax = MinMaxScaler()
    data_minmax = data.copy()
    data_minmax[numeric_cols] = scaler_minmax.fit_transform(data[numeric_cols])
    
    print("\nAfter Standard Scaling:")
    print(data_standard[numeric_cols].describe())
    
    print("\nAfter MinMax Scaling:")
    print(data_minmax[numeric_cols].describe())
    
    return data_standard, data_minmax

# Main execution
if __name__ == "__main__":
    print("Lab Session 02 - Data Science and Linear Algebra")
    print("="*60)
    
    # Execute all tasks
    A, C, X = analyze_linear_system()
    model, le = classify_customers()
    stock_data = analyze_stock_data()
    thyroid_data = explore_thyroid_data()
    jc, smc = calculate_similarity_measures()
    cos_sim = calculate_cosine_similarity()
    jc_matrix, smc_matrix, cos_matrix = create_similarity_heatmap()
    imputed_data = impute_missing_values()
    normalized_standard, normalized_minmax = normalize_data()
    
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print("="*60)
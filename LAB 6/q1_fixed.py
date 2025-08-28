import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================
# Load the dataset from the provided CSV file with the correct encoding
try:
    # This is the corrected line
    df = pd.read_csv('original.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'original.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# The rest of your code remains the same...
# Preprocessing Steps
df['teacher'] = df['teacher'].fillna('')
df['student'] = df['student'].fillna('')

# Drop rows where the target variable 'fluency' is 'NA'
df.dropna(subset=['fluency'], inplace=True)
df = df[df['fluency'] != 'NA']

# --- Feature Engineering ---
df['teacher_word_count'] = df['teacher'].apply(lambda x: len(str(x).split()))
df['student_word_count'] = df['student'].apply(lambda x: len(str(x).split()))

# ========================================================================
# A4: BINNING FUNCTION FOR CONTINUOUS DATA
# ========================================================================
def create_bins(data_series, num_bins=4, bin_type='equal_width'):
    if bin_type == 'equal_width':
        max_val, min_val = data_series.max(), data_series.min()
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins + 1)]
        bins[-1] = max_val
        labels = [f'Bin_{i+1}' for i in range(num_bins)]
        return pd.cut(data_series, bins=bins, labels=labels, include_lowest=True)
    elif bin_type == 'equal_frequency':
        labels = [f'Bin_{i+1}' for i in range(num_bins)]
        try:
            return pd.qcut(data_series, q=num_bins, labels=labels, duplicates='drop')
        except ValueError as e:
            print(f"Warning for {data_series.name}: Could not create {num_bins} unique bins. {e}. Falling back to equal width.")
            return create_bins(data_series, num_bins=num_bins, bin_type='equal_width')
    else:
        raise ValueError("Invalid bin_type. Choose 'equal_width' or 'equal_frequency'.")

# Apply binning
df['student_wc_binned'] = create_bins(df['student_word_count'], num_bins=3, bin_type='equal_frequency')
df['teacher_wc_binned'] = create_bins(df['teacher_word_count'], num_bins=3, bin_type='equal_frequency')

print("--- A4: Binning Example ---")
print("Binned student word counts:")
print(df[['student_word_count', 'student_wc_binned']].head())
print("\n" + "="*50 + "\n")

# ==============================================================================
# A1: ENTROPY CALCULATION
# ==============================================================================
def calculate_entropy(data_series):
    """
    Calculates the entropy for a given pandas Series.
    Formula: H = -sum(p_i * log2(p_i))
    """
    counts = Counter(data_series)
    total_count = len(data_series)
    entropy = 0.0
    for count in counts.values():
        probability = count / total_count
        if probability > 0:
            entropy -= probability * np.log2(probability)
    return entropy

# Calculate entropy of fluency
entropy_fluency = calculate_entropy(df['fluency'])
print("--- A1: Entropy Calculation ---")
print(f"The entropy of the 'fluency' column is: {entropy_fluency:.4f}")
print("\n" + "="*50 + "\n")

# ==============================================================================
# A2: GINI INDEX CALCULATION
# ==============================================================================
def calculate_gini(data_series):
    """
    Calculates the Gini index for a given pandas Series.
    Formula: Gini = 1 - sum(p_j^2)
    """
    counts = Counter(data_series)
    total_count = len(data_series)
    gini = 1.0
    for count in counts.values():
        probability = count / total_count
        gini -= probability * probability
    return gini

# Calculate Gini index of fluency
gini_fluency = calculate_gini(df['fluency'])
print("--- A2: Gini Index Calculation ---")
print(f"The Gini index for the 'fluency' column is: {gini_fluency:.4f}")
print("\n" + "="*50 + "\n")

# ==============================================================================
# A3: ROOT NODE DETECTION MODULE
# ==============================================================================
def find_root_node(dataframe, features, target_col):
    """
    Identifies the best feature for the root node using Information Gain.
    """
    parent_entropy = calculate_entropy(dataframe[target_col])
    best_info_gain = -1
    root_node = None
    
    for feature in features:
        unique_values = dataframe[feature].unique()
        info_gain = parent_entropy
        
        # Calculate weighted entropy for each value
        for value in unique_values:
            subset = dataframe[dataframe[feature] == value]
            weight = len(subset) / len(dataframe)
            subset_entropy = calculate_entropy(subset[target_col])
            info_gain -= weight * subset_entropy
        
        # Update best feature if current info_gain is higher
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            root_node = feature
    
    return root_node, best_info_gain

# Identify root node
categorical_features = ['student_wc_binned', 'teacher_wc_binned']
target = 'fluency'

print("--- A3: Root Node Detection ---")
root_node_feature, gain = find_root_node(df, categorical_features, target)
print(f"\nThe feature for the root node is '{root_node_feature}' with an Information Gain of {gain:.4f}.")
print("\n" + "="*50 + "\n")

# ==============================================================================
# A5: EXPANDED DECISION TREE MODULE
# ==============================================================================
def build_my_tree(dataframe, features, target_col, parent_node_class=None):
    """
    A recursive function to build a decision tree structure.
    """
    # Base Cases
    # 1. If all target values are the same
    if len(dataframe[target_col].unique()) <= 1:
        return {"class": dataframe[target_col].iloc[0]}
    
    # 2. If no features left
    elif len(features) == 0:
        mode_class = dataframe[target_col].mode().iloc[0]
        return {"class": mode_class}
    
    # Find best feature to split on
    best_feature, _ = find_root_node(dataframe, features, target_col)
    
    # Create tree structure
    tree = {
        "feature": best_feature,
        "children": {}
    }
    
    # Create branches
    for value in dataframe[best_feature].unique():
        subset = dataframe[dataframe[best_feature] == value]
        if len(subset) == 0:
            tree["children"][value] = {"class": parent_node_class}
        else:
            remaining_features = [f for f in features if f != best_feature]
            tree["children"][value] = build_my_tree(
                subset, 
                remaining_features,
                target_col,
                subset[target_col].mode().iloc[0]
            )
    
    return tree

print("--- A5: Custom Decision Tree Module (Structure) ---")
print("Building a simplified tree structure (output as a nested dictionary):")
custom_tree_structure = build_my_tree(df, categorical_features, target)
print("Custom tree structure generated successfully.")
print("\n" + "="*50 + "\n")

# ==============================================================================
# A6: VISUALIZE THE DECISION TREE
# ==============================================================================
print("--- A6: Visualizing the Decision Tree (using scikit-learn) ---")
print("Generating plot... Please close the plot window to continue.")

# Prepare data for scikit-learn model
X_vis = pd.get_dummies(df[categorical_features])
le = LabelEncoder()
y_vis = le.fit_transform(df[target])
class_names_vis = le.classes_

# Build and fit the Decision Tree Classifier
dt_classifier_vis = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_classifier_vis.fit(X_vis, y_vis)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_classifier_vis,
    feature_names=X_vis.columns.tolist(),
    class_names=class_names_vis,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("A6: Decision Tree Visualization for Fluency Classification")
plt.show()
print("Plot generation complete.")
print("\n" + "="*50 + "\n")

# ==============================================================================
# A7: VISUALIZE THE DECISION BOUNDARY
# ==============================================================================
print("--- A7: Visualizing the Decision Boundary (using scikit-learn) ---")
print("Generating plot... Please close the plot window to finish.")

features_boundary = ['teacher_word_count', 'student_word_count']
X_boundary = df[features_boundary].values
y_boundary = y_vis

dt_boundary = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_boundary.fit(X_boundary, y_boundary)

x_min, x_max = X_boundary[:, 0].min() - 1, X_boundary[:, 0].max() + 1
y_min, y_max = X_boundary[:, 1].min() - 1, X_boundary[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))

Z = dt_boundary.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
sns.scatterplot(x=X_boundary[:, 0], y=X_boundary[:, 1], hue=df['fluency'],
                palette='bright', s=60, edgecolor='k')

plt.title('A7: Decision Boundary for Fluency Classification')
plt.xlabel('Teacher Word Count')
plt.ylabel('Student Word Count')
plt.legend(title='Fluency')
plt.show()

print("Plot generation complete. Lab 6 script finished.")

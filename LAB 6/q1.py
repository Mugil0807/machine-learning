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
# [cite_start]Load the dataset from the provided CSV file [cite: 37]
try:
    df = pd.read_csv('original.csv')
except FileNotFoundError:
    print("Error: 'original.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Preprocessing Steps
# Fill NaN values in 'teacher' and 'student' columns with empty strings
df['teacher'] = df['teacher'].fillna('')
df['student'] = df['student'].fillna('')

# Drop rows where the target variable 'fluency' is 'NA'
df.dropna(subset=['fluency'], inplace=True)
df = df[df['fluency'] != 'NA']

# --- Feature Engineering ---
# [cite_start]Create continuous numeric features by counting the words in the dialogues [cite: 21]
df['teacher_word_count'] = df['teacher'].apply(lambda x: len(str(x).split()))
df['student_word_count'] = df['student'].apply(lambda x: len(str(x).split()))

# ========================================================================
# [cite_start]A4: BINNING FUNCTION FOR CONTINUOUS DATA [cite: 21, 23]
# ========================================================================
def create_bins(data_series, num_bins=4, bin_type='equal_width'):
    """
    Converts a continuous-valued Series into a categorical one using binning.
    [cite_start]Supports function overloading with default parameters. [cite: 24]
    """
    if bin_type == 'equal_width':
        # [cite_start]Equal width binning [cite: 21]
        max_val, min_val = data_series.max(), data_series.min()
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins + 1)]
        bins[-1] = max_val # Ensure the last bin includes the max value
        labels = [f'Bin_{i+1}' for i in range(num_bins)]
        return pd.cut(data_series, bins=bins, labels=labels, include_lowest=True)
    elif bin_type == 'equal_frequency':
        # [cite_start]Equal frequency binning [cite: 21]
        # Uses pandas qcut which ensures each bin has approx the same number of observations
        labels = [f'Bin_{i+1}' for i in range(num_bins)]
        try:
            return pd.qcut(data_series, q=num_bins, labels=labels, duplicates='drop')
        except ValueError as e:
            print(f"Warning for {data_series.name}: Could not create {num_bins} unique bins for equal frequency. {e}. Falling back to equal width.")
            return create_bins(data_series, num_bins=num_bins, bin_type='equal_width')
    else:
        raise ValueError("Invalid bin_type. Choose 'equal_width' or 'equal_frequency'.")

# Apply binning to the word count features to create categorical features
df['student_wc_binned'] = create_bins(df['student_word_count'], num_bins=3, bin_type='equal_frequency')
df['teacher_wc_binned'] = create_bins(df['teacher_word_count'], num_bins=3, bin_type='equal_frequency')

print("--- A4: Binning Example ---")
print("Binned student word counts:")
print(df[['student_word_count', 'student_wc_binned']].head())
print("\n" + "="*50 + "\n")


# ==============================================================================
# [cite_start]A1: ENTROPY CALCULATION [cite: 12]
# ==============================================================================
def calculate_entropy(data_series):
    """
    Calculates the entropy for a given pandas Series.
    [cite_start]Formula: H = -sum(p_i * log2(p_i)) [cite: 14]
    """
    counts = Counter(data_series)
    total_count = len(data_series)
    entropy = 0.0
    for count in counts.values():
        probability = count / total_count
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy

# Calculate and display the entropy of the 'fluency' target variable
entropy_fluency = calculate_entropy(df['fluency'])
print("--- A1: Entropy Calculation ---")
print(f"The entropy of the 'fluency' column is: {entropy_fluency:.4f}")
print("\n" + "="*50 + "\n")


# ==============================================================================
# [cite_start]A2: GINI INDEX CALCULATION [cite: 16]
# ==============================================================================
def calculate_gini(data_series):
    """
    Calculates the Gini index for a given pandas Series.
    [cite_start]Formula: Gini = 1 - sum(p_j^2) [cite: 17]
    """
    counts = Counter(data_series)
    total_count = len(data_series)
    gini = 1.0
    for count in counts.values():
        probability = count / total_count
        gini -= probability**2
    return gini

# Calculate and display the Gini index of the 'fluency' target variable
gini_fluency = calculate_gini(df['fluency'])
print("--- A2: Gini Index Calculation ---")
print(f"The Gini index for the 'fluency' column is: {gini_fluency:.4f}")
print("\n" + "="*50 + "\n")

# ==============================================================================
# [cite_start]A3: ROOT NODE DETECTION MODULE [cite: 18]
# ==============================================================================
def find_root_node(dataframe, features, target_col):
    """
    Identifies the best feature for the root node of a Decision Tree
    [cite_start]using Information Gain as the impurity measure. [cite: 19]
    """
    # Calculate the total entropy of the dataset (parent entropy)
    parent_entropy = calculate_entropy(dataframe[target_col])
    
    best_info_gain = -1
    root_node = None
    
    # Iterate through each feature to calculate its information gain
    for feature in features:
        unique_values = dataframe[feature].unique()
        weighted_child_entropy = 0.0
        
        for value in unique_values:
            subset = dataframe[dataframe[feature] == value]
            subset_weight = len(subset) / len(dataframe)
            weighted_child_entropy += subset_weight * calculate_entropy(subset[target_col])
            
        # Information Gain = Parent Entropy - Weighted Child Entropy
        info_gain = parent_entropy - weighted_child_entropy
        
        print(f"  - Information Gain for '{feature}': {info_gain:.4f}")

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            root_node = feature
            
    return root_node, best_info_gain

# Identify the root node using the binned categorical features
categorical_features = ['student_wc_binned', 'teacher_wc_binned']
target = 'fluency'

print("--- A3: Root Node Detection ---")
root_node_feature, gain = find_root_node(df, categorical_features, target)
print(f"\nThe feature for the root node is '{root_node_feature}' with an Information Gain of {gain:.4f}.")
print("\n" + "="*50 + "\n")

# ==============================================================================
# [cite_start]A5: EXPANDED DECISION TREE MODULE (Simplified for Demonstration) [cite: 25]
# ==============================================================================
def build_my_tree(dataframe, features, target_col, parent_node_class=None):
    """
    A simplified recursive function to build a decision tree structure.
    This demonstrates the expansion of the root node logic.
    """
    # Base Cases
    # 1. If all target values are the same, return that value
    if len(dataframe[target_col].unique()) <= 1:
        return dataframe[target_col].unique()[0]
    # 2. If no features left, return the majority class of the parent
    elif len(features) == 0:
        return parent_node_class
    # Recursive Step
    else:
        # Determine the parent's majority class for the next recursion level
        parent_node_class = dataframe[target_col].mode()[0]
        
        # Find the best feature to split on for the current dataset
        best_feature, _ = find_root_node(dataframe, features, target_col)
        
        # Create the tree structure as a nested dictionary
        tree = {best_feature: {}}
        
        # Remove the best feature from the list for subsequent splits
        remaining_features = [f for f in features if f != best_feature]
        
        # Split the data and recursively build the tree
        for value in dataframe[best_feature].unique():
            subset = dataframe[dataframe[best_feature] == value]
            subtree = build_my_tree(subset, remaining_features, target_col, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree

print("--- A5: Custom Decision Tree Module (Structure) ---")
print("Building a simplified tree structure (output as a nested dictionary):")
# Note: This is a conceptual demonstration. The output can be complex.
# We build it silently and just print a confirmation for brevity.
custom_tree_structure = build_my_tree(df, categorical_features, target)
print("Custom tree structure generated successfully.")
# To see the full structure, you could uncomment the following line:
# import json; print(json.dumps(custom_tree_structure, indent=2))
print("\n" + "="*50 + "\n")


# ==============================================================================
# [cite_start]A6: VISUALIZE THE DECISION TREE [cite: 26]
# ==============================================================================
print("--- A6: Visualizing the Decision Tree (using scikit-learn) ---")
print("Generating plot... Please close the plot window to continue.")

# Prepare data for scikit-learn model
# Features (X) must be numerical, so we'll use one-hot encoding for our binned features
X_vis = pd.get_dummies(df[categorical_features])
# Target (y) must be numerically encoded
le = LabelEncoder()
y_vis = le.fit_transform(df[target])
class_names_vis = le.classes_

# Build and fit the Decision Tree Classifier
dt_classifier_vis = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_classifier_vis.fit(X_vis, y_vis)

# [cite_start]Visualize the tree using scikit-learn's plot_tree [cite: 8]
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
# [cite_start]A7: VISUALIZE THE DECISION BOUNDARY [cite: 29]
# ==============================================================================
print("--- A7: Visualizing the Decision Boundary (using scikit-learn) ---")
print("Generating plot... Please close the plot window to finish.")

# [cite_start]Use 2 continuous features from the dataset for the classification problem [cite: 28]
features_boundary = ['teacher_word_count', 'student_word_count']
X_boundary = df[features_boundary].values
y_boundary = y_vis # Use the same encoded target as before

# Train a new Decision Tree on these two continuous features
dt_boundary = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_boundary.fit(X_boundary, y_boundary)

# Create a mesh grid to plot the decision boundary
x_min, x_max = X_boundary[:, 0].min() - 1, X_boundary[:, 0].max() + 1
y_min, y_max = X_boundary[:, 1].min() - 1, X_boundary[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))

# Predict the class for each point in the mesh grid
Z = dt_boundary.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# [cite_start]Plot the decision boundary and the data points [cite: 10, 30]
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
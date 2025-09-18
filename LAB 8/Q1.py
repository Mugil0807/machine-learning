import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# A1. Write your own functions for the following modules:
# a) Summation unit
# b) Activation Unit - Step, Bipolar Step, Sigmoid, TanH, ReLU and Leaky ReLU
# c) Comparator unit for Error calculation
# =============================================================================
print("## A1: Core Perceptron and Neural Network Functions")

# a) Summation Unit
def summation_unit(inputs, weights, bias):
    """Calculates the weighted sum of inputs."""
    return np.dot(inputs, weights) + bias

# b) Activation Units
def step_activation(y):
    """Step activation function."""
    return 1 if y >= 0 else 0

def bipolar_step_activation(y):
    """Bipolar Step activation function."""
    return 1 if y > 0 else -1

def sigmoid_activation(y):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-y))

def tanh_activation(y):
    """TanH activation function."""
    return np.tanh(y)

def relu_activation(y):
    """ReLU activation function."""
    return np.maximum(0, y)

def leaky_relu_activation(y, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(y > 0, y, y * alpha)

# c) Comparator Unit for Error Calculation
def comparator_unit(target, predicted):
    """Calculates the simple error."""
    return target - predicted

print("All functions for A1 have been defined.\n")

# =============================================================================
# A2. Develop the perceptron to learn the AND gate logic using a Step activation function.
# Initial weights: W0=10, W1=0.2, w2=-0.75, learning rate (α)=0.05
# =============================================================================
print("---")
print("\n## A2: Perceptron Learning for AND Gate (Step Activation)")

def train_perceptron_step(X, T, initial_weights, initial_bias, lr, max_epochs, convergence_error):
    """Trains a single-layer perceptron with Step activation."""
    weights = np.array(initial_weights)
    bias = initial_bias
    errors_over_epochs = []
    
    for epoch in range(max_epochs):
        sum_squared_error = 0
        for i in range(X.shape[0]):
            y = summation_unit(X[i], weights, bias)
            z = step_activation(y)
            error = comparator_unit(T[i], z)
            sum_squared_error += error**2
            
            # Update weights and bias
            weights += lr * error * X[i]
            bias += lr * error
            
        errors_over_epochs.append(sum_squared_error)
        
        if sum_squared_error <= convergence_error:
            print(f"Convergence reached at epoch {epoch + 1}.")
            return epoch + 1, errors_over_epochs, weights, bias
            
    print("Max epochs reached without convergence.")
    return max_epochs, errors_over_epochs, weights, bias

# AND Gate data
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T_and = np.array([0, 0, 0, 1])

# [cite_start]Initial parameters from PDF [cite: 37]
initial_weights_A2 = [0.2, -0.75]
initial_bias_A2 = 10
lr_A2 = 0.05
max_epochs_A2 = 1000
[cite_start]convergence_error_A2 = 0.002 # [cite: 50]

epochs_needed_A2, errors_A2, final_weights_A2, final_bias_A2 = train_perceptron_step(
    X_and, T_and, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2
)

print(f"Final Weights: {final_weights_A2}, Final Bias: {final_bias_A2}")

# Plotting epochs vs. error
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(errors_A2) + 1), errors_A2, marker='o', linestyle='-')
plt.title('Epochs vs. Sum-Squared Error for AND Gate (Step Activation)')
plt.xlabel('Epochs')
plt.ylabel('Sum-Squared Error')
plt.grid(True)
plt.show()

# =============================================================================
# A3. Repeat the A2 experiment with Bi-Polar Step, Sigmoid, and ReLU functions.
# =============================================================================
print("---")
print("\n## A3: Perceptron Learning with Other Activation Functions")

def train_perceptron_general(X, T, activation_func, derivative_func, initial_weights, initial_bias, lr, max_epochs, convergence_error):
    """A general perceptron trainer for differentiable activation functions."""
    weights = np.array(initial_weights)
    bias = initial_bias
    
    for epoch in range(max_epochs):
        sum_squared_error = 0
        for i in range(X.shape[0]):
            y_in = summation_unit(X[i], weights, bias)
            z = activation_func(y_in)
            error = comparator_unit(T[i], z)
            sum_squared_error += error**2
            
            # Weight update using delta rule
            delta = error * derivative_func(y_in)
            weights += lr * delta * X[i]
            bias += lr * delta
            
        if sum_squared_error <= convergence_error:
            return epoch + 1
            
    return max_epochs

# --- Bi-Polar Step Function ---
# Note: For Bi-Polar Step, the target values must be {-1, 1}
T_and_bipolar = np.array([-1, -1, -1, 1])
# The update rule is the same as the standard step function
epochs_bipolar, _, _, _ = train_perceptron_step(X_and, T_and_bipolar, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2)
print(f"Bi-Polar Step Function converged in {epochs_bipolar} epochs.")


# --- Sigmoid Function ---
def sigmoid_derivative(y):
    return sigmoid_activation(y) * (1 - sigmoid_activation(y))

epochs_sigmoid = train_perceptron_general(X_and, T_and, sigmoid_activation, sigmoid_derivative, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2)
print(f"Sigmoid Function converged in {epochs_sigmoid} epochs.")

# --- ReLU Function ---
def relu_derivative(y):
    return 1 if y > 0 else 0

epochs_relu = train_perceptron_general(X_and, T_and, relu_activation, relu_derivative, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2)
print(f"ReLU Function converged in {epochs_relu} epochs.")

print("\nComparison of epochs to converge:")
print(f"- Step Function: {epochs_needed_A2}")
print(f"- Bi-Polar Step Function: {epochs_bipolar}")
print(f"- Sigmoid Function: {epochs_sigmoid}")
print(f"- ReLU Function: {epochs_relu}")


# =============================================================================
# A4. Repeat exercise A2 with varying learning rates.
# =============================================================================
print("---")
print("\n## A4: Varying Learning Rates for AND Gate Perceptron")

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epochs_for_lr = []

for lr in learning_rates:
    epochs_needed, _, _, _ = train_perceptron_step(
        X_and, T_and, initial_weights_A2, initial_bias_A2, lr, max_epochs_A2, convergence_error_A2
    )
    epochs_for_lr.append(epochs_needed)

print("Epochs needed for convergence at different learning rates:")
for lr, epochs in zip(learning_rates, epochs_for_lr):
    print(f"- Learning Rate: {lr}, Epochs: {epochs}")

# Plotting learning rates vs. epochs
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, epochs_for_lr, marker='o', linestyle='-')
plt.title('Learning Rate vs. Epochs to Converge for AND Gate')
plt.xlabel('Learning Rate')
plt.ylabel('Epochs')
plt.grid(True)
plt.show()

# =============================================================================
# A5. Repeat exercises A1 to A3 for XOR gate logic.
# =============================================================================
print("---")
print("\n## A5: Perceptron Learning for XOR Gate")
print("A single-layer perceptron cannot learn the XOR function as it is not linearly separable.")
print("The training is expected to fail to converge for all activation functions.\n")

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
T_xor = np.array([0, 1, 1, 0])

# --- XOR with Step Activation ---
print("Training XOR with Step Activation:")
epochs_xor_step, _, _, _ = train_perceptron_step(
    X_xor, T_xor, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2
)

# --- XOR with Bi-Polar Step Activation ---
print("\nTraining XOR with Bi-Polar Step Activation:")
T_xor_bipolar = np.array([-1, 1, 1, -1])
epochs_xor_bipolar, _, _, _ = train_perceptron_step(
    X_xor, T_xor_bipolar, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2
)

# --- XOR with Sigmoid Activation ---
print("\nTraining XOR with Sigmoid Activation:")
epochs_xor_sigmoid = train_perceptron_general(
    X_xor, T_xor, sigmoid_activation, sigmoid_derivative, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2
)
print(f"Sigmoid reached max epochs ({epochs_xor_sigmoid}).")

# --- XOR with ReLU Activation ---
print("\nTraining XOR with ReLU Activation:")
epochs_xor_relu = train_perceptron_general(
    X_xor, T_xor, relu_activation, relu_derivative, initial_weights_A2, initial_bias_A2, lr_A2, max_epochs_A2, convergence_error_A2
)
print(f"ReLU reached max epochs ({epochs_xor_relu}).")


# =============================================================================
# A6. Build a perceptron for customer data classification using Sigmoid.
# =============================================================================
print("---")
print("\n## A6: Customer Data Classification with Perceptron (Sigmoid)")

# [cite_start]Customer data from PDF [cite: 85]
customer_data = {
    'Candies': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk Packets': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198],
    'High Value Tx?': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No']
}
df_customer = pd.DataFrame(customer_data)

# Preprocessing
X_customer = df_customer[['Candies', 'Mangoes', 'Milk Packets', 'Payment']].values
T_customer = df_customer['High Value Tx?'].apply(lambda x: 1 if x == 'Yes' else 0).values

# Normalize features for better training performance
X_customer_norm = (X_customer - X_customer.mean(axis=0)) / X_customer.std(axis=0)

# Initialize weights and bias randomly
np.random.seed(42)
initial_weights_A6 = np.random.rand(X_customer_norm.shape[1])
initial_bias_A6 = np.random.rand()
[cite_start]lr_A6 = 0.1 # User choice as per prompt [cite: 84]

epochs_customer = train_perceptron_general(
    X_customer_norm, T_customer, sigmoid_activation, sigmoid_derivative, 
    initial_weights_A6, initial_bias_A6, lr_A6, max_epochs_A2, convergence_error_A2
)

print(f"Customer data perceptron converged in {epochs_customer} epochs.")


# =============================================================================
# A7. Compare perceptron learning with matrix pseudo-inverse for AND gate.
# =============================================================================
print("---")
print("\n## A7: Comparison with Matrix Pseudo-Inverse")

# Add a bias input (a column of 1s) to the AND gate data
X_and_bias = np.c_[X_and, np.ones(X_and.shape[0])]

# Calculate weights using Pseudo-Inverse: W = (X^T * X)^-1 * X^T * T
try:
    pseudo_inverse_weights = np.linalg.inv(X_and_bias.T @ X_and_bias) @ X_and_bias.T @ T_and
    print(f"Weights calculated by Pseudo-Inverse: {pseudo_inverse_weights}")

    # Test the results
    print("\nTesting Pseudo-Inverse Weights:")
    for i in range(X_and_bias.shape[0]):
        prediction = step_activation(np.dot(X_and_bias[i], pseudo_inverse_weights))
        print(f"Input: {X_and[i]}, Target: {T_and[i]}, Prediction: {prediction}")

except np.linalg.LinAlgError:
    print("Could not compute pseudo-inverse (singular matrix).")
    
print("\nComparison:")
print("The iterative Perceptron algorithm successfully found weights to solve the AND gate.")
print("The Pseudo-Inverse method provides a direct analytical solution for linearly separable problems.")


# =============================================================================
# A8. Develop a Neural Network using backpropagation for the AND gate.
# =============================================================================
print("---")
print("\n## A8: Backpropagation for AND Gate")

def train_mlp_backprop(X, T, n_hidden, lr, max_epochs, convergence_error):
    """Trains a simple 1-hidden-layer MLP with backpropagation."""
    n_inputs = X.shape[1]
    n_outputs = T.shape[1] if T.ndim > 1 else 1

    # Initialize weights randomly
    np.random.seed(42)
    V = np.random.uniform(-0.05, 0.05, (n_inputs, n_hidden)) # Input to Hidden
    W = np.random.uniform(-0.05, 0.05, (n_hidden, n_outputs)) # Hidden to Output
    bias_V = np.random.uniform(-0.05, 0.05, n_hidden)
    bias_W = np.random.uniform(-0.05, 0.05, n_outputs)

    for epoch in range(max_epochs):
        sum_squared_error = 0
        for i in range(X.shape[0]):
            # --- Forward Pass ---
            hidden_in = np.dot(X[i], V) + bias_V
            hidden_out = sigmoid_activation(hidden_in)
            
            output_in = np.dot(hidden_out, W) + bias_W
            final_output = sigmoid_activation(output_in)

            error = T[i] - final_output
            sum_squared_error += np.sum(error**2)

            # --- Backward Pass ---
            # Delta for output layer
            delta_output = error * final_output * (1 - final_output)
            
            # Delta for hidden layer
            error_hidden = delta_output.dot(W.T)
            delta_hidden = error_hidden * hidden_out * (1 - hidden_out)
            
            # --- Weight Updates ---
            W += hidden_out.reshape(-1, 1) * delta_output * lr
            bias_W += delta_output.flatten() * lr
            V += X[i].reshape(-1, 1) * delta_hidden * lr
            bias_V += delta_hidden * lr
            
        if sum_squared_error <= convergence_error:
            print(f"MLP for AND gate converged at epoch {epoch + 1}.")
            return
            
    print("MLP for AND gate did not converge within max epochs.")

# AND gate target needs to be in a 2D array for this function
T_and_mlp = T_and.reshape(-1, 1)
train_mlp_backprop(X_and, T_and_mlp, n_hidden=2, lr=0.05, max_epochs=1000, convergence_error=0.002)


# =============================================================================
# A9. Repeat the backpropagation experiment (A8) for XOR Gate logic.
# =============================================================================
print("---")
print("\n## A9: Backpropagation for XOR Gate")
print("An MLP with backpropagation can solve the non-linearly separable XOR problem.")

T_xor_mlp = T_xor.reshape(-1, 1)
train_mlp_backprop(X_xor, T_xor_mlp, n_hidden=2, lr=0.05, max_epochs=5000, convergence_error=0.002)
# Note: Increased max_epochs as XOR is a harder problem to learn.

# =============================================================================
# A10. Repeat exercises with 2 output nodes.
# Mapping: 0 -> [1, 0], 1 -> [0, 1]
# =============================================================================
print("---")
print("\n## A10: MLP with Two Output Nodes")

# [cite_start]New target mappings [cite: 128]
T_and_2out = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])
T_xor_2out = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

print("\n--- Training AND Gate with 2 Outputs ---")
train_mlp_backprop(X_and, T_and_2out, n_hidden=2, lr=0.05, max_epochs=5000, convergence_error=0.002)

print("\n--- Training XOR Gate with 2 Outputs ---")
train_mlp_backprop(X_xor, T_xor_2out, n_hidden=2, lr=0.05, max_epochs=5000, convergence_error=0.002)

# =============================================================================
# A11. Use MLPClassifier() from Scikit-learn for AND and XOR gates.
# =============================================================================
print("---")
print("\n## A11: Using Scikit-learn's MLPClassifier")

# --- AND Gate with MLPClassifier ---
mlp_and = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=2000, random_state=42)
mlp_and.fit(X_and, T_and)
predictions_and = mlp_and.predict(X_and)
print("AND Gate Predictions using sklearn's MLPClassifier:")
print(f"Inputs:\n{X_and}")
print(f"Predictions: {predictions_and}")
print(f"Accuracy: {accuracy_score(T_and, predictions_and):.2f}\n")


# --- XOR Gate with MLPClassifier ---
mlp_xor = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.05, max_iter=5000, random_state=42)
mlp_xor.fit(X_xor, T_xor)
predictions_xor = mlp_xor.predict(X_xor)
print("XOR Gate Predictions using sklearn's MLPClassifier:")
print(f"Inputs:\n{X_xor}")
print(f"Predictions: {predictions_xor}")
print(f"Accuracy: {accuracy_score(T_xor, predictions_xor):.2f}")


# =============================================================================
# A12. Use MLPClassifier() on your project dataset (original.csv).
# =============================================================================
print("---")
print("\n## A12: MLPClassifier on Project Dataset (original.csv)")

# Load the dataset
try:
    [cite_start]df_project = pd.read_csv('original.csv') [cite: 158]

    # Preprocessing
    # Combine text columns for features, handle missing values
    df_project['text'] = df_project['teacher'].fillna('') + ' ' + df_project['student'].fillna('')
    # The 'fluency' column is our target. Drop rows where it's missing (e.g., NA).
    df_project.dropna(subset=['fluency'], inplace=True)
    
    X_text = df_project['text']
    T_fluency = df_project['fluency']

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = vectorizer.fit_transform(X_text)

    # Encode target labels
    label_encoder = LabelEncoder()
    T_encoded = label_encoder.fit_transform(T_fluency)
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, T_encoded, test_size=0.3, random_state=42)

    # Train MLPClassifier
    mlp_project = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                                solver='adam', random_state=42, tol=0.0001)
    mlp_project.fit(X_train, y_train)

    # Evaluate the model
    y_pred = mlp_project.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"The target variable is 'fluency' with classes: {label_encoder.classes_}")
    print(f"Text data was vectorized using TF-IDF.")
    print(f"MLPClassifier Accuracy on the test set: {accuracy:.2f}")

except FileNotFoundError:
    print("Could not find 'original.csv'. Please ensure the file is in the same directory.")
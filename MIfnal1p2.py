import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import MIfinal1p1 as mf


num_hidden = 10

# Step 1: Train the Vascular Neural Network (VNN)
# Assume simulate_neurovascular_network returns the output of the VNN
X = np.random.rand(100, 1)  # Replace with your actual data
y = np.random.randint(2, size=100)  # Replace with your actual labels

vascular_output = mf.simulate_neurovascular_network(X, y, num_hidden)

# Step 2: Split the data for the classifier network
iris = load_iris()
X_classifier, y_classifier = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X_classifier, y_classifier, test_size=0.2, random_state=42)
# Step 3: Use VNN output as input to the classifier neural network# Assuming vascular_output is a 1D array, reshape it to a 2D array
# Check if vascular_output is not None before reshaping
if vascular_output is not None:
    vascular_output_2d = vascular_output.reshape(-1, 1)
else:
    # Handle the case where vascular_output is None
    print("Error: vascular_output is None.")


# Transpose vascular_output_2d to have shape (1, 784)
vascular_output_2d_transposed = vascular_output_2d.T

# Assuming X_combined_train has shape (120, n) and vascular_output_2d_transposed has shape (1, 784)

# Repeat vascular_output_2d_transposed to have the same size along dimension 0 as X_train
num_rows_to_repeat = X_train.shape[0]  # Assuming X_train is the larger array along axis 0
vascular_output_2d_repeated = np.repeat(vascular_output_2d_transposed, num_rows_to_repeat, axis=0)

# Now, both arrays have the same number of rows (120), and you can concatenate them horizontally
X_combined_train = np.hstack((X_train, vascular_output_2d_repeated))

# # Concatenate the arrays horizontally
# X_combined_train = np.hstack((X_train, vascular_output_2d))

# Assuming X_combined_test has shape (30, m) and vascular_output_2d_transposed has shape (1, 784)

# Repeat vascular_output_2d_transposed to have the same size along dimension 0 as X_test
num_rows_to_repeat_test = X_test.shape[0]  # Assuming X_test is the larger array along axis 0
vascular_output_2d_repeated_test = np.repeat(vascular_output_2d_transposed, num_rows_to_repeat_test, axis=0)

# Now, both arrays have the same number of rows (30), and you can concatenate them horizontally
X_combined_test = np.hstack((X_test, vascular_output_2d_repeated_test))

# Step 4: Train the classifier neural network
classifier = MLPClassifier(hidden_layer_sizes=(num_hidden), max_iter=50, random_state=42)
classifier.fit(X_combined_train, y_train)

# Predictions
y_pred = classifier.predict(X_combined_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Classifier Accuracyv: {accuracy}')
print(y_test)
print(y_pred)

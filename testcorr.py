import numpy as np

# Example weight matrix from intermediate to final layer (100 intermediate neurons, 10 final neurons)
intermediate_to_final_weights = np.random.rand(100, 10)

# Compute correlation matrix between the intermediate neurons based on weights to final neurons
correlation_matrix = np.corrcoef(intermediate_to_final_weights)

# Extract the relevant part of the correlation matrix (the part that corresponds to intermediate neurons)
correlation_matrix = correlation_matrix[:100, :100]

# Calculate the mean correlation for each intermediate neuron with all final layer neurons
mean_correlations = np.mean(correlation_matrix, axis=1)

# Normalize mean correlations to use as scaling factors
scaling_factors = (mean_correlations - mean_correlations.min()) / (mean_correlations.max() - mean_correlations.min())

# Example weight matrix from input to intermediate layer (784 input neurons, 100 intermediate neurons)
input_to_intermediate_weights = np.random.rand(784, 100)

# Adjust weights based on scaling factors, ensuring correct broadcasting
# Scaling factors should be of shape (100,), we need to reshape it to (1, 100) for broadcasting
adjusted_weights = input_to_intermediate_weights * scaling_factors[np.newaxis, :]

print("Adjusted Weights:")
print(adjusted_weights.shape)

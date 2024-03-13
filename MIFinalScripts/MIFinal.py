import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(type(X_train))

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Assuming X_train has shape (num_samples, num_features)
# Specify the input shape with batch size (None for variable batch size)
input_shape = (X_train.shape[1],)

# Custom loss function to calculate gradients
def custom_loss(y_true, y_pred):
    # Use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        # Forward pass through model1
        predictions = model1(X_train, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)

    # Convert Keras tensor to NumPy array before calculating gradients
    gradients = tape.gradient(loss, model1.layers[1].output.numpy())

    # Use gradients to determine energy distribution
    energy_distribution = tf.reduce_sum(gradients, axis=0, keepdims=True)

    # Mean Squared Error loss between predicted energy distribution and true energy values
    return tf.keras.losses.mean_squared_error(energy_distribution, y_true)

# First Classifier Network
model1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),  # Input layer for iris dataset
    tf.keras.layers.Dense(8, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer
])

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the first classifier network
model1.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
model1.summary()

# Custom Neurovascular Network
model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Input layer with a single energy value
    tf.keras.layers.Dense(8, activation='linear')  # Output layer with one neuron per hidden neuron, linear activation
])

model2.compile(optimizer='adam', loss=custom_loss)

# Generate energy values for training (replace this with actual energy values for each sample)
energy_values = np.random.rand(len(X_train), 1)

# Train the second neurovascular network
model2.fit(energy_values, energy_values, epochs=10)

model2.summary()

# Now you can use both models for prediction or further analysis
predictions = model1.predict(X_test)

# If you have a classification problem with multiple classes, you might want to get the predicted class
predicted_classes = tf.argmax(predictions, axis=1)

# Display the predictions
print("Predictions:")
print(predicted_classes.numpy())

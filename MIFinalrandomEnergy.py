import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the MLP model
input_size = X_train.shape[1]
hidden_size = 64  # You can adjust the number of neurons in the hidden layer
output_size = len(set(y_train))

# Input layer
input_layer = tf.keras.layers.Input(shape=(input_size,))

# Hidden layer with trainable weights and biases
hidden_layer = tf.keras.layers.Dense(hidden_size, activation='relu', trainable=True)(input_layer)

# Softmax output layer
output_layer = tf.keras.layers.Dense(output_size, activation='softmax')(hidden_layer)

# Create the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Replace biases of the hidden layer with new values
new_biases = tf.random.normal(shape=(hidden_size,))

# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Accuracy on the test set: {accuracy}")

#simulate random energy values:
model.layers[1].set_weights([model.layers[1].get_weights()[0], new_biases])
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Accuracy on the test set with energy: {accuracy}")

# Save the model if needed
# model.save("mlp_model.h5")

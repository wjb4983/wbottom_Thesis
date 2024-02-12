import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# CustomHidden layer definition
class CustomHidden(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', **kwargs):
        super(CustomHidden, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", (input_shape[1], self.units))
        self.bias = self.add_weight("bias", (self.units,))
        super(CustomHidden, self).build(input_shape)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.kernel) + self.bias)

# Build the MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    CustomHidden(units=8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential()

model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10))

"""
activation: Set the activation function for the layer. This parameter is specified by the name of a built-in function or as a callable object. By default, no activation is applied.

kernel_initializer and bias_initializer: The initialization schemes that create the layer's weights (kernel and bias). This parameter is a name or a callable object. The kernel defaults to the "Glorot uniform" initializer, and the bias defaults to zeros.

kernel_regularizer and bias_regularizer: The regularization schemes that apply the layer's weights (kernel and bias), such as L1 or L2 regularization. By default, no regularization is applied.
"""

layers.Dense(64, activation="relu")

layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))

layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.01))

layers.Dense(64, kernel_initializer="orthogonal")

layers.Dense(64, bias_initializer=keras.initializers.Constant(2.0))

model = tf.keras.Sequential(
    [
        # Adds a densely-connected layer with 64 units to the model:
        layers.Dense(64, activation="relu", input_shape=(32,)),
        # Add another:
        layers.Dense(64, activation="relu"),
        # Add an output layer with 10 output units:
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

"""
optimizer: This object specifies the training procedure. Pass it optimizer instances from the tf.keras.optimizers module, such as tf.keras.optimizers.Adam or tf.keras.optimizers.SGD. If you just want to use the default parameters, you can also specify optimizers via strings, such as 'adam' or 'sgd'.

loss: The function to minimize during optimization. Common choices include mean square error (mse), categorical_crossentropy, and binary_crossentropy. Loss functions are specified by name or by passing a callable object from the tf.keras.losses module.

metrics: Used to monitor training. These are string names or callables from the tf.keras.metrics module.

Additionally, to make sure the model trains and evaluates eagerly, you can make sure to pass run_eagerly=True as a parameter to compile.
"""

model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse", metrics=["mae"])

model.compile(
    optimizer=keras.optimizers.RMSprop(0.01),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

"""
epochs: Training is structured into epochs. An epoch is one iteration over the entire input data (this is done in smaller batches).
batch_size: When passed NumPy data, the model slices the data into smaller batches and iterates over these batches during training. This integer specifies the size of each batch. Be aware that the last batch may be smaller if the total number of samples is not divisible by the batch size.
validation_data: When prototyping a model, you want to easily monitor its performance on some validation data. Passing this argument—a tuple of inputs and labels—allows the model to display the loss and metrics in inference mode for the passed data, at the end of each epoch.
"""

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(
    data, labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels)
)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

model.fit(dataset, epochs=10)

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32)

model.fit(dataset, epochs=10, validation_data=val_dataset)

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset)

result = model.predict(data, batch_size=32)
print(result.shape)

inputs = keras.Input(shape=(32,))

x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)
predictions = layers.Dense(10)(x)


model = keras.Model(inputs=inputs, outputs=predictions)
# The compile step specifies the training configuration.
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name="my_model")
        self.num_classes = num_classes
        # Define your layers here.
        self.dense_1 = layers.Dense(32, activation="relu")
        self.dense_2 = layers.Dense(num_classes)

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        return self.dense_2(x)


model = MyModel(num_classes=10)

model.compile(
    optimizer=keras.optimizers.RMSprop(0.001),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(data, labels, batch_size=32, epochs=5)

"""
Custom layers
Create a custom layer by subclassing tf.keras.layers.Layer and implementing the following methods:

__init__: Optionally define sublayers to be used by this layer.

build: Create the weights of the layer. Add weights with the add_weight method.

call: Define the forward pass.

Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.
Here's an example of a custom layer that implements a matmul of an input with a kernel matrix:
"""


class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_dim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config["output_dim"] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = tf.keras.Sequential([MyLayer(10)])

# The compile step specifies the training configuration
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)

callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss"),
    # Write TensorBoard logs to `./logs` directory
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
]

model.fit(
    data,
    labels,
    batch_size=32,
    epochs=5,
    callbacks=callbacks,
    validation_data=(val_data, val_labels),
)

model = tf.keras.Sequential(
    [layers.Dense(64, activation="relu", input_shape=(32,)), layers.Dense(10)]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Save weights to a TensorFlow Checkpoint file
model.save_weights("./weights/my_model")

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights("./weights/my_model")

# Save weights to a HDF5 file
model.save_weights("my_model.h5", save_format="h5")

# Restore the model's state
model.load_weights("my_model.h5")

# Serialize a model to JSON format
json_string = model.to_json()
print(json_string)

import json
import pprint

pprint.pprint(json.loads(json_string))

fresh_model = tf.keras.models.model_from_json(json_string)

yaml_string = model.to_yaml()
print(yaml_string)

fresh_model = tf.keras.models.model_from_yaml(yaml_string)

# Create a simple model
model = tf.keras.Sequential(
    [layers.Dense(10, activation="relu", input_shape=(32,)), layers.Dense(10)]
)
model.compile(
    optimizer="rmsprop",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(data, labels, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save("my_model")

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model("my_model")


# (input: 784-dimensional vectors)
#       ↧
# [Dense (64 units, relu activation)]
#       ↧
# [Dense (64 units, relu activation)]
#       ↧
# [Dense (10 units, softmax activation)]
#       ↧
# (output: logits of a probability distribution over 10 classes)

inputs = keras.Input(shape=(784,))
print(inputs.shape)
print(inputs.dtype)

dense = layers.Dense(64, activation="relu")
x = dense(inputs)

x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

print(keras.utils.plot_model(model, "my_first_model.png"))

# MNIST EXAMPLE
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

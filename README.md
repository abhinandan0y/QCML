# QCML

Some practical code and exercise 
To implement Hybrid Quantum Neural Networks for Remote Sensing Imagery Classification. 
Also links to do analysis on remote sensing  image classification

#### Familiarity with libraries such as Qiskit, Pennylane, TensorFlow, and Keras.

```python
pip install qiskit pennylane tensorflow numpy matplotlib
pip install tensorflow_datasets
```
#### Step 2: Prepare the Data
Use a remote sensing dataset. For this example, we will use the EuroSAT dataset available via TensorFlow Datasets.

```python

import tensorflow as tf
import tensorflow_datasets as tfds
import pennylane as qml
import numpy as np
from tensorflow.keras import layers, models

# Load EuroSAT dataset
dataset, info = tfds.load('eurosat/rgb', with_info=True)
#train_data = dataset['train']

train_data, val_data = tfds.load('eurosat/rgb', split=['train[:80%]', 'train[80%:]'], as_supervised=True)

# Preprocess the data
#def preprocess(features):
#    image = tf.image.resize(features['image'], (64, 64)) / 255.0
#    label = features['label']
#    return image, label

def preprocess(image, label):
    image = tf.image.resize(image, (64, 64)) / 255.0
    return image, label

train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
```

#### Step 3: Define the Quantum Layer
#Here, we will define a simple quantum circuit using Pennylane.

```python

import pennylane as qml

n_qubits = 4
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[2, 3])
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def quantum_layer(inputs):
    inputs = tf.cast(inputs, dtype=tf.float32)
    outputs = np.array([quantum_circuit(input) for input in inputs])
    outputs = outputs.astype(np.float32)  # Ensure the numpy array is float32
    return tf.convert_to_tensor(outputs, dtype=tf.float32)
```

#### Step 4: Integrate Quantum Layer with Classical Model
#We integrate the quantum layer into a Keras model.

```python

from tensorflow.keras import layers, models
import numpy as np

class HybridModel(tf.keras.Model):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(n_qubits, activation='relu')
        self.dense2 = layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = tf.numpy_function(quantum_layer, [x], tf.float32)
        x.set_shape((None, n_qubits))  # Ensure shape is set for the output of quantum layer
        x = self.dense2(x)
        return x

model = HybridModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Custom training loop
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

epochs = 10

for epoch in range(epochs):
    print(f'Start of epoch {epoch+1}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y_batch_train, logits)
        if step % 100 == 0:
            print(f'Epoch {epoch+1} Step {step} Loss {loss_value.numpy()} Accuracy {train_acc_metric.result().numpy()}')
    train_acc = train_acc_metric.result()
    print(f'Training accuracy over epoch {epoch+1}: {train_acc.numpy()}')
    train_acc_metric.reset_states()
```
#### Save the model
```python
model.save('hybrid_model.h5') #NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional model or a Sequential model. It does not work for subclassed models, because such models are defined via the body of a Python method, which isn't safely serializable. Consider saving to the Tensorflow SavedModel format (by setting save_format="tf") or using `save_weights`.

# Save the model in TensorFlow SavedModel format
model.save('saved_model', save_format='tf')
>>> model.save('QCML_v1_model', save_format='tf')
INFO:tensorflow:Assets written to: QCML_v1_model/assets
INFO:tensorflow:Assets written to: QCML_v1_model/assets

or simply
model.save('saved_model')

# Load the model
loaded_model = tf.keras.models.load_model('saved_model', custom_objects={'quantum_layer': quantum_layer})
```

#### Load the saved model
```python
loaded_model = tf.keras.models.load_model('hybrid_model.h5', custom_objects={'quantum_layer': quantum_layer})

# Evaluate the loaded model to verify it works correctly
val_loss, val_accuracy = loaded_model.evaluate(val_data)

print("Loaded Model Validation Loss:", val_loss)
print("Loaded Model Validation Accuracy:", val_accuracy)
```

#### Step 5: Evaluate the Model
#Evaluate the model's performance on a validation dataset.

```python

# Preprocess the validation data
#val_data = val_data.map(preprocess).batch(32)
val_data = val_data.map(lambda image, label: preprocess(image, label)).batch(32)

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_data)

# Print the validation loss and accuracy
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
```
#### Results:

```bash
169/169 [==============================] - 65s 381ms/step - loss: 2.2502 - accuracy: 0.1667
>>> # Print the validation loss and accuracy
>>> print("Validation Loss:", val_loss)
Validation Loss: 2.250182867050171
>>> print("Validation Accuracy:", val_accuracy)
Validation Accuracy: 0.1666666716337204
```

#### Monitor Training Progress:
You can use TensorBoard to monitor training progress. First, install TensorBoard using pip install tensorboard. Then, add the following code to your existing script to enable TensorBoard:
```python
# Add TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

# Train the model with TensorBoard callback
model.fit(train_data, epochs=10, callbacks=[tensorboard_callback])
You can then visualize the training metrics using the TensorBoard interface by running tensorboard --logdir=./logs in the terminal and navigating to http://localhost:6006 in your web browser.
```

#### Hybrid quantum-classical model for image classification using EuroSAT dataset. Here are some suggestions for potential Improvements:
```
#Data Augmentation:
    Incorporate data augmentation techniques like rotation, flipping, and scaling to increase the diversity of the training dataset. This can help improve model generalization.

#Complexity of Quantum Circuit: 
    Experiment with different quantum circuit architectures, including varying the number of qubits, layers, and types of gates. More complex quantum circuits may capture richer features from the data.

#Hyperparameter Tuning: 
    Tune hyperparameters such as learning rate, batch size, and the number of epochs to find the optimal configuration for your model.

#Regularization: 
    Apply regularization techniques like dropout or L2 regularization to prevent overfitting and improve model robustness.

#Ensemble Learning: 
    Train multiple hybrid models with different initializations or architectures and combine their predictions to boost performance.

#Quantum Embedding: 
    Explore methods to embed classical data into quantum states more effectively, such as amplitude encoding or quantum feature maps.

#Transfer Learning: 
    Utilize pre-trained classical convolutional neural networks (CNNs) as feature extractors before passing the features to the quantum layer.

#Monitor Training Progress: 
    Visualize training metrics like loss and accuracy over epochs using tools like TensorBoard to identify potential issues or areas for improvement.
```

### Exercise
```
Experiment with different quantum circuits: Modify the quantum_circuit function to use different quantum gates and configurations.
Data Augmentation: Apply data augmentation techniques to improve model robustness.
Hyperparameter Tuning: Tune hyperparameters such as learning rate, batch size, and the number of epochs.
Model Architecture: Experiment with different classical neural network architectures to see their impact on performance.
Deploy on Quantum Hardware: If you have access to quantum hardware, try running the quantum circuit on actual quantum devices provided by IBM Q or others.
```
### Additional Resources and Links
```
To deepen your understanding and find more resources, consider the following links:

Qiskit Documentation: Qiskit Documentation
Pennylane Tutorials: Pennylane Tutorials
TensorFlow Quantum: TensorFlow Quantum
Remote Sensing Image Classification with Deep Learning: Medium Article
Kaggle Datasets: EuroSAT dataset on Kaggle
These steps and resources should help you get started with implementing HQNNs for remote sensing imagery classification.
```



















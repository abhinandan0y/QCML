Python 3.10.9 (main, Mar  1 2023, 18:23:06) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2024-06-07 14:17:16.238228: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-06-07 14:17:17.866354: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-06-07 14:17:22.488943: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-06-07 14:17:33.017170: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
>>> import tensorflow_datasets as tfds
>>> import pennylane as qml
>>> import numpy as np
>>> from tensorflow.keras import layers, models
>>> 
>>> #### Step 2: Prepare the Data
>>> # Load EuroSAT dataset
>>> dataset, info = tfds.load('eurosat/rgb', with_info=True)
>>> #train_data = dataset['train']
>>> 
>>> train_data, val_data = tfds.load('eurosat/rgb', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
>>> 
>>> 
>>> # Preprocess the data
>>> #def preprocess(features):
>>> #    image = tf.image.resize(features['image'], (64, 64)) / 255.0
>>> #    label = features['label']
>>> #    return image, label
>>> 
>>> def preprocess(image, label):
...     image = tf.image.resize(image, (64, 64)) / 255.0
...     return image, label
... 
>>> train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
WARNING:tensorflow:AutoGraph could not transform <function preprocess at 0x7f0df995bbe0> and will run it as-is.
Cause: Unable to locate the source code of <function preprocess at 0x7f0df995bbe0>. Note that functions defined in certain environments, like the interactive Python shell, do not expose their source code. If that is the case, you should define them in a .py source file. If you are certain the code is graph-compatible, wrap the call using @tf.autograph.experimental.do_not_convert. Original error: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <function preprocess at 0x7f0df995bbe0> and will run it as-is.
Cause: Unable to locate the source code of <function preprocess at 0x7f0df995bbe0>. Note that functions defined in certain environments, like the interactive Python shell, do not expose their source code. If that is the case, you should define them in a .py source file. If you are certain the code is graph-compatible, wrap the call using @tf.autograph.experimental.do_not_convert. Original error: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING: AutoGraph could not transform <function preprocess at 0x7f0df995bbe0> and will run it as-is.
Cause: Unable to locate the source code of <function preprocess at 0x7f0df995bbe0>. Note that functions defined in certain environments, like the interactive Python shell, do not expose their source code. If that is the case, you should define them in a .py source file. If you are certain the code is graph-compatible, wrap the call using @tf.autograph.experimental.do_not_convert. Original error: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
>>> 
>>> #### Step 3: Define the Quantum Layer
>>> n_qubits = 4
>>> dev = qml.device('default.qubit', wires=n_qubits)
>>> 
>>> @qml.qnode(dev)
... def quantum_circuit(inputs):
...     for i in range(n_qubits):
...         qml.RY(inputs[i], wires=i)
...     qml.CZ(wires=[0, 1])
...     qml.CZ(wires=[2, 3])
...     for i in range(n_qubits):
...         qml.RY(inputs[i], wires=i)
...     return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
... 
>>> def quantum_layer(inputs):
...     inputs = tf.cast(inputs, dtype=tf.float32)
...     outputs = np.array([quantum_circuit(input) for input in inputs])
...     outputs = outputs.astype(np.float32)  # Ensure the numpy array is float32
...     return tf.convert_to_tensor(outputs, dtype=tf.float32)
... 
>>> #### Step 4: Define the Model
>>> class HybridModel(tf.keras.Model):
...     def __init__(self):
...         super(HybridModel, self).__init__()
...         self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
...         self.pool1 = layers.MaxPooling2D((2, 2))
...         self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
...         self.pool2 = layers.MaxPooling2D((2, 2))
...         self.flatten = layers.Flatten()
...         self.dense1 = layers.Dense(n_qubits, activation='relu')
...         self.dense2 = layers.Dense(10, activation='softmax')
...         
...     def call(self, x):
...         x = self.conv1(x)
...         x = self.pool1(x)
...         x = self.conv2(x)
...         x = self.pool2(x)
...         x = self.flatten(x)
...         x = self.dense1(x)
...         x = tf.numpy_function(quantum_layer, [x], tf.float32)
...         x.set_shape((None, n_qubits))  # Ensure shape is set for the output of quantum layer
...         x = self.dense2(x)
...         return x
... 
>>> model = HybridModel()
>>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
>>> 
>>> # Custom training loop
>>> loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
>>> optimizer = tf.keras.optimizers.Adam()
>>> train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
>>> 
>>> epochs = 2
>>> 
>>> for epoch in range(epochs):
...     print(f'Start of epoch {epoch+1}')
...     for step, (x_batch_train, y_batch_train) in enumerate(train_data):
...         with tf.GradientTape() as tape:
...             logits = model(x_batch_train, training=True)
...             loss_value = loss_fn(y_batch_train, logits)
...         grads = tape.gradient(loss_value, model.trainable_weights)
...         optimizer.apply_gradients(zip(grads, model.trainable_weights))
...         train_acc_metric.update_state(y_batch_train, logits)
...         if step % 100 == 0:
...             print(f'Epoch {epoch+1} Step {step} Loss {loss_value.numpy()} Accuracy {train_acc_metric.result().numpy()}')
...     train_acc = train_acc_metric.result()
...     print(f'Training accuracy over epoch {epoch+1}: {train_acc.numpy()}')
...     train_acc_metric.reset_states()
...     
... #### Step 5: Evaluate the Model
... 
Start of epoch 1
2024-06-07 14:18:21.821805: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 15745024 exceeds 10% of free system memory.
2024-06-07 14:18:22.159840: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 15745024 exceeds 10% of free system memory.
2024-06-07 14:18:22.211710: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 15745024 exceeds 10% of free system memory.
/home/abhinandan/anaconda3/lib/python3.10/site-packages/keras/src/optimizers/base_optimizer.py:664: UserWarning: Gradients do not exist for variables ['kernel', 'bias', 'kernel', 'bias', 'kernel', 'bias'] when minimizing the loss. If using `model.compile()`, did you forget to provide a `loss` argument?
  warnings.warn(
<KerasVariable shape=(), dtype=int64, path=adam/iteration>
Epoch 1 Step 0 Loss 2.7758798599243164 Accuracy 0.15625
2024-06-07 14:18:24.757313: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 15745024 exceeds 10% of free system memory.
2024-06-07 14:18:24.772974: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 15745024 exceeds 10% of free system memory.
<KerasVariable shape=(), dtype=int64, path=adam/iteration>
<KerasVariable shape=(), dtype=int64, path=adam/iteration>
<KerasVariable shape=(), dtype=int64, path=adam/iteration>
...

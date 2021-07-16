import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras
import tensorflow as tf
import numpy as np

arr = np.array([[1, 2, 3]])

optimizer = keras.optimizers.SGD()

model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(3,))
])

model.compile()

grad = []
loss_log = []

for i in range(5):

    with tf.GradientTape() as tape:
        pred = model(arr)
        loss = (3 - pred)**2
    grad.append(tape.gradient(loss, model.trainable_variables))
    loss_log.append(loss)

    optimizer.apply_gradients(zip(grad[i], model.trainable_variables))

print(grad, "\n\nloss\n",loss_log)
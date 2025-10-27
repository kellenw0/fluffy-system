import pandas as pd
import numpy as np
import tensorflow as tf

labels = pd.read_csv('csv/labels.csv')
inputs = pd.read_csv('csv/inputs.csv')
xs = []
ys = []

for i, (x, y) in enumerate(zip(inputs.to_numpy(), labels.to_numpy())):
    if np.isnan(x).any():
        # print(i, x, y)
        pass
    else:
        xs.append(x)
        ys.append(y)

xs = np.array(xs)
ys = np.array(ys)
xs -= xs.mean(axis=0)
xs /= xs.std(axis=0)

print(len(xs))
print(xs)
print(ys)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 4. Compile the model
model.compile(optimizer='rmsprop',
              loss="mse",
              metrics=['mae'])

# 5. Train the model
model.fit(xs, ys, epochs=50)

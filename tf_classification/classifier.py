import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

print('Loading Data!')
(a, b), (c, d) = mnist.load_data() #cifar10.load_data()

train_cut = np.where(b < 8)[0]
val_cut = np.where(d < 8)[0]

X_train = a[train_cut][:,:,:,np.newaxis]
y_train = tf.keras.utils.to_categorical(b[train_cut])

X_val = c[val_cut][:,:,:,np.newaxis]
y_val = d[val_cut]

print('Building Model!')
model = tf.keras.Sequential([
    Conv2D(64, (5,5), input_shape = X_train.shape[1:], activation='relu'),
    Conv2D(64, (5,5), activation='relu'),
    MaxPool2D(),
    
    Conv2D(128, (3,3), activation='relu'),
    Conv2D(128, (3,3), activation='relu'),
    MaxPool2D(),
    
    Flatten(),
    
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')    
])

adam = Adam(learning_rate = 0.001)
model.compile(optimizer=adam, 
              loss=CategoricalCrossentropy(), 
              metrics=['acc'])

print(model.summary())

print('Training!')
model.fit(X_train, y_train, batch_size=32, shuffle=True, epochs=1)

import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imread, imshow, subplots, show

from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

(X_train, y_train), (X_val, y_val) = mnist.load_data()
X_train = X_train[:,:,:,np.newaxis]
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)


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

model.fit(X_train, y_train, batch_size=32, shuffle=True, epochs=1, verbose=1)

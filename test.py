import tensorflow as tf
from tensorflow.keras import models,layers

gpu_device = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_device[0],True)

height = 360
width = 480

model = models.Sequential()
model.add(layers.Conv2D(filters=64,kernel_size=9, padding = 'same', activation='relu', input_shape = (height,width,3)))
model.add(layers.LeakyReLU(alpha = 0.1))
model.add(layers.Conv2D(filters=16,kernel_size=1, padding = 'same', activation='relu'))
model.add(layers.Conv2D(filters=32,kernel_size=7, padding = 'same', activation='relu'))
model.add(layers.LeakyReLU(alpha = 0.1))
model.add(layers.Conv2D(filters=8,kernel_size=1, padding = 'same', activation='relu'))
model.add(layers.Conv2D(filters=16,kernel_size=3, padding = 'same', activation='relu'))
model.add(layers.LeakyReLU(alpha = 0.1))
model.add(layers.Conv2D(filters=8,kernel_size=1, padding = 'same', activation='relu'))
model.add(layers.Conv2D(filters=3,kernel_size=5, padding = 'same', activation='relu'))
model.compile(optimizer='adam', metrics=['accuracy'], loss = 'mean_squared_error')

model.compile(optimizer='adam', metrics=['accuracy'])
model.summary()


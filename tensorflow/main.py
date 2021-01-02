import tensorflow as tf
from scipy.io import loadmat
import numpy as np

# Load data
train = loadmat("../data/training_set.mat")
val = loadmat("../data/validation_set.mat")

# Preprocessing
# the samples must change on the first dimension, reshape array
train['trainSet'] = np.reshape(train['trainSet'],
                               (train['trainSet'].shape[3],
                                train['trainSet'].shape[0],
                                train['trainSet'].shape[1],
                                train['trainSet'].shape[2]))
val['valSet'] = np.reshape(val['valSet'],
                           (val['valSet'].shape[3],
                            val['valSet'].shape[0],
                            val['valSet'].shape[1],
                            val['valSet'].shape[2]))
# labels must be one-hot encoded
train['trainLabel'] = tf.keras.utils.to_categorical(np.transpose(train['trainLabel'] - 1))
val['valLabel'] = tf.keras.utils.to_categorical(np.transpose(val['valLabel'] - 1))

# Model definition
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(train['trainSet'].shape[1:]), name='rgbd_images'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.Conv2D(64, 3, padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.LeakyReLU(),
                             tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                             tf.keras.layers.Conv2D(128, 3, padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.LeakyReLU(),
                             tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                             tf.keras.layers.Conv2D(256, 3, padding='same'),
                             tf.keras.layers.BatchNormalization(),
                             tf.keras.layers.LeakyReLU(),
                             tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(784),
                             tf.keras.layers.Dense(3)])
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Model training
model.fit(train['trainSet'], train['trainLabel'], epochs=3, batch_size=30,
          validation_data=(val['valSet'], val['valLabel']))
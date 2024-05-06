import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten
import keras
from tensorflow.keras.layers import Layer, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



img_height, img_width = 180, 180
batch_size = 32
data_dir = r"C:\Users\DELL\PycharmProjects\ResNet50\dataset\training"
train_ds1 = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=123,
    label_mode="categorical",  # Use this to categorize the data
    image_size=(img_height, img_width),
    batch_size=batch_size
)
data_dir2 = r"C:\Users\DELL\PycharmProjects\ResNet50\dataset\evaluation"
train_ds2 = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir2,
    seed=123,
    label_mode="categorical",  # Use this to categorize the data
    image_size=(img_height, img_width),
    batch_size=batch_size
)
train_ds = train_ds1.concatenate(train_ds2)

val_dir = r"C:\Users\DELL\PycharmProjects\ResNet50\dataset\validation"
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    seed=123,
    label_mode="categorical",  # Use this to categorize the data
    image_size=(img_height, img_width),
    batch_size=batch_size)



# Ensuring the class names
class_names = train_ds1.class_names
print(class_names)

resnet_model = Sequential()
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(180, 180, 3),
                                                  pooling='avg',
                                                  weights='imagenet')
for layer in pretrained_model.layers:
    layer.trainable = False
resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dropout(0.5))
# Adding Dense layers for desired outcome
resnet_model.add(Dense(512, activation="relu"))
resnet_model.add(Dense(11, activation='softmax'))

resnet_model.summary()
resnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



# Train your model and store the training history
epochs = 3
history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
# Print training and validation accuracy and loss
print("Training History:")
print("Accuracy:", history.history['accuracy'])
print("Loss:", history.history['loss'])
print("Validation History:")
print("Validation Accuracy:", history.history['val_accuracy'])
print("Validation Loss:", history.history['val_loss'])



# Save the model in the Keras native format
# model_filename = 'my_model_05.keras'
# resnet_model.save(model_filename)
# print("Model saved in H5 format as", model_filename)
#
# """
# # Load the Keras model
# import tensorflow as tf
# loaded_model = tf.keras.models.load_model('my_model.keras')
# """


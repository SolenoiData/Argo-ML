# librería para separar la información
import splitfolders
# Librerías estándar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
import math

from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

# Separar los datos
input_folder = "/mnt/vol/flowers"
output = "/mnt/vol/processed_data"
splitfolders.ratio(input_folder, output, seed=42, ratio=(.6, .2, .2))

img_height, img_width = (224, 224)
batch_size = 32
train_data_dir = r"/mnt/vol/processed_data/train"
valid_data_dir = r"/mnt/vol/processed_data/val"
test_data_dir = r"/mnt/vol/processed_data/test"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.4)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')  # set as training data

valid_generator = train_datagen.flow_from_directory(
    valid_data_dir,  # same directory training data
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # set as validation data

test_generator = train_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    subset='validation')  # set as validation data

base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_generator, epochs=10)

model.save('/mnt/vol/output/ResNet50_Flowers.h5')

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest accuracy: ', test_acc)

# Save text in an exterior file

with open("/mnt/vol/output/example.txt", "w") as f:
    f.write(str(test_acc))

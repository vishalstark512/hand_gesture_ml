from keras.models import Sequential
from network import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import keras

ap = argparse.ArgumentParser()
ap.add_argument("-image", "--image", type=str, default='C-train026.ppm',help="Path of test image")
ap.add_argument("-num_class","--class",type=int, default=6,help="(required) number of classes to be trained")
args = vars(ap.parse_args())

base_model = VGG16.VGG16(include_top=False, weights=None)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128)(x)
predictions = Dense(args["class"], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("hands_weights.h5")

inputShape = (224,224) # Assumes 3 channel image
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)   # shape is (224,224,3)
image = np.expand_dims(image, axis=0)  # Now shape is (1,224,224,3)

image = image/255.0

preds = model.predict(image)
print(preds)

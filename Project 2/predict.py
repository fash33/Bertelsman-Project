import argparse
import warnings 
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub
import tensorflow_datasets as tfds 
import logging
import json

p = argparse.ArgumentParser()

p.add_argument('input', action="store", type = str, help = 'Image path')
p.add_argument('model',action="store", type = str, help = 'Classifier path')
p.add_argument('--top_k', default = 5, action = "store", type = int, help = 'Return the top K most likely classes')
p.add_argument('--category_names', default = './label_map.json', action = "store", type = str, help = 'JSON file mapping labels')
arg_parser = p.parse_args()
top_k = arg_parser.top_k

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image,(224,224))
    image /= 255
    return image

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    test_image = np.asarray(img)
    processed_test_image = process_image(test_image)
    prob_preds = model.predict(np.expand_dims(processed_test_image, axis = 0))
    
    values,indices = tf.nn.top_k(prob_preds, k = top_k)
    probs = list(values.numpy()[0])
    classes = list(indices.numpy()[0])
    
    return probs, classes

with open(arg_parser.category_names, "r") as file:
    mapping = json.load(file)

loaded_model = tf.keras.models.load_model(arg_parser.model, compile = False, custom_objects = {'KerasLayer':hub.KerasLayer})
print(f"\n Top {top_k} Classes \n")
probs, labels = predict(arg_parser.input, loaded_model, top_k)

for prob, label in zip(probs, labels):
    print('Label:', label)
    print('Class name:', mapping[str(label+1)].title())
    print('Probability:',prob)
                                 

                             
    

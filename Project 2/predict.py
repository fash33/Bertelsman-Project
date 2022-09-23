import argparse, time, warnings

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow_hub as hub
import tensorflow_datasets as tfds 
import logging,json

parser = argparse.ArgumentParser()

parser.add_argument('input', action="store", type = str, help = 'Image path')
parser.add_argument('model',action="store", type = str, help = 'Classifier path')
parser.add_argument('--top_k', default = 5, action = "store", type = int, help = 'Return the top K most likely classes')
parser.add_argument('--category_names', default = './label_map.json', action = "store", type = str, help = 'JSON file mapping labels')
arg_p = parser.parse_args()
top_k = arg_p.top_k

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    samp_image = np.asarray(img)
    processed_samp_image = process_image(samp_image)
    probability_predictions = model.predict(np.expand_dims(processed_samp_image, axis = 0))
    
    vals,indices = tf.nn.top_k(probability_predictions, k = top_k)
    probabilities = list(vals.numpy()[0])
    classes = list(indices.numpy()[0])
    
    return probabilities, classes

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image,(224,224))
    image /= 255
    return image

with open(arg_p.category_names, "r") as file:
    mapping = json.load(file)

loaded_model = tf.keras.models.load_model(arg_p.model, custom_objects = {'KerasLayer':hub.KerasLayer}, compile = False)
print(f"\n Top {top_k} Classes \n")
probabilities, labels = predict(arg_p.input, loaded_model, top_k)

for probability, label in zip(probabilities, labels):
    print('Image Label:', label)
    print('Classname:', mapping[str(label+1)].title())
    print('Probability:',probability)
                                 

                             
    


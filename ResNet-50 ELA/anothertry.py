import cv2
import numpy as np
import os
import urllib.request
import tensorflow as tf
from tensorflow import keras
from keras_resnet.models import ResNet50
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing import image
from PIL import Image, ImageChops, ImageEnhance

def ela(img_path):
    resaved_path = 'resaved.jpg'
    ELA_path = 'ELA.png'
    im = Image.open(img_path)
    im.save(resaved_path, 'JPEG', quality=90)
    resaved_im = Image.open(resaved_path)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    ela_im.save(ELA_path)
    ela_img = image.load_img(ELA_path, target_size=(224, 224))
    ela_array = image.img_to_array(ela_img)
    os.remove(resaved_path)
    os.remove(ELA_path)
    return ela_array

def predict_image(img_path, model):
    ela_array = ela(img_path)
    x = np.expand_dims(ela_array, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0][0]

model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))

# Example usage
img_path = './Deepika_TShirt_Fake.jpg'
prediction = predict_image(img_path, model)
if prediction < 0.5:
    print("The image is real.")
else:
    print("The image is fake.")

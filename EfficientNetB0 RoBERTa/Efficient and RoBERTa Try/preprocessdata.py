from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os
from efficientnet.tfkeras import EfficientNetB0
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

efficientnet_model = EfficientNetB0(weights='imagenet')
roberta_model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
img = Image.open(os.path.join(os.getcwd(), 'C:\\Users\\This PC\\Desktop\\Major Project\\Efficient Roberta\\chatgpt\\Aircraft_Fake.jpg'))

plt.imshow(img)

img = img.resize((224, 224))
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
features = efficientnet_model.predict(img_array)

import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

efficientnet_model = EfficientNetB0(weights='imagenet')
roberta_model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

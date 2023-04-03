
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.layers import SimpleRNN
from keras.layers.embeddings import Embedding
import keras
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

#seeding for reproducibility i.e for same results for a given input every time the program is run
seed = 7
np.random.seed(seed)


#
positive_examples=list(open('hindi_pos.txt',mode='r'))
positive_examples = [s.strip() for s in positive_examples]
negative_examples=list(open('hindi_neg.txt',mode='r'))
negative_examples = [s.strip() for s in negative_examples]
x=positive_examples+negative_examples
#print x
xa=np.asarray(x)
print(xa.shape)
positive_labels = [[1] for _ in positive_examples]
negative_labels = [[0] for _ in negative_examples]
y = np.concatenate([positive_labels, negative_labels], 0)
print(y.shape)

#using train_test_split to automatically split data into train and test in ratio 70:30 respectively.
from sklearn.model_selection import train_test_split
xa_train,xa_test,y_train,y_test=train_test_split(xa,y,test_size=0.3,random_state=4)
print(xa_train.shape)
print(xa_test.shape)
print(y_train.shape)
print(y_test.shape)

#tokenizer used to split each line into words and then labelling each word with a key(a numeric value) 
#so as to generate an array to pass in furthur functions using pad_sequences.
tokenizer = Tokenizer(num_words=None,split=' ',lower=True)
tokenizer.fit_on_texts(xa_train)
integer_sentences_train = tokenizer.texts_to_sequences(xa_train)
data_train = pad_sequences(integer_sentences_train,padding='post',truncating='post',value=0.)
print(data_train[0])
top_words = 5000 #len(tokenizer.word_index)
print(top_words)
max_words = 30

tokenizer.fit_on_texts(xa_test)
integer_sentences_test = tokenizer.texts_to_sequences(xa_test)
data_test = pad_sequences(integer_sentences_test,padding='post',truncating='post',value=0.)
print(data_test[0])

data_train = sequence.pad_sequences(data_train, maxlen=max_words, dtype='float32')
data_test = sequence.pad_sequences(data_test, maxlen=max_words, dtype='float32')

#creating the model i.e adding the layers to the model
model = Sequential()
model.add(Embedding(input_dim = top_words, output_dim = 20, input_length=max_words))
model.add(SimpleRNN(100,return_sequences=True))
model.add(LSTM(80,return_sequences=True))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#training and testing the model and generating accuracy score.
model.fit(data_train[:-1],y_train[:-1], batch_size=128,epochs=5,validation_data=(data_test, y_test), verbose=1,sample_weight=None, initial_epoch=0)
yp = model.predict(data_test[:-1], batch_size=32, verbose=1)
ypreds = np.argmax(yp, axis=1)
scores = model.evaluate(data_test[:-1], y_test[:-1], verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


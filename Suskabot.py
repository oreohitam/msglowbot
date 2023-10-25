"""KELOMPOK 1 (THE FASTAR )
PROJEK AKHIR SUSKABOY

NAMA ANGGOTA :
1. CANDRALIKA DIFA SENA
2. SYAFRIDHO
3. AULI NURRAHMAN
4. FEBRIA RAHMANIKA
5. MUHAMMAD RAMADHAN
"""

import json
import nltk
import time
import random
import string
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import IPython.display as ipd
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import tensorflow as tf 
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder

# Package sentence tokenizer
nltk.download('punkt') 
# Package lemmatization
nltk.download('wordnet')
# Package multilingual wordnet data
nltk.download('omw-1.4')

# melakukan import dataset
with open('/content/DataSet_SuskaBot.json') as content:
  suskabot_dataset = json.load(content)
# Mendapatkan semua data ke dalam list
tags = [] 
inputs = [] 
responses = {} 
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in suskabot_dataset['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    inputs.append(lines)
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
      w = nltk.word_tokenize(pattern)
      words.extend(w)
      documents.append((w, intent['tag']))
      if intent['tag'] not in classes:
        classes.append(intent['tag'])

suskabot_dataset = pd.DataFrame({"patterns":inputs, "tags":tags})

# Cetak data keseluruhan
suskabot_dataset

# Cetak data baris pertama sampai baris kelima
suskabot_dataset.head() 

# Cetak 5 data akhir
suskabot_dataset.tail() 

#Menghilangkan Punktuasi
suskabot_dataset['patterns'] = suskabot_dataset['patterns'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
suskabot_dataset['patterns'] = suskabot_dataset['patterns'].apply(lambda wrd: ''.join(wrd))

#menghilangkan inflectional endings only dan untuk mengembalikan bentuk dictionary (kata dalam kamus) dari sebuah kata yang
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
print(len(words), "unique lemmatized words", words)

#Menyortir Data Kelas Tags
classes = sorted(list(set(classes)))
print(len(classes), "classes", classes)

#Mencari Jumlah Keseluruhan Data Teks
print(len(documents), "documents")

# Tokenisasi Data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000)
tokenizer.fit_on_texts(suskabot_dataset['patterns'])
train = tokenizer.texts_to_sequences(suskabot_dataset['patterns'])
train

# melakukan padding 
X_train = tf.keras.preprocessing.sequence.pad_sequences(train)
print(X_train)

# Encoding Label atau Tag
le = LabelEncoder()
Y_train = le.fit_transform(suskabot_dataset['tags'])
print(Y_train) 

# Input length
input_shape = X_train.shape[1]
print("Input Shape : ", input_shape)

# Define vocabulary
vocabulary = len(tokenizer.word_index)
print("Number of unique words : ", vocabulary)

# Output length
output_length = le.classes_.shape[0]
print("Output length: ", output_length)

#simpan model pemrosesan teks tersebut dengan menggunakan format pickle.
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#simpan model pemrosesan teks tersebut dengan menggunakan format pickle.
pickle.dump(le, open('le.pkl','wb'))
pickle.dump(tokenizer, open('tokenizers.pkl','wb'))

def suskabot_bot(trainx, trainy, neuron, batch_size, epochs):
    # Input Layer
    i = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Embedding(vocabulary+1,10)(i) 
    # Hidden Layer
    x = tf.keras.layers.LSTM(neuron, return_sequences = True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Hidden Layer
    x = tf.keras.layers.LSTM(neuron, return_sequences = True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Flatten Layer
    x = tf.keras.layers.Flatten()(x) 
    # Output Layer
    x = tf.keras.layers.Dense(output_length, activation="softmax")(x) 
    model  = tf.keras.models.Model(i,x)
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Menampilkan Parameter Model
    print(model.summary())
    history = model.fit(trainx,trainy,batch_size=batch_size,epochs=epochs,verbose=1,shuffle=False)
    return model, history

    # LSTM Hyperparameters
neuron = 32
batch_size = 32
epochs = 500

# Training the model
model, history_lstm = suskabot_bot(X_train, Y_train, neuron, batch_size, epochs) 

# Plot Akurasi
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'],label='Training Set Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'],label='Training Set Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# Membuat Input Chat
while True:
  texts_p = []
  prediction_input = input('Kamu : ')
  
  # Menghapus punktuasi dan konversi ke huruf kecil
  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p.append(prediction_input)

  # Tokenisasi dan Padding
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  prediction_input = tf.keras.preprocessing.sequence.pad_sequences([prediction_input],input_shape)

  # Mendapatkan hasil keluaran pada model 
  output = model.predict(prediction_input, verbose=0)
  output = output.argmax()

  response_tag = le.inverse_transform([output])[0]
  print("suskaBot : ", random.choice(responses[response_tag]))
  # tambahkan break yang berupa kata kata pada tag closing untuk mengakhiri chatbot
  if response_tag == "goodbye":
    break

#melakukan save model
model.save('Suskabot.h5')
print('Model Created Successfully!')
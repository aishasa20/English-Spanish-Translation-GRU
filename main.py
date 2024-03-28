import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout, Embedding
import matplotlib.pyplot as plt

eng_span_translations = pd.read_csv("/Users/aisha/Downloads/english-spanish-dataset.csv", nrows=50000)

# Split the dataset into English and Spanish sentences
eng_sentences = eng_span_translations['english'].tolist()
spa_sentences = eng_span_translations['spanish'].tolist()

# Tokenize the sentences
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(eng_sentences)
eng_sequences = eng_tokenizer.texts_to_sequences(eng_sentences)

spa_tokenizer = Tokenizer()
spa_tokenizer.fit_on_texts(spa_sentences)
spa_sequences = spa_tokenizer.texts_to_sequences(spa_sentences)

# Pad the sequences
max_len_eng = max([len(seq) for seq in eng_sequences])
max_len_spa = max([len(seq) for seq in spa_sequences])

eng_sequences = pad_sequences(eng_sequences, maxlen=max_len_eng, padding='post')
spa_sequences = pad_sequences(spa_sequences, maxlen=max_len_spa, padding='post')

# Split the data into training and validation sets
eng_train, eng_val, spa_train, spa_val = train_test_split(eng_sequences, spa_sequences, test_size=0.3, random_state=2023)

#Set the hyperparameter
num_dense = len(spa_tokenizer.word_index) + 1 # the output dimension of the dense layer should be the size of the Spanish vocabulary plus 1 for the padding token

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(eng_tokenizer.word_index) + 1, output_dim=128, input_length=max_len_eng))
model.add(Dropout(0.5))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(Dense(units=num_dense, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# Train the model
history = model.fit(eng_train, spa_train, validation_data=(eng_val, spa_val), epochs=5)

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Prompt the user to enter an English sentence for translation
eng_sentence = input('Enter an English sentence: ')

# Tokenize and pad the sentence
eng_seq = eng_tokenizer.texts_to_sequences([eng_sentence])
eng_seq = pad_sequences(eng_seq, maxlen=max_len_eng, padding='post')

# Translate the sentence
spa_seq = model.predict(eng_seq)
spa_seq = spa_seq.argmax(axis=-1)
spa_sentence = ' '.join([spa_tokenizer.index_word[i] for i in spa_seq[0] if i > 0])

# Print the translation
print(f'The Spanish translation is: {spa_sentence}')


import os
import pickle
import random
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

class LangInfo:
    def __init__(self, langname: str, filename: str):
        self.langname = langname
        self.filename = filename

class NamesDataset(ABC):

    @abstractmethod
    def load_names(self) -> tuple:
        pass

class NamesDatasetChar:
    def __init__(self, linfo: LangInfo):
        self.linfo = linfo
        self.X = []
        self.y = []

    def load_names(self) -> tuple:
        with open(self.linfo.filename, 'r', encoding='utf-8') as file:
            self.X = file.readlines()

        self.X = list(map(lambda s: s[:-1].lower(), self.X))

        for idx, name in enumerate(self.X):
            self.X[idx] = []
            for i in range(1, len(name)):
                self.X[idx].append(name[:i])
                self.y.append(name[i])

        return (self.X, self.y)
    
class NamesDatasetToken:
    def __init__(self, linfo: LangInfo):
        self.linfo = linfo
        self.X = []
        self.y = []
        self.tokenizer = None

    def load_names(self) -> tuple:
        with open(self.linfo.filename, 'r', encoding='utf-8') as file:
            names = [line.strip() + '<' for line in file.readlines()]

        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(names)

        sequences = self.tokenizer.texts_to_sequences(names)

        for seq in sequences:
            for i in range(1, len(seq)):
                self.X.append(seq[:i])
                self.y.append(seq[i])

        max_seq_length = max([len(seq) for seq in self.X])
        self.X = pad_sequences(self.X, maxlen=max_seq_length, padding='pre')

        self.y = tf.keras.utils.to_categorical(self.y, num_classes=len(self.tokenizer.word_index) + 1)
        return (self.X, self.y)

def train_and_save_models(datasets, save_path='../saved_models'):
    os.makedirs(save_path, exist_ok=True)

    for dataset in datasets:
        print(f"Treniram model za jezik: {dataset.linfo.langname}")
        X, y = dataset.load_names()
        
        model = Sequential()
        model.add(Embedding(input_dim=len(dataset.tokenizer.word_index)+1, output_dim=100, input_length=max(len(seq) for seq in X)))
        model.add(LSTM(units=128))
        model.add(Dense(units=len(dataset.tokenizer.word_index)+1, activation='softmax'))

        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(X, y, epochs=30, verbose=1)

        # Spremanje modela i tokenizatora
        model_save_path = os.path.join(save_path, f"{dataset.linfo.langname}_model.h5")
        tokenizer_save_path = os.path.join(save_path, f"{dataset.linfo.langname}_tokenizer.pkl")

        model.save(model_save_path)
        with open(tokenizer_save_path, 'wb') as f:
            pickle.dump(dataset.tokenizer, f)

        print(f"Model i tokenizer za {dataset.linfo.langname} spremljeni su u {save_path}")

data_path = '../data/'
datasets = [
    NamesDatasetToken(LangInfo('CRO', data_path + 'Croatia_Cities.txt')),
    NamesDatasetToken(LangInfo('CAN', data_path + 'Canada_Cities.txt')),
    NamesDatasetToken(LangInfo('GER', data_path + 'Deutschland_Cities.txt')),
    NamesDatasetToken(LangInfo('UK', data_path + 'UK_Cities.txt')),
    NamesDatasetToken(LangInfo('US', data_path + 'US_Cities.txt')),
    NamesDatasetToken(LangInfo('SPN', data_path + 'Spain_Cities.txt')),
    NamesDatasetToken(LangInfo('FRA', data_path + 'France_Cities.txt'))
]

def load_model_and_tokenizer(lang_code):
    """Učitava model i pripadajući tokenizer za određeni jezik."""
    
    # Putanje do spremljenih modela i tokenizera
    model_path = f'../models/{lang_code}_model.h5'
    tokenizer_path = f'../models/{lang_code}_tokenizer.pkl'
    
    # Učitavanje modela
    model = load_model(model_path)
    
    # Učitavanje tokenizera
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer

#trenutno staticki, inace se bira na GUI-u
# curr_country = 'CRO'
# curr_dataset = [d for d in datasets if d.linfo.langname == curr_country][0]
# X, y = curr_dataset.load_names()

# model = Sequential()
# model.add(Embedding(input_dim=len(curr_dataset.tokenizer.word_index)+1, output_dim=100, input_length=max(len(name) for name in X)+1))
# model.add(LSTM(units=128))
# model.add(Dense(units=len(curr_dataset.tokenizer.word_index)+1, activation='softmax'))

# model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Define a custom callback to store training progress
# class TrainingProgressCallback(Callback):
#     def on_train_begin(self, logs=None):
#         self.epochs = []
#         self.accuracies = []
#         self.losses = []
        
#     def on_epoch_end(self, epoch, logs=None):
#         accuracy_percentage = logs.get('accuracy') * 100
#         loss_percentage = logs.get('loss') * 100
#         self.epochs.append(epoch + 1)
#         self.accuracies.append(accuracy_percentage)
#         self.losses.append(loss_percentage)

def generate_name(model, tokenizer, max_seq_length, stop_char='<'):
    valid_chars = [char for char, index in tokenizer.word_index.items() if char != stop_char]
    # Nasumično odabiremo početni znak
    seed_text = random.choice(valid_chars)
    generated_text = seed_text.upper()  # Inicijalizira generirani tekst s nasumičnim početnim znakom
    ind = 0
    max_iterations = 100  # Ograničenje iteracija za sprječavanje beskonačne petlje
    
    while ind < max_iterations:
        print(f"Alo {ind}, Seed text: '{seed_text}'")
        
        # Pretvaranje teksta u sekvencu brojeva
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_seq_length, padding='pre')
        
        # Predikcija sljedećeg znaka
        predicted_probabilities = model.predict(sequence, verbose=0)[0]
        predicted_id = np.argmax(predicted_probabilities)
        
        # Pronalaženje znaka iz predikcije
        predicted_char = tokenizer.index_word.get(predicted_id, '')
        print(f"Predicted ID: {predicted_id}, Predicted char: '{predicted_char}'")
        
        # Provjera zaustavnog znaka ili prazne predikcije
        if predicted_char == stop_char or predicted_char == '':
            print("Prekidam petlju: pronađen <END> ili prazan znak")
            break
        
        # Dodavanje predikcije u generirani tekst
        generated_text += predicted_char
        seed_text += predicted_char
        ind += 1
    
    if ind == max_iterations:
        print("Prekida se nakon maksimalnog broja iteracija")

    return generated_text

# # Train the model with the progress callback
# progress_callback = TrainingProgressCallback()
# model.fit(X, y, epochs=100, verbose=2, callbacks=[progress_callback])

# max_seq_length = max([len(seq) for seq in X])  # Maksimalna duljina sekvenci iz trening skupa
# new_name = generate_name(model, curr_dataset.tokenizer, max_seq_length)
# print(f"Generirani naziv: {new_name}")

train_and_save_models(datasets)


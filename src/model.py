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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gui import temp


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

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

        # Podjela na trening i testni skup
        return split_data(self.X, self.y, test_size=0.2)

def train_and_save_models(datasets):
    for dataset in datasets:
        lang_code = dataset.linfo.langname
        model_path = f'../saved_models/{lang_code}_model.h5'
        tokenizer_path = f'../saved_models/{lang_code}_tokenizer.pkl'

        # Provjera postoje li već spremljeni modeli
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            print(f"Model za jezik {lang_code} već postoji, preskačem treniranje.")
            continue

        print(f"Treniram model za jezik: {lang_code}")

        # Učitavanje podataka
        X_train, X_test, y_train, y_test = dataset.load_names()

        # Model
        model = Sequential([
            Embedding(input_dim=len(dataset.tokenizer.word_index) + 1, output_dim=32, input_length=X_train.shape[1]),
            LSTM(units=128),
            Dense(units=len(dataset.tokenizer.word_index) + 1, activation='softmax')
        ])
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),
                            callbacks=[early_stopping], verbose=1)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Testni gubitak za jezik {lang_code}: {loss:.4f}, Točnost: {accuracy:.4f}")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model za jezik {lang_code} spremljen u: {model_path}")

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(dataset.tokenizer, f)
        print(f"Tokenizer za jezik {lang_code} spremljen u: {tokenizer_path}")

        plot_training_history(history, lang_code)

def plot_training_history(history, lang_code):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Trening gubitak')
    plt.plot(history.history['val_loss'], label='Validacijski gubitak')
    plt.title(f'Gubitak za {lang_code}')
    plt.xlabel('Epoha')
    plt.ylabel('Gubitak')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../saved_models/{lang_code}_loss_plot.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Trening točnost')
    plt.plot(history.history['val_accuracy'], label='Validacijska točnost')
    plt.title(f'Točnost za {lang_code}')
    plt.xlabel('Epoha')
    plt.ylabel('Točnost')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../saved_models/{lang_code}_accuracy_plot.png')
    plt.close()



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
    model_path = f'../saved_models/{lang_code}_model.h5'
    tokenizer_path = f'../saved_models/{lang_code}_tokenizer.pkl'
    model = load_model(model_path)

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer

def generate_name(model, tokenizer, max_seq_length, stop_char='<'):
    valid_chars = [char for char, index in tokenizer.word_index.items() if char.isalpha() and char != stop_char]
    seed_text = random.choice(valid_chars) # Nasumično odabiremo početni znak
    generated_text = seed_text.upper()  # Inicijalizira generirani tekst s nasumičnim početnim znakom
    capitalize_next = False  # Zastavica za veliko slovo nakon razmaka
    ind = 0
    max_iterations = 100 
    
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
        
        # Provjera da prvo slovo mora biti veliko
        if len(generated_text) == 1:
            generated_text = generated_text.upper() 
        
        # Ako je prethodni znak bio razmak, naredni znak čini velikim slovom
        if capitalize_next and predicted_char.isalpha():
            predicted_char = predicted_char.upper()
            capitalize_next = False

        # Provjera za razmaka i postavljanje zastavice
        if predicted_char == ' ':
            capitalize_next = True

        # Dodavanje predikcije u generirani tekst
        generated_text += predicted_char
        seed_text += predicted_char
        ind += 1
    
    if ind == max_iterations:
        print("Prekida se nakon maksimalnog broja iteracija")

    return generated_text

train_and_save_models(datasets)


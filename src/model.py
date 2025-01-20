import os
import pickle
import random
import numpy as np
from numpy import random
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Lambda
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch, GridSearch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


num_of_epochs = 50



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
    

def build_lstm_model(dataset, X_train, hyperparameters, embedding_units, lstm_units, lr, temperature):
    model = Sequential()

    iterations = 32
    minval, maxval = 32, 256 + 128
    step = (maxval - minval)/iterations
    model.add(Embedding(input_dim=len(dataset.tokenizer.word_index) + 1,
                         output_dim=embedding_units if embedding_units is not None else hyperparameters.Int('embedding_units', min_value=minval, max_value=maxval, step=step, sampling='log'), 
                        input_length=X_train.shape[1]))

    model.add(LSTM(units= lstm_units if lstm_units is not None else hyperparameters.Int('lstm_units', min_value=minval, max_value=maxval, step=step)))


    iterations = 40
    minval, maxval = 0.2, 2
    step = (maxval - minval)/iterations
    model.add(Lambda(lambda y: y / (temperature if temperature is not None else hyperparameters.Float('temperature', min_value=minval, max_value=maxval, step=step, sampling='linear'))))
    
    model.add(Dense(units=len(dataset.tokenizer.word_index) + 1, 
                    activation='softmax'))
    
    minval, maxval = 1e-4, 1e-2
    step = (maxval - minval)/iterations
    model.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate= lr if lr is not None else hyperparameters.Float('learning_rate', min_value=minval, max_value=maxval, step=step, sampling='linear')),
                  loss='categorical_crossentropy')
    
    return model

def tune_hyperparameters(datasets, lang, embedding_units=100, lstm_units=128, lr=0.001, temperature=0.00802, random_search=False):
    for dataset in datasets:
        lang_code = dataset.linfo.langname
        model_path = f'../saved_models/{lang_code}_model.h5'
        tokenizer_path = f'../saved_models/{lang_code}_tokenizer.pkl'

        # Provjera postoje li već spremljeni modeli
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and lang_code != lang:
            print(f"Model za jezik {lang_code} već postoji, preskačem optimiranje.")
            continue

        print(f"Optimiram model za jezik: {lang_code}")

        # Učitavanje podataka
        X_train, X_test, y_train, y_test = dataset.load_names()

        if(random_search):
            tuner = RandomSearch(
                lambda h: build_lstm_model(dataset, X_train, h, embedding_units, lstm_units, lr, temperature),
                objective='val_loss',
                max_trials=60,
                executions_per_trial=1,
                directory='tuning_results_rand',
                project_name='lstm_model_tuning_rand'
            )
        else:
            tuner = GridSearch(
                lambda h: build_lstm_model(dataset, X_train, h, embedding_units, lstm_units, lr, temperature=temperature),
                objective='val_loss',
                max_trials=60,
                executions_per_trial=1,
                directory='tuning_results',
                project_name='lstm_model_tuning'
            )

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=num_of_epochs, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        return best_hyperparameters
    
    return None

def train_and_save_models(datasets, langs, embedding_units=100, lstm_units=128, lr=0.001, temperature=0.00802):
    for dataset in datasets:
        lang_code = dataset.linfo.langname
        model_path = f'../saved_models/{lang_code}_model.h5'
        tokenizer_path = f'../saved_models/{lang_code}_tokenizer.pkl'

        # Provjera postoje li već spremljeni modeli
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and not lang_code in set(langs):
            print(f"Model za jezik {lang_code} već postoji, preskačem treniranje.")
            continue

        print(f"Treniram model za jezik: {lang_code}")

        # Učitavanje podataka
        X_train, X_test, y_train, y_test = dataset.load_names()

        # Model
        model = Sequential([
            Embedding(input_dim=len(dataset.tokenizer.word_index) + 1, output_dim=embedding_units, input_length=X_train.shape[1]),
            LSTM(units=lstm_units),
        ])

        model.add(Lambda(lambda y: y / temperature))
        model.add(Dense(units=len(dataset.tokenizer.word_index) + 1, activation='softmax'))

        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # treniranje
        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
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
        # Pretvaranje teksta u sekvencu brojeva
        sequence = tokenizer.texts_to_sequences([seed_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_seq_length, padding='pre')
        
        # Predikcija sljedećeg znaka
        predicted_probabilities = model.predict(sequence, verbose=0)[0]
        predicted_id = np.random.choice(predicted_probabilities)
        
        # Pronalaženje znaka iz predikcije
        predicted_char = tokenizer.index_word.get(predicted_id, '')
        
        # Provjera zaustavnog znaka ili prazne predikcije
        if predicted_char == stop_char or predicted_char == '':
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

    return generated_text


#tune_hyperparameters(datasets=datasets, lang="US", embedding_units=32, lstm_units=224, lr=0.00406, temperature=0.00802, random_search=False)
tune_hyperparameters(datasets=datasets, lang="US", embedding_units=None, lstm_units=None, lr=None, temperature=None, random_search=True)


#generate name with some model
# Model
#train_and_save_models(datasets=datasets, langs="US")
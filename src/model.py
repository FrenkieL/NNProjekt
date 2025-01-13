import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.text import Tokenizer

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
            names = file.readlines()

        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(names)

        sequences = tokenizer.texts_to_sequences(names)

        for seq in sequences:
            for i in range(1, len(seq)):
                self.X.append(seq[:i])
                self.y.append(seq[i])

        max_seq_length = max([len(seq) for seq in self.X])
        self.X = pad_sequences(self.X, maxlen=max_seq_length, padding='pre')

        self.y = tf.keras.utils.to_categorical(self.y, num_classes=len(tokenizer.word_index) + 1)
        return (self.X, self.y)



data_path = '../data/'
datasets = [
    NamesDatasetToken(LangInfo('CRO', data_path + 'Croatia_Cities.txt')),
    NamesDatasetToken(LangInfo('CAN', data_path + 'Canada_Citites.txt')),
    NamesDatasetToken(LangInfo('GER', data_path + 'Deutschland_Cities.txt')),
    NamesDatasetToken(LangInfo('UK', data_path + 'UK_Cities.txt')),
    NamesDatasetToken(LangInfo('US', data_path + 'US_Cities.txt')),
    NamesDatasetToken(LangInfo('SPN', data_path + 'Spain_Citites.txt')),
    NamesDatasetToken(LangInfo('FRA', data_path + 'France_Cities.txt'))
]

#trenutno staticki, inace se bira na GUI-u
curr_country = 'CRO'
curr_dataset = [d for d in datasets if d.linfo.langname == curr_country][0]
X, y = curr_dataset.load_names()

model = Sequential()
model.add(Embedding(input_dim=len(language_chars[selected_country]), output_dim="100", input_length=max(len(name) for name in city_names[selected_country])))
model.add(LSTM(units=128))
model.add(Dense(units=len(language_chars[selected_country]), activation='softmax'))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define a custom callback to store training progress
class TrainingProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = []
        self.accuracies = []
        self.losses = []
        
    def on_epoch_end(self, epoch, logs=None):
        accuracy_percentage = logs.get('accuracy') * 100
        loss_percentage = logs.get('loss') * 100
        self.epochs.append(epoch + 1)
        self.accuracies.append(accuracy_percentage)
        self.losses.append(loss_percentage)

# Train the model with the progress callback
progress_callback = TrainingProgressCallback()
model.fit(X, y, epochs=100, verbose=0, callbacks=[progress_callback])

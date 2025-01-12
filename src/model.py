import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split


files = ['Croatia_Cities.txt', 'Canada_Cities.txt', 'Deutschland_Cities.txt', 'UK_Cities.txt', 'US_Cities.txt','Spain_Cities.txt','France_Cities.txt']
countries = ['CRO', 'CAN', 'GER', 'UK', 'USA', 'SPN', 'FRA']


def get_names(filename, country_tag):

    with open(filename, 'r', encoding='utf-8') as file:
        names = file.readlines()

    names = list(map(lambda s: s[:-1].lower(), names))

    chars = [f'<START_{country_tag}>','<END>'] + sorted(set("".join(names)))

    vectors = {}

    for idx, char in enumerate(chars):
        vectors[char] = [0] * len(chars)
        vectors[char][idx] = 1

    return (names, chars, vectors)


CRO_names, CRO_chars, CRO_vectors = get_names('Croatia_Cities.txt','CRO')
CAN_names, CAN_chars, CAN_vectors = get_names('Canada_Cities.txt','CAN')
GER_names, GER_chars, GER_vectors = get_names('Deutschland_Cities.txt','GER')
UK_names, UK_chars, UK_vectors = get_names('UK_Cities.txt','UK')
USA_names, USA_chars, USA_vectors = get_names('US_Cities.txt','USA')
SPN_names, SPN_chars, SPN_vectors = get_names('Spain_Cities.txt','SPN')
FRA_names, FRA_chars, FRA_vectors = get_names('France_Cities.txt','FRA')


#char_to_idx = {char: idx for idx, char in enumerate(chars)}
#idx_to_char = {idx: char for char, idx in char_to_idx.items()}

language_chars = [CRO_chars, CAN_chars, GER_chars, UK_chars, USA_chars, SPN_chars, FRA_chars]
city_names = [CRO_names, CAN_names, GER_names, UK_names, USA_names, SPN_names, FRA_names]

# ovdje ce ulaz biti X, y, test
# X nam je sequence - [[1], [1, 2], [1, 2, 3]] (redoslijed slova imena)
# y nam je [2, 3] - sljedeca slova
# treba nam padding jer cemo sigurno imati najduze ime koje se isto tako predaje kao input length u LSTMu
X_train, X_pred, y_train, y_pred = train_test_split(X, y, test_size=0.3)

model = Sequential()
model.add(Embedding(input_dim=len(language_chars[selected_country]), output_dim="100", input_length=max(len(name) for name in city_names[selected_country])))
model.add(LSTM(units=128))
model.add(Dense(units=len(language_chars[selected_country]), activation='softmax'))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])

# batch_size -> nakon koliko se primjera updateaju tezine modela
fitting = modle.fit(X_train, y_train, validation_data(X_pred, y_pred), epochs=30, batch_size=32)

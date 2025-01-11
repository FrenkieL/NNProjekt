import numpy as np


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

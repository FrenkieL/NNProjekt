import tensorflow as tf


with open('Croatia_Cities.txt', 'r', encoding='utf-8') as file:
    CRO_names = file.readlines()

with open('Canada_Cities.txt', 'r', encoding='utf-8') as file:
    CAN_names = file.readlines()

with open('Deutschland_Cities.txt', 'r', encoding='utf-8') as file:
    GER_names = file.readlines()

with open('UK_Cities.txt', 'r', encoding='utf-8') as file:
    UK_names = file.readlines()

with open('US_Cities.txt', 'r', encoding='utf-8') as file:
    USA_names = file.readlines()

with open('Spain_Cities.txt', 'r', encoding='utf-8') as file:
    SPN_names = file.readlines()

with open('France_Cities.txt', 'r', encoding='utf-8') as file:
    FRA_names = file.readlines()


CRO_names = list(map(lambda s: s[:-1].lower(), CRO_names))
CAN_names = list(map(lambda s: s[:-1].lower(), CAN_names))
GER_names = list(map(lambda s: s[:-1].lower(), GER_names))
UK_names = list(map(lambda s: s[:-1].lower(), UK_names))
USA_names = list(map(lambda s: s[:-1].lower(), USA_names))
SPN_names = list(map(lambda s: s[:-1].lower(), SPN_names))
FRA_names = list(map(lambda s: s[:-1].lower(), FRA_names))



countries = ['CRO', 'CAN', 'GER', 'UK', 'USA', 'SPN', 'FRA']
chars = [f'<START_{tag}>' for tag in countries] + ['<END>'] + sorted(set("".join(CRO_names + CAN_names + GER_names + UK_names + USA_names + SPN_names + FRA_names)))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}




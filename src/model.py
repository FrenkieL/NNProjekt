with open('imena_naselja.txt', 'r', encoding='utf-8') as file:
    dataset = file.readlines()

dataset = list(map(lambda s: s[:-1].lower(), dataset))
    
#print(dataset)


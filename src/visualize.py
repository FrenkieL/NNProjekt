import os
import json
from model import num_of_epochs

vis_data = []
rootdir = 'random_search/fashion_mnist'
for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("trial.json"):
          with open(subdirs + '/' + file, 'r') as json_file:
            data = json_file.read()
          vis_data.append(json.loads(data))

import hiplot as hip

data = [{'num_filters_1': vis_data[idx]['hyperparameters']['values']['num_filters_1'],
         'num_filters_2': vis_data[idx]['hyperparameters']['values']['num_filters_2'], 
         'units': vis_data[idx]['hyperparameters']['values']['units'], 
         'dense_activation': vis_data[idx]['hyperparameters']['values']['dense_activation'], 
         'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
         'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],  
         'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value'], 
         'accuracy': vis_data[idx]['metrics']['metrics']['accuracy']['observations'][0]['value'],
         'val_accuracy': vis_data[idx]['metrics']['metrics']['val_accuracy']['observations'][0]['value']} for idx in range(num_of_epochs)]

hip.Experiment.from_iterable(data).display()
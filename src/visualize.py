import os
import json
import sys
import hiplot as hip
from model import num_of_epochs
from streamlit.web import cli as stcli
from streamlit import runtime


vis_data = []
rootdir = 'tuning_results_rand/lstm_model_tuning_rand'
for subdirs, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("trial.json"):
            with open(subdirs + '/' + file, 'r') as json_file:
                data = json_file.read()
            vis_data.append(json.loads(data))

data = [{
        'lstm_units': vis_data[idx]['hyperparameters']['values']['lstm_units'], 
        'embedding_units': vis_data[idx]['hyperparameters']['values']['embedding_units'], 
        'temperature': vis_data[idx]['hyperparameters']['values']['temperature'], 
        'learning_rate': vis_data[idx]['hyperparameters']['values']['learning_rate'], 
        'loss': vis_data[idx]['metrics']['metrics']['loss']['observations'][0]['value'],  
        'val_loss': vis_data[idx]['metrics']['metrics']['val_loss']['observations'][0]['value']
        } for idx in range(num_of_epochs)]

print(len(vis_data))
hip.Experiment.from_iterable(data).display()


from config import Configuration
from embedder import train_embedder
import os
import pandas as pd



data_name = 'SPECT.csv'
print(data_name)

config_data = Configuration(train_batch_size=10, val_batch_size=10,
                            embedding_size=10, max_iteration=10000,
                            learning_rate=0.01, print_loss_every=500,
                            LOG_DIR = './{}/embedding/log/'.format(data_name),
                            model_save_filename = 'model.ckpt',
                            metadata_filename = 'metadata.tsv')



if not os.path.exists(config_data.LOG_DIR):
    os.makedirs(config_data.LOG_DIR)

with open(os.path.join(config_data.LOG_DIR, 'config.txt'), 'w') as f:
    f.write(repr(config_data))

data = pd.read_csv('data/{}'.format(data_name)).drop('class',axis=1)

for col in data.columns:
    if data[col].dtype == int or data[col].dtype == float:
        data[col] = data[col].astype('str')

train_embedder(data, config_data)
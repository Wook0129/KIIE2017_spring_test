from config import Configuration
from embedder import train_embedder
import os
import pandas as pd



data_name = 'Mushroom'
print(data_name)

data = pd.read_csv('data/{}.csv'.format(data_name)).drop('class',axis=1)

embedding_size = 10
config_data = Configuration(train_batch_size=128, val_batch_size=len(data) * 0.2,
                            embedding_size=embedding_size, max_iteration=10000,
                            learning_rate=0.01, print_loss_every=500,
                            LOG_DIR = './{}/embedding/{}/log/'.format(data_name, embedding_size),
                            model_save_filename = 'model.ckpt',
                            metadata_filename = 'metadata.tsv',
                            corruption_ratio=0.2)

if not os.path.exists(config_data.LOG_DIR):
    os.makedirs(config_data.LOG_DIR)

with open(os.path.join(config_data.LOG_DIR, 'config.txt'), 'w') as f:
    f.write(repr(config_data))

for col in data.columns:
    if data[col].dtype == int or data[col].dtype == float:
        data[col] = data[col].astype('str')

train_embedder(data, config_data)
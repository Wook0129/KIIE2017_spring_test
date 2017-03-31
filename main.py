import pandas as pd
from embedder import train_embedder

data = pd.read_csv('data/Davis.csv')[['weight','height','sex']]

train_embedder(data)
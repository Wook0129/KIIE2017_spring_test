import pandas as pd
from embedder import train_embedder

# data = pd.read_csv('data/Davis.csv')[['weight','height','sex']]

data = pd.read_csv('data/Mushroom.csv').drop('class',axis=1)

train_embedder(data)
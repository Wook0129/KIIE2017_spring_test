import pandas as pd
from embedder import train_embedder
import urllib.request
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/car/Davis.csv'
response = urllib.request.urlopen(url)
data = pd.read_csv(response, header=0)[['weight','height','sex']]
train_embedder(data)
import pandas as pd
import numpy as np
import pickle as pkl

# Load METR-LA
data = pd.read_hdf(
    "./data/METR-LA/metr-la.h5"
)

print("Original shape:", data.shape)

# Missing values check
print("Missing values:", data.isnull().sum().sum())

# Convert to numpy
data = data.values

print("Numpy shape:", data.shape)

# Load adjacency
with open(
    "./data/METR-LA/adj_mx.pkl",
    "rb"
) as f:

    adj = pkl.load(f)

print("Adjacency shape:", adj.shape)

print("Nodes in data:", data.shape[1])
print("Nodes in adjacency:", adj.shape[0])
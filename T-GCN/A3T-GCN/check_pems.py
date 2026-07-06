import numpy as np
import pickle as pkl

print("Loading PEMS...")

data = np.load(
    "../data/PEMSBAY/PEMSBAY_2022.npy",
    allow_pickle=True
)

print("Raw shape:", data.shape)

for i in range(data.shape[2]):
    print("\nFeature", i)
    print("Min:", np.min(data[:,:,i]))
    print("Max:", np.max(data[:,:,i]))
    print("Mean:", np.mean(data[:,:,i]))

print("Processed shape:", data.shape)

with open(
    "../data/PEMSBAY/adj_mx_bay.pkl",
    "rb"
) as f:

    sensor_ids, sensor_dict, adj = pkl.load(
        f,
        encoding='latin1'
    )

print("Adj shape:", adj.shape)

print("Missing values:", np.isnan(data).sum())
print("Infinite values:", np.isinf(data).sum())
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
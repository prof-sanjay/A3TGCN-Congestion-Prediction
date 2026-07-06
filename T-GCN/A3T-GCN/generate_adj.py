import pandas as pd
import numpy as np
import pickle as pkl
from math import radians, sin, cos, sqrt, atan2

# Load sensor coordinates
df = pd.read_csv(
    "../data/METR-LA/graph_sensor_locations.csv"
)

coords = df[['latitude','longitude']].values

num_nodes = len(coords)

distance_matrix = np.zeros((num_nodes,num_nodes))


# Haversine distance
def haversine(lat1,lon1,lat2,lon2):

    R = 6371

    lat1,lon1,lat2,lon2 = map(
        radians,
        [lat1,lon1,lat2,lon2]
    )

    dlat = lat2-lat1
    dlon = lon2-lon1

    a = (
        sin(dlat/2)**2
        +
        cos(lat1)
        *cos(lat2)
        *sin(dlon/2)**2
    )

    c = 2*atan2(
        sqrt(a),
        sqrt(1-a)
    )

    return R*c


# Compute pairwise distances
for i in range(num_nodes):
    for j in range(num_nodes):

        lat1,lon1=coords[i]
        lat2,lon2=coords[j]

        distance_matrix[i,j]=haversine(
            lat1,lon1,
            lat2,lon2
        )


# Gaussian adjacency
sigma=np.std(distance_matrix)

adj=np.exp(
    -(distance_matrix**2)/(sigma**2)
)

# remove weak connections
adj[adj<0.1]=0

# self-connections
np.fill_diagonal(adj,1)

print(adj.shape)

with open(
    "../data/METR-LA/adj_mx.pkl",
    "wb"
) as f:

    pkl.dump(adj,f)

print("adjacency saved")
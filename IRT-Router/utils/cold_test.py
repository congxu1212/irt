import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings_dir = "zhipu"
cold_dir = "cold"

with open(f"{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
    query_embeddings_data = pickle.load(f)

query_df = pd.read_csv("map/query.csv")

train_indices = query_df[query_df["split"] == "train"]["index"].tolist()
test_indices = query_df[query_df["split"] == "test"]["index"].tolist()

train_embeddings = np.array([entry["embedding"] for entry in query_embeddings_data if entry["index"] in train_indices])
test_embeddings = np.array([entry["embedding"] for entry in query_embeddings_data if entry["index"] in test_indices])


knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(train_embeddings)


distances, indices = knn.kneighbors(test_embeddings)


test_avg_embeddings = []
for test_idx, neighbors_indices in zip(test_indices, indices):
    neighbors_embeddings = [train_embeddings[i] for i in neighbors_indices]
    avg_embedding = np.mean(neighbors_embeddings, axis=0)
    test_avg_embeddings.append({"index": test_idx, "avg_embedding": avg_embedding})


with open(f"{cold_dir}/test_avg_embeddings_{embeddings_dir}.pkl", "wb") as f:
    pickle.dump(test_avg_embeddings, f)

print(f"Average embeddings for test set saved successfully in {cold_dir}/test_avg_embeddings__{embeddings_dir}.pkl")
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings_dir = "bert"

with open(f"{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
    query_embeddings_data = pickle.load(f)


query_df = pd.read_csv("map/query.csv")


train_indices = query_df[query_df["split"] == "train"]["index"].tolist()
test_indices = query_df[query_df["split"] == "test"]["index"].tolist()

train_embeddings = np.array([entry["embedding"] for entry in query_embeddings_data if entry["index"] in train_indices])
test_embeddings = np.array([entry["embedding"] for entry in query_embeddings_data if entry["index"] in test_indices])


with open(f"relevance/relevance_vectors_cluster_train_{embeddings_dir}.pkl", "rb") as f:
    relevance_data = pickle.load(f)


index_to_relevance = {entry["index"]: entry["relevance_vector"] for entry in relevance_data}


knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(train_embeddings)


distances, indices = knn.kneighbors(test_embeddings)


test_relevance_vectors = []
for test_idx, neighbors_indices in zip(test_indices, indices):
    neighbors_relevance_vectors = [index_to_relevance[train_indices[i]] for i in neighbors_indices]
    avg_relevance_vector = np.mean(neighbors_relevance_vectors, axis=0)
    test_relevance_vectors.append({"index": test_idx, "relevance_vector": avg_relevance_vector})


with open(f"relevance/relevance_vectors_cluster_test_{embeddings_dir}.pkl", "wb") as f:
    pickle.dump(test_relevance_vectors, f)

print(f"Relevance vectors for test set saved successfully in relevance/relevance_vectors_cluster_test_{embeddings_dir}.pkl")
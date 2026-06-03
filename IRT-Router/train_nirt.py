import logging
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from router import NIRT
import torch
import pickle

emb_name = "bert"


def load_embeddings(embeddings_dir):
    with open(f"utils/{embeddings_dir}_embeddings/llm_embeddings.pkl", "rb") as f:
        llm_embeddings_data = pickle.load(f)
    llm_embeddings = {llm["index"]: np.array(llm["embedding"]) for llm in llm_embeddings_data}
    with open(f"utils/{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
        query_embeddings_data = pickle.load(f)
    query_embeddings = {query["index"]: np.array(query["embedding"]) for query in query_embeddings_data}
    
    llm_id_map = pd.read_csv(f"utils/map/llm.csv", index_col="name").to_dict()["index"]
    query_id_map = pd.read_csv(f"utils/map/query.csv", index_col="question").to_dict()["index"]
    return llm_embeddings, query_embeddings, llm_id_map, query_id_map


def load_relevance_vectors(relevance_file):
    with open(relevance_file, "rb") as f:
        relevance_data = pickle.load(f)
    relevance_vectors = {entry["index"]: np.array(entry["relevance_vector"]) for entry in relevance_data}
    return relevance_vectors


def map_ids_to_vectors(data, llm_embeddings, query_embeddings, llm_id_map, query_id_map, relevance_vectors=None):
    llm_vectors = []
    query_vectors = []
    relevance_vecs = [] 
    for _, row in data.iterrows():
        llm_id = llm_id_map[row["llm"]]
        query_id = query_id_map[row['question']]
        
        llm_vectors.append(llm_embeddings[llm_id])
        query_vectors.append(query_embeddings[query_id])
        
        if relevance_vectors is not None:
            relevance_vecs.append(relevance_vectors.get(query_id, np.ones(25))) 
        
    return np.array(llm_vectors), np.array(query_vectors), np.array(relevance_vecs)


train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test1.csv")


llm_embeddings, query_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)


train_relevance_vectors = load_relevance_vectors(f"utils/relevance/relevance_vectors_cluster_train_{emb_name}.pkl")
test_relevance_vectors = load_relevance_vectors(f"utils/relevance/relevance_vectors_cluster_test_{emb_name}.pkl")


train_llm, train_query, train_relevance = map_ids_to_vectors(train_data, llm_embeddings, query_embeddings, llm_id_map, query_id_map, train_relevance_vectors)
test_llm, test_query, test_relevance = map_ids_to_vectors(test_data, llm_embeddings, query_embeddings, llm_id_map, query_id_map, test_relevance_vectors)

batch_size = 512
train_set = DataLoader(TensorDataset(
    torch.tensor(train_llm, dtype=torch.float32),
    torch.tensor(train_query, dtype=torch.float32),
    torch.tensor(train_relevance, dtype=torch.float32),  
    torch.tensor(train_data["performance"].values, dtype=torch.float32)
), batch_size=batch_size, shuffle=True)

test_set = DataLoader(TensorDataset(
    torch.tensor(test_llm, dtype=torch.float32),
    torch.tensor(test_query, dtype=torch.float32),
    torch.tensor(test_relevance, dtype=torch.float32),  
    torch.tensor(test_data["performance"].values, dtype=torch.float32)
), batch_size=batch_size, shuffle=False)


if emb_name == "open":
    query_dim = 1536
    llm_dim = 1536
    knowledge_n = 25
elif emb_name == "zhipu":
    query_dim = 512
    llm_dim = 512
    knowledge_n = 25
elif emb_name == "bge":
    query_dim = 1024
    llm_dim = 1024
    knowledge_n = 25
elif emb_name == "bert":
    query_dim = 768
    llm_dim = 768
    knowledge_n = 25

logging.getLogger().setLevel(logging.INFO)
cdm = NIRT.NIRT(llm_dim, query_dim, knowledge_n)
cdm.train(train_set, test_set, epoch=5, device="cuda", lr=0.001)
cdm.save(f"nirt_{emb_name}.snapshot")

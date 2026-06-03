# save_embeddings.py
import pandas as pd
import numpy as np
import embedding
import os
# from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import pickle


def parallel_embedding(data, func, batch_size=16, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(func, data[i:i+batch_size])
            for i in range(0, len(data), batch_size)
        ]
        results = []
        for future in futures:
            results.extend(future.result())
    return results


def save_embeddings(llm_id_map_path, query_id_map_path, output_dir, batch_size=16, max_workers=4):
    llm_id_map = pd.read_csv(llm_id_map_path)
    query_id_map = pd.read_csv(query_id_map_path)

    llm_inputs = [f"{profile}" for profile in llm_id_map["profile"].tolist()]
    llm_ids = llm_id_map["index"].tolist()
    print(f"Generating embeddings for {len(llm_inputs)} llms...")
    # llm_embeddings = parallel_embedding(llm_inputs, get_llm_embedding, processes)
    llm_embeddings = parallel_embedding(
        llm_inputs,
        lambda batch: embedding.bert(batch),
        batch_size=batch_size,
        max_workers=max_workers
    )
    os.makedirs(output_dir, exist_ok=True)
    llm_data = [{"index": idx, "embedding": vec} for idx, vec in zip(llm_ids, llm_embeddings)]
    with open(os.path.join(output_dir, "llm_embeddings.pkl"), "wb") as f:
        pickle.dump(llm_data, f)
    print(f"llm embeddings saved to {output_dir}/llm_embeddings.pkl")
    
    query_inputs = [f"{question}" for question in query_id_map["question"].tolist()]
    # print(query_inputs[0])
    query_ids = query_id_map["index"].tolist()
    print(f"Generating embeddings for {len(query_inputs)} querys...")
    # query_embeddings = parallel_embedding(
    #     query_inputs,
    #     get_query_embedding,
    #     processes,
    # )
    query_embeddings = parallel_embedding(
        query_inputs,
        lambda batch: embedding.bert(batch),
        batch_size=batch_size,
        max_workers=max_workers
    )
    query_data = [{"index": idx, "embedding": vec} for idx, vec in zip(query_ids, query_embeddings)]
    with open(os.path.join(output_dir, "query_embeddings.pkl"), "wb") as f:
        pickle.dump(query_data, f)
    print(f"query embeddings saved to {output_dir}/query_embeddings.pkl")
    print(f"All embeddings saved successfully in {output_dir}")



if __name__ == "__main__":
    save_embeddings(
        "map/llm.csv",
        "map/query.csv",
        "bert_embeddings",
        batch_size=16,
        max_workers=4
    )
    
    
    
    
import pandas as pd
import numpy as np
from router import NIRT, MIRT
import torch
import pickle
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'glm_4_air': {
        'input_cost': 0.137e-6,
        'output_cost': 0.137e-6
    },
    'glm_4_flash': {
        'input_cost': 0.0137e-6,
        'output_cost': 0.0137e-6
    },
    'glm_4_plus': {
        'input_cost': 6.85e-6,
        'output_cost': 6.85e-6
    },
    'gpt_4o': {
        'input_cost': 2.5e-6,
        'output_cost': 10e-6
    },
    'gpt_4o_mini': {
        'input_cost': 0.15e-6,
        'output_cost': 0.6e-6
    },
    'gpt_4o_mini_cot': {
        'input_cost': 0.15e-6,
        'output_cost': 0.6e-6
    },
    'deepseek_coder': {
        'input_cost': 0.14e-6,
        'output_cost': 0.28e-6
    },
    'deepseek_chat': {
        'input_cost': 0.14e-6,
        'output_cost': 0.28e-6
    },
    'qwen25_32b_int4': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'qwen25_7b_instruct': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'qwen25_72b_instruct': {
        'input_cost': 1.08e-6,
        'output_cost': 1.08e-6
    },
    'qwq_32b_preview': {
        'input_cost': 1.2e-6,
        'output_cost': 1.2e-6
    },
    'qwen25_math_7b_instruct': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'llama31_8b_instruct': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'llama31_70b_instruct': {
        'input_cost': 0.792e-6,
        'output_cost': 0.792e-6
    },
    'llama31_405b_instruct': {
        'input_cost': 3.15e-6,
        'output_cost': 3.15e-6
    },
    'mixtral_8x7b_instruct': {
        'input_cost': 0.54e-6,
        'output_cost': 0.54e-6
    },
    'mistral_7b_instruct_v02': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'ministral_8b_instruct_2410': {
        'input_cost': 0.1e-6,
        'output_cost': 0.2e-6
    },
    'gemini15_flash': {
        'input_cost': 0.075e-6,
        'output_cost': 0.3e-6
    },
    'claude35_haiku20241022': {
        'input_cost': 0.8e-6,
        'output_cost': 4e-6
    },
}


def load_embeddings(embeddings_dir):
    with open(f"utils/{embeddings_dir}_embeddings/llm_embeddings.pkl", "rb") as f:
        llm_embeddings_data = pickle.load(f)
    llm_embeddings = {llm["index"]: np.array(llm["embedding"]) for llm in llm_embeddings_data}
    
    with open(f"utils/{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
        query_embeddings_data = pickle.load(f)
    query_embeddings = {query["index"]: np.array(query["embedding"]) for query in query_embeddings_data}
    
    with open(f"utils/relevance/relevance_vectors_cluster_test_{embeddings_dir}.pkl", "rb") as f:
        relevance_embeddings_data = pickle.load(f)
    relevance_embeddings = {relevance["index"]: np.array(relevance["relevance_vector"]) for relevance in relevance_embeddings_data}
    
    with open(f"utils/cold/test_avg_embeddings_{embeddings_dir}.pkl", "rb") as f:
        cold_embeddings_data = pickle.load(f)
    cold_embeddings = {cold["index"]: np.array(cold["avg_embedding"]) for cold in cold_embeddings_data}
    
    llm_id_map = pd.read_csv(f"utils/map/llm.csv", index_col="name").to_dict()["index"]
    query_id_map = pd.read_csv(f"utils/map/query.csv", index_col="question").to_dict()["index"]
    return llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map


def map_ids_to_vectors(llm_name, question, llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map):
    llm_id = llm_id_map[llm_name]
    query_id = query_id_map[question]
    return np.array(llm_embeddings[llm_id]), np.array(query_embeddings[query_id]), np.array(cold_embeddings.get(query_id, np.zeros_like(query_embeddings[query_id]))), np.array(relevance_embeddings.get(query_id, np.ones(25)))




def main(router, emb_name, test_path, a, lamda):
    test_data = pd.read_csv(f"data/{test_path}.csv")
    llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map = load_embeddings(emb_name)
    
    llms = [
        'glm_4_air',
        'glm_4_flash',
        'glm_4_plus',
        'gpt_4o',
        'gpt_4o_mini',
        'gpt_4o_mini_cot',
        'deepseek_coder',
        'deepseek_chat',
        'qwen25_32b_int4',
        'qwen25_7b_instruct',
        'qwen25_72b_instruct',
        'qwq_32b_preview',
        'qwen25_math_7b_instruct',
        'llama31_8b_instruct',
        'llama31_70b_instruct',
        'llama31_405b_instruct',
        'mixtral_8x7b_instruct',
        'mistral_7b_instruct_v02',
        'ministral_8b_instruct_2410',
        'gemini15_flash',
    ]
    
    print(a)
    a = a 
    b = - (1 - a) 

    final_performance = []
    final_cost = []

    unique_questions = test_data['question'].unique()
    # print(len(unique_questions))

    for question in unique_questions:
        if router == "large":
            llm_name = "gpt_4o"
            best_row = test_data[(test_data['question'] == question) & (test_data['llm'] == llm_name)].iloc[0]
            final_performance.append(best_row['performance'])
            final_cost.append(best_row['input_tokens'] * config[best_row["llm"]]["input_cost"] + best_row['output_tokens'] * config[best_row["llm"]]["output_cost"])
        elif router == "small":
            llm_name = "ministral_8b_instruct_2410"
            best_row = test_data[(test_data['question'] == question) & (test_data['llm'] == llm_name)].iloc[0]
            final_performance.append(best_row['performance'])
            final_cost.append(best_row['input_tokens'] * config[best_row["llm"]]["input_cost"] + best_row['output_tokens'] * config[best_row["llm"]]["output_cost"])
        else:
            if router == "nirt":
                if emb_name == "open":
                    query_dim = 1536
                    llm_dim = 1536
                elif emb_name == "zhipu":
                    query_dim = 512
                    llm_dim = 512
                elif emb_name == "bge":
                    query_dim = 1024
                    llm_dim = 1024
                elif emb_name == "bert":
                    query_dim = 768
                    llm_dim = 768
                knowledge_n = 25
                cdm = NIRT.NIRT(llm_dim, query_dim, knowledge_n)
                cdm.load(f"nirt_{emb_name}.snapshot")
            elif router == "mirt":
                if emb_name == "open":
                    query_dim = 1536
                    llm_dim = 1536
                elif emb_name == "zhipu":
                    query_dim = 512
                    llm_dim = 512
                elif emb_name == "bge":
                    query_dim = 1024
                    llm_dim = 1024
                elif emb_name == "bert":
                    query_dim = 768
                    llm_dim = 768
                knowledge_n = 25
                cdm = MIRT.MIRT(llm_dim, query_dim, knowledge_n)
                cdm.load(f"mirt_{emb_name}.snapshot")

            
            candidates = []
            for llm_name in llms:
                llm_vector, query_vector, cold_vector, relevance_vector = map_ids_to_vectors(llm_name, question, llm_embeddings, query_embeddings, cold_embeddings, relevance_embeddings, llm_id_map, query_id_map)
                if router == "nirt":
                    query_vector = (1 - lamda) * query_vector + lamda * cold_vector
                    performance_pred = cdm.generate(torch.Tensor(llm_vector), torch.Tensor(query_vector), torch.Tensor(relevance_vector), device=device)
                elif router == "mirt":
                    query_vector = (1 - lamda) * query_vector + lamda * cold_vector
                    performance_pred = cdm.generate(torch.Tensor(llm_vector), torch.Tensor(query_vector), device=device)
                else:
                    performance_pred = cdm.generate(torch.Tensor(llm_vector), torch.Tensor(query_vector), device=device)
                cost_llm = config[llm_name]["output_cost"] * 1e5
                score = a * performance_pred + b * cost_llm
                candidates.append((llm_name, score, performance_pred, cost_llm))

            best_llm = max(candidates, key=lambda x: x[1])
            best_row = test_data[(test_data['question'] == question) & (test_data['llm'] == best_llm[0])].iloc[0]
            final_performance.append(best_row['performance'])
            final_cost.append(best_row['input_tokens'] * config[best_row["llm"]]["input_cost"] + best_row['output_tokens'] * config[best_row["llm"]]["output_cost"])

    avg_performance = np.mean(final_performance)
    total_cost = np.sum(final_cost)

    print(f"Performance: {avg_performance}")
    print(f"Total Cost: {total_cost}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", type=str, default="mirt", help="Router type (e.g., 'mirt', 'nirt', etc.)")
    parser.add_argument("--emb_name", type=str, default="open", help="Embedding name (e.g., 'open', 'bert')")
    parser.add_argument("--test_path", type=str, default="test1", help="test path")
    parser.add_argument("--a", type=float, default=0.8, help="a")
    parser.add_argument("--lamda", type=float, default=0.1, help="lamda")
    args = parser.parse_args()
    main(args.router, args.emb_name, args.test_path, args.a, args.lamda)
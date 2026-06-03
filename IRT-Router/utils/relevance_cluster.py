import pandas as pd
import numpy as np
import pickle
import umap
import hdbscan
from openai import OpenAI
from sklearn.metrics import pairwise_distances_argmin_min

emb_name = "bert"

abilities = [
    "Reasoning", "Understanding", "Generation", "Information retrieval", 
    "Multidisciplinary knowledge", "Emotion understanding and expression", 
    "Adaptability and robustness", "Interactive", "Ethical and moral consideration", 
    "Mathematical calculation", "Data analysis", "Symbolic processing", 
    "Geometric and spatial reasoning", "Programming and algorithms", 
    "Scientific knowledge application", "Technical documentation understanding", 
    "Current affairs and common knowledge", "Cultural understanding", 
    "Language conversion", "Music and art understanding", 
    "Editing and proofreading", "Prediction and hypothesis testing", 
    "Inference", "Decision support", "Content summarization"
]

def parse(response):
    return response.strip().split(", ")


def get_abilities_for_question_mini(question):
    prompt = f"""
    You will be provided with the following query: {question}\n
    Identify which of the following abilities it requires from the LLM: {', '.join(abilities)}.\n
    Output the abilities as a comma-separated list.
    """
    client = OpenAI(
            api_key=""
        )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            top_p=0.9,
            temperature=0.7,
            max_tokens=2000
        )
    except:
        return ","
    response = response.choices[0].message.content if response and response.choices else ""
    return parse(response)


def get_train_embeddings(embeddings_dir, split_csv_path):
    with open(f"{embeddings_dir}_embeddings/query_embeddings.pkl", "rb") as f:
        query_embeddings_data = pickle.load(f)
    split_data = pd.read_csv(split_csv_path)
    train_data = split_data[split_data["split"] == "train"]
    train_questions = train_data["question"].tolist()
    train_indices = train_data["index"].tolist()
    train_embeddings = np.array([entry["embedding"] for entry in query_embeddings_data if entry["index"] in train_indices])
    return train_embeddings, train_indices, train_questions


def reduce_and_cluster(train_embeddings, n_components=5):
    reducer = umap.UMAP(n_components=n_components)
    reduced_embeddings = reducer.fit_transform(train_embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(reduced_embeddings)
    return reduced_embeddings, labels


def select_samples_per_cluster(questions, labels, num_samples=5):
    cluster_samples = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        selected_indices = np.random.choice(cluster_indices, min(num_samples, len(cluster_indices)), replace=False)
        cluster_samples[label] = [questions[i] for i in selected_indices]
    return cluster_samples


def create_relevance_vector_for_cluster(sample_questions, abilities, get_abilities_function):
    llm_abilities = get_abilities_function(sample_questions)
    return [1 if ability in llm_abilities else 0 for ability in abilities]


def assign_relevance_vectors_to_questions(indices, labels, cluster_vectors):
    relevance_vectors = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_vector = cluster_vectors[label]
        cluster_indices = np.where(labels == label)[0]
        for index in cluster_indices:
            relevance_vectors[indices[index]] = cluster_vector
    return relevance_vectors


def assign_relevance_vector_for_noise(indices, reduced_embeddings, labels, cluster_vectors):
    noise_indices = np.where(labels == -1)[0]
    relevance_vectors = {}
    if noise_indices.size > 0:
        cluster_centers = {label: reduced_embeddings[np.where(labels == label)].mean(axis=0) for label in set(labels) if label != -1}
        cluster_labels = list(cluster_centers.keys())
        cluster_center_embeddings = np.array(list(cluster_centers.values()))
        closest_clusters, _ = pairwise_distances_argmin_min(reduced_embeddings[noise_indices], cluster_center_embeddings)
        for i, noise_index in enumerate(noise_indices):
            nearest_cluster_label = cluster_labels[closest_clusters[i]]
            relevance_vectors[indices[noise_index]] = cluster_vectors[nearest_cluster_label]
    return relevance_vectors


train_embeddings, train_indices, train_questions = get_train_embeddings(emb_name, "map/query.csv")
reduced_embeddings, labels = reduce_and_cluster(train_embeddings)
cluster_samples = select_samples_per_cluster(train_questions, labels)
cluster_vectors = {label: create_relevance_vector_for_cluster(samples, abilities, get_abilities_for_question_mini) for label, samples in cluster_samples.items()}
relevance_vectors = assign_relevance_vectors_to_questions(train_indices, labels, cluster_vectors)
noise_relevance_vectors = assign_relevance_vector_for_noise(train_indices, reduced_embeddings, labels, cluster_vectors)
relevance_vectors.update(noise_relevance_vectors)


relevance_data = [{"index": idx, "relevance_vector": vec} for idx, vec in relevance_vectors.items()]

with open(f"relevance/relevance_vectors_cluster_train_{emb_name}.pkl", 'wb') as f:
    pickle.dump(relevance_data, f)

print("Relevance vectors saved successfully in relevance/relevance_vectors_cluster_train.pkl")
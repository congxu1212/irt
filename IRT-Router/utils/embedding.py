import os
from zhipuai import ZhipuAI
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

os.environ["ZHIPUAI_API_KEY"] = ""


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def l2_normalize(vector):
    norm = np.linalg.norm(vector, ord=2)
    if norm == 0:
        return vector
    return vector / norm


def zhipu(inputs):
    client = ZhipuAI()
    response = client.embeddings.create(
        model="embedding-3",
        input=inputs,
        dimensions=512
    )
    if response.data and len(response.data) > 0:
        return [item.embedding for item in response.data]
    else:
        raise ValueError("No embeddings found in the response")
    

def open(inputs):
    client = OpenAI(
        api_key=""
    )
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=inputs,
    )
    if response.data and len(response.data) > 0:
        return [item.embedding for item in response.data]
    else:
        raise ValueError("No embeddings found in the response")


def bert(inputs):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained('')
    model = BertModel.from_pretrained('')
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)
    model = model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
    sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
    embeddings_list = sentence_embeddings.tolist()
    normalized_embeddings_list = l2_normalize(np.array(embeddings_list))
    return normalized_embeddings_list.tolist()


def bge_m3(inputs):
    bge_m3_tokenizer = AutoTokenizer.from_pretrained('')
    bge_m3_model = AutoModel.from_pretrained('')
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # Tokenize sentences
    encoded_input = bge_m3_tokenizer(inputs, padding=True, truncation=True, return_tensors='pt').to(device)
    bge_m3_model_input = bge_m3_model.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = bge_m3_model_input(**encoded_input)
        embeddings = model_output.last_hidden_state
    # return embeddings.mean(dim=1)
    
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Convert to a list and return as JSON response
    embeddings_list = sentence_embeddings.tolist()
    # normalize handle
    normalized_embeddings_list = l2_normalize(np.array(embeddings_list))
    return normalized_embeddings_list.tolist()

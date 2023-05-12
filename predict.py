import os

import torch.cuda
from sentence_transformers import util
import json


def get_top_k(model, query, corpus, top_k=10, pre_train=False, refresh=False):
    save_path = 'top_k_indices.json'
    if pre_train:
        save_path = 'top_k_indices_pre_train.json'
    if os.path.exists(save_path) and not refresh:
        with open(save_path, 'r') as f:
            tok_k_indices = json.load(f)
        return tok_k_indices

    tok_k_indices = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=True).to(device)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True).to(device)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    for hit in hits:
        hit = sorted(hit, key=lambda x: x['score'], reverse=True)
        tok_k_indices.append([corpus['corpus_id'] for corpus in hit])

    # store the top k indices in a json file
    with open('save_path', 'w') as f:
        json.dump(tok_k_indices, f)
    return tok_k_indices

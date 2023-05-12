import os
import pickle

import torch.cuda
from sentence_transformers import util
import json


def get_top_k(model, query, corpus, top_k=10, pre_train=False, refresh=False, new_evidence=False):
    save_path = 'output/top_k_indices.json'
    corpus_embeddings_path = 'output/corpus_embeddings.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pre_train:
        save_path = 'top_k_indices_pre_train.json'
    if os.path.exists(save_path) and not refresh:
        with open(save_path, 'r') as f:
            tok_k_indices = json.load(f)
        return tok_k_indices

    tok_k_indices = []
    model.to(device)
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False).to(device)
    if new_evidence or not os.path.exists(corpus_embeddings_path):
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False).to(device)
        pickle.dump(corpus_embeddings, open(corpus_embeddings_path, 'wb'))
    else:
        corpus_embeddings = pickle.load(open(corpus_embeddings_path, 'rb'))
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    for hit in hits:
        hit = sorted(hit, key=lambda x: x['score'], reverse=True)
        tok_k_indices.append([corpus['corpus_id'] for corpus in hit])

    # store the top k indices in a json file
    with open('save_path', 'w') as f:
        json.dump(tok_k_indices, f)
    return tok_k_indices
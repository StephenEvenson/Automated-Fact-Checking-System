import os
import pickle

import torch.cuda
from sentence_transformers import util


def get_top_k(model, query, corpus, top_k=10, refresh=False):
    corpus_embeddings_path = 'output/corpus_embeddings.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_k_indices = []
    model.to(device)
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False).to(device)
    if refresh or not os.path.exists(corpus_embeddings_path):
        print("Encoding corpus...")
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True).to(device)
        pickle.dump(corpus_embeddings, open(corpus_embeddings_path, 'wb'))
    else:
        corpus_embeddings = pickle.load(open(corpus_embeddings_path, 'rb'))
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    for hit in hits:
        hit = sorted(hit, key=lambda x: x['score'], reverse=True)
        tok_k_indices.append([corpus['corpus_id'] for corpus in hit])

    return tok_k_indices

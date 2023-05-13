import json
import os
import pickle

import numpy as np
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


def get_final_k(model, query, corpus, top_k_indices, final_k=5, refresh=False):
    if not refresh and os.path.exists('data/final_k_indices.json'):
        final_k_indices = json.load(open('data/final_k_indices.json', 'r'))
        return final_k_indices

    cross_inp = [[query[i], corpus[top_k_indices[i][j]]]
                 for i in range(len(query))
                 for j in range(len(top_k_indices[i]))]
    cross_scores = model.predict(cross_inp)

    top_k = len(top_k_indices[0])
    final_k_indices = []
    for i in range(len(query)):
        final_k_indices.append(top_k_indices[i][cross_scores[i * top_k:(i + 1) * top_k].argsort()[-final_k:][::-1]])

    json.dump(final_k_indices, open('data/final_k_indices.json', 'w'))
    return final_k_indices


def get_classification(model, texts):

    classification_scores = model.predict(texts)
    classification = np.argmax(classification_scores, dim=1)
    return classification

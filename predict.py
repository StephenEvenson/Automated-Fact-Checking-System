import json
import os
import pickle

import numpy as np
import torch.cuda
from sentence_transformers import util, SentenceTransformer, CrossEncoder

from preprocess import get_test_data, get_raw_test_data, get_evidence_data

label_mapping = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT_ENOUGH_INFO': 2,
    'DISPUTED': 3
}

idx_to_label = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO', 'DISPUTED']


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
    if not refresh and os.path.exists('data/final_k_indices.npy'):
        final_k_indices = np.load('data/final_k_indices.npy')
        return final_k_indices

    cross_inp = [[query[i], corpus[top_k_indices[i][j]]]
                 for i in range(len(query))
                 for j in range(len(top_k_indices[i]))]
    cross_scores = model.predict(cross_inp)

    top_k = len(top_k_indices[0])
    final_k_indices = []
    top_k_indices = np.array(top_k_indices)
    for i in range(len(query)):
        query_evidence_scores = cross_scores[i * top_k:(i + 1) * top_k]
        final_k_scores = query_evidence_scores.argsort()[-final_k:][::-1]
        final_k_indices.append(top_k_indices[i][final_k_scores])

    np.save('data/final_k_indices.npy', final_k_indices)
    return final_k_indices


def get_classification(model, texts):
    classification_scores = model.predict(texts)
    classification = np.argmax(classification_scores, axis=1)
    return classification


def get_test_claim_result():
    top_k = 200
    final_k = 5
    print("Loading models...")
    retrieve_model_path = 'output/retrieve_model'
    rerank_model_path = 'output/rerank_model'
    classifier_model_path = 'output/classifier_model'

    test_data = get_test_data()
    raw_test_data = get_raw_test_data()
    test_claims = [data['claim_text'] for data in test_data.values()]
    evidence_data = get_evidence_data()
    test_evidences = list(evidence_data.values())

    top_k_indices = get_top_k(SentenceTransformer(retrieve_model_path), test_claims,
                              test_evidences, top_k=top_k, refresh=False)
    final_k_indices = get_final_k(CrossEncoder(rerank_model_path, num_labels=1, max_length=256), test_claims,
                                  test_evidences, top_k_indices, final_k=final_k, refresh=False)

    texts = []
    for index, (claim_id, data) in enumerate(test_data.items()):
        claim_text = data['claim_text']
        for evidence_index in final_k_indices[index]:
            sentence_pair = [claim_text, evidence_data['evidence-' + str(evidence_index)]]
            texts.append(sentence_pair)

    classifier_model = CrossEncoder(classifier_model_path, num_labels=4, max_length=256)
    classification_score = classifier_model.predict(texts)
    classification_score = sigmoid(classification_score)
    merged_arr = np.mean(classification_score.reshape(-1, final_k, 4), axis=1)
    classification = np.argmax(merged_arr, axis=1)

    test_preds = {}
    for index, (claim_id, pred) in enumerate(zip(test_data.keys(), classification)):
        test_preds[claim_id] = {
            "claim_text": raw_test_data[claim_id]["claim_text"],
            "claim_label": idx_to_label[pred],
            "evidences": [f"evidence-{x}" for x in final_k_indices[index]]
        }

    print("Writing predictions to file...")
    with open("output/test-claims-predictions.json", "w") as outfile:
        json.dump(test_preds, outfile)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

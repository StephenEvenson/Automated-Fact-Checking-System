import json
import os
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm


# Define a function for text preprocessing
def preprocess_text(text):

    # Lowercase the text
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-z0-9]+', ' ', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)


def preprocess():
    # Download the NLTK stop words
    nltk.download('stopwords')
    nltk.download('punkt')
    # Load the raw data
    with open('data/train-claims.json', 'r') as f:
        train_data = json.load(f)

    with open('data/dev-claims.json', 'r') as f:
        dev_data = json.load(f)

    with open('data/evidence.json', 'r') as f:
        evidence_data = json.load(f)

    with open('data/test-claims-unlabelled.json', 'r') as f:
        test_data = json.load(f)

    # Preprocess the claims and evidence in train_data and dev_data
    print('Preprocessing the claims and evidence in train_data and dev_data')
    for dataset in [train_data, dev_data]:
        for claim_id, data in dataset.items():
            data['claim_text'] = preprocess_text(data['claim_text'])

    print('Preprocessing the evidence in test_data')
    for claim_id, data in test_data.items():
        data['claim_text'] = preprocess_text(data['claim_text'])

    for evidence_id, evidence in tqdm(evidence_data.items(), desc='Preprocessing the evidence in evidence_data'):
        evidence_data[evidence_id] = preprocess_text(evidence)

    with open('data/preprocessed_train_data.json', 'w') as f:
        json.dump(train_data, f)

    with open('data/preprocessed_dev_data.json', 'w') as f:
        json.dump(dev_data, f)

    # Save preprocessed evidence_data to a JSON file
    with open('data/preprocessed_evidence.json', 'w') as f:
        json.dump(evidence_data, f)

    with open('data/preprocessed_test_data.json', 'w') as f:
        json.dump(test_data, f)


def get_evidence_data():
    evidence_path = 'data/preprocessed_evidence.json'
    if not os.path.exists(evidence_path):
        preprocess()
    with open(evidence_path, 'r') as f:
        evidence_data = json.load(f)
    return evidence_data


def get_train_data():
    train_data_path = 'data/preprocessed_train_data.json'
    if not os.path.exists(train_data_path):
        preprocess()
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    return train_data


def get_dev_data():
    dev_data_path = 'data/preprocessed_dev_data.json'
    if not os.path.exists(dev_data_path):
        preprocess()
    with open(dev_data_path, 'r') as f:
        dev_data = json.load(f)
    return dev_data


def get_test_data():
    test_data_path = 'data/preprocessed_test_data.json'
    if not os.path.exists(test_data_path):
        preprocess()
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    return test_data


def get_raw_test_data():
    test_data_path = 'data/test-claims-unlabelled.json'
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    return test_data

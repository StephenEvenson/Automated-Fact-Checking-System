import numpy as np
from sentence_transformers import InputExample
from torch.utils.data import Dataset, DataLoader

from predict import get_top_k
from preprocess import get_train_data, get_evidence_data


class RetrieveTrainDataset(Dataset):
    def __init__(self, model):
        self.train_data = get_train_data()
        self.evidence_data = get_evidence_data()
        query = [data['claim_text'] for data in self.train_data.values()]
        corpus = list(self.evidence_data.values())
        print("Retrieving top k evidences for training data...")
        self.top_k_indices = get_top_k(model, query, corpus, top_k=10, refresh=True)

        train_examples = []
        for index, (claim_id, data) in enumerate(self.train_data.items()):
            claim_text = data['claim_text']
            ng_evidences = list(
                set(["evidence-" + str(evidence_index) for evidence_index in self.top_k_indices[index]])
                - set(data['evidences']))
            for i in range(len(ng_evidences)):
                evidence_text = self.evidence_data[data['evidences'][i % len(data['evidences'])]]
                ng_evidence_text = self.evidence_data[ng_evidences[i]]
                train_examples.append(InputExample(texts=[claim_text, evidence_text, ng_evidence_text]))
        self.train_examples = np.array(train_examples)

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        return self.train_examples[idx]


def get_retrieve_train_dataloader(model, shuffle=True, batch_size=125):
    dataset = RetrieveTrainDataset(model=model)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


class RerankTrainDataset(Dataset):
    def __init__(self, retrieve_model):
        self.train_data = get_train_data()
        self.evidence_data = get_evidence_data()
        query = [data['claim_text'] for data in self.train_data.values()]
        corpus = list(self.evidence_data.values())
        print("Retrieving top k evidences for training data...")
        self.top_k_indices = get_top_k(retrieve_model, query, corpus, top_k=100, refresh=False)

        train_examples = []
        for index, (claim_id, data) in enumerate(self.train_data.items()):
            claim_text = data['claim_text']
            ng_evidences = ["evidence-" + str(evidence_index) for evidence_index in self.top_k_indices[index]
                            if "evidence-" + str(evidence_index) not in data['evidences']]
            for i in range(len(data['evidences'])):
                evidence_text = self.evidence_data[data['evidences'][i]]
                train_examples.append(InputExample(texts=[claim_text, evidence_text], label=1))
            for i in range(len(ng_evidences)):
                ng_evidence_text = ng_evidences[i]
                train_examples.append(InputExample(texts=[claim_text, ng_evidence_text], label=0))
        self.train_examples = np.array(train_examples)

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        return self.train_examples[idx]


def get_rerank_train_dataloader(retrieve_model, shuffle=True, batch_size=125):
    dataset = RerankTrainDataset(retrieve_model=retrieve_model)
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


class ClassifierTrainDataset(Dataset):
    def __init__(self):
        self.train_data = get_train_data()
        self.evidence_data = get_evidence_data()

        label_mapping = {
            'SUPPORTS': 0,
            'REFUTES': 1,
            'NOT_ENOUGH_INFO': 2,
            'DISPUTED': 3
        }

        idx_to_label = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO', 'DISPUTED']

        train_examples = []
        for claim_id, data in self.train_data.items():
            claim_text = data['claim_text']
            for evidence_index in data['evidences']:
                evidence_text = self.evidence_data[evidence_index]
                train_examples.append(InputExample(texts=[claim_text, evidence_text],
                                                   label=label_mapping[data['claim_label']]))
        self.train_examples = np.array(train_examples)

    def __len__(self):
        return len(self.train_examples)

    def __getitem__(self, idx):
        return self.train_examples[idx]

def get_classifier_train_dataloader(shuffle=True, batch_size=125):
    dataset = ClassifierTrainDataset()
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)

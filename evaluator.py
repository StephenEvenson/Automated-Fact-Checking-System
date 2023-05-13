import torch
from sentence_transformers.evaluation import SentenceEvaluator

from predict import get_top_k, get_final_k, get_classification
from preprocess import get_dev_data, get_evidence_data

label_mapping = {
    'SUPPORTS': 0,
    'REFUTES': 1,
    'NOT_ENOUGH_INFO': 2,
    'DISPUTED': 3
}


class RetrieveNgEvaluator(SentenceEvaluator):
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.dev_data = get_dev_data()
        self.dev_claims = [data['claim_text'] for data in self.dev_data.values()]
        self.evidence_data = get_evidence_data()
        self.dev_evidences = list(self.evidence_data.values())

    # calculate the recall, accuracy, and f1 score in the dev set
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        top_k_indices = get_top_k(model, self.dev_claims, self.dev_evidences,
                                  top_k=self.top_k, refresh=True)

        acc, recall, f1 = [], [], []
        for index, (claim_id, data) in enumerate(self.dev_data.items()):
            true_evidences = [int(evidence[len("evidence-"):]) for evidence in data['evidences']]
            correct = len(set(true_evidences).intersection(set(top_k_indices[index])))

            acc.append(correct / self.top_k)
            recall.append(correct / len(true_evidences))
            f1.append(2 * acc[-1] * recall[-1] / (acc[-1] + recall[-1]) if acc[-1] + recall[-1] > 0 else 0)

        print("Retrieve evaluator accuracy: ", sum(acc) / len(acc))
        print("Retrieve evaluator recall: ", sum(recall) / len(recall))
        print("Retrieve evaluator f1: ", sum(f1) / len(f1))
        return sum(f1) / len(f1)


class RerankEvaluator(SentenceEvaluator):
    def __init__(self, retrieve_model, final_k=5):
        self.dev_data = get_dev_data()
        self.dev_claims = [data['claim_text'] for data in self.dev_data.values()]
        self.evidence_data = get_evidence_data()
        self.dev_evidences = list(self.evidence_data.values())
        self.final_k = final_k
        self.retrieve_model = retrieve_model
        self.top_k_indices = get_top_k(self.retrieve_model, self.dev_claims, self.dev_evidences,
                                       top_k=10, refresh=False)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        final_k_indices = get_final_k(model, self.dev_claims, self.dev_evidences, self.top_k_indices, final_k=5)
        acc, recall, f1 = [], [], []
        for index, (claim_id, data) in enumerate(self.dev_data.items()):
            true_evidences = [int(evidence[len("evidence-"):]) for evidence in data['evidences']]
            correct = len(set(true_evidences).intersection(set(final_k_indices[index])))

            acc.append(correct / self.final_k)
            recall.append(correct / len(true_evidences))
            f1.append(2 * acc[-1] * recall[-1] / (acc[-1] + recall[-1]) if acc[-1] + recall[-1] > 0 else 0)

        print("Retrieve evaluator accuracy: ", sum(acc) / len(acc))
        print("Retrieve evaluator recall: ", sum(recall) / len(recall))
        print("Retrieve evaluator f1: ", sum(f1) / len(f1))
        return sum(f1) / len(f1)


class ClassifierEvaluator(SentenceEvaluator):
    def __init__(self):
        self.dev_data = get_dev_data()
        self.dev_claims = [data['claim_text'] for data in self.dev_data.values()]
        self.evidence_data = get_evidence_data()
        self.dev_evidences = list(self.evidence_data.values())

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        texts = []
        true_labels = []
        for index, (claim_id, data) in enumerate(self.dev_data.items()):
            claim_text = data['claim_text']
            true_labels.append(label_mapping[data['claim_label']])
            for evidence_index in data['evidences']:
                sentence_pair = [claim_text, self.evidence_data[evidence_index]]
                texts.append(sentence_pair)
        classification = get_classification(model, texts)
        true_labels = torch.tensor(true_labels)

        score = torch.sum(classification == true_labels).item() / len(true_labels)
        print("Classifier evaluator accuracy: ", score)

        return score

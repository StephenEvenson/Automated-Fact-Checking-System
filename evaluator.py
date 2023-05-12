from sentence_transformers.evaluation import SentenceEvaluator

from predict import get_top_k
from preprocess import get_dev_data, get_evidence_data


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
                                  top_k=self.top_k, pre_train=False, refresh=True)

        recall = []
        for index, (claim_id, data) in enumerate(self.dev_data.items()):
            true_evidences = [int(evidence[len("evidence-"):]) for evidence in data['evidences']]
            correct = len(set(true_evidences).intersection(set(top_k_indices[index])))

            # acc.append(correct / self.top_k)
            recall.append(correct / len(true_evidences))
            # f1.append(2 * acc[-1] * recall[-1] / (acc[-1] + recall[-1]))

        print("Retrieve evaluator recall: ", sum(recall) / len(recall))
        return sum(recall) / len(recall)




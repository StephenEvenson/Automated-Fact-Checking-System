from predict import get_test_claim_result
from preprocess import get_train_data
from train import retrieve_train, rerank_train, classifier_train


def main():
    # retrieve_train(epochs=100)
    # rerank_train(epochs=20, load_old_model=False)
    # classifier_train(epochs=20, load_old_model=False)
    get_test_claim_result()

if __name__ == '__main__':
    main()

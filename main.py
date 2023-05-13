from predict import get_test_claim_result
from preprocess import get_train_data
from train import retrieve_train, rerank_train, classifier_train


def main():
    # retrieve_train(epochs=20)
    # rerank_train(epochs=20, load_old_model=True)
    # classifier_train(epochs=30, load_old_model=True)
    get_test_claim_result()

if __name__ == '__main__':
    main()

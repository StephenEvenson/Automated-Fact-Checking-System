from preprocess import get_train_data
from train import retrieve_train, rerank_train, classifier_train


def main():
    retrieve_train(epochs=20)
    rerank_train(epochs=60)
    classifier_train(epochs=30)


if __name__ == '__main__':
    main()

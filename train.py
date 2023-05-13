import os.path

from sentence_transformers import SentenceTransformer, losses, CrossEncoder
from tqdm import tqdm

from dataset import get_retrieve_train_dataloader, get_rerank_train_dataloader, get_classifier_train_dataloader
from evaluator import RetrieveNgEvaluator, RerankEvaluator, ClassifierEvaluator


def retrieve_train(epochs=100):
    model_path = 'output/retrieve_model'
    if os.path.exists(model_path):
        print("Loading pretrained model from {}".format(model_path))
        bi_encoder = SentenceTransformer(model_path)
    else:
        bi_encoder = SentenceTransformer('distilbert-base-uncased')
    dataloader = get_retrieve_train_dataloader(bi_encoder, shuffle=True, batch_size=125)
    loss_function = losses.MultipleNegativesRankingLoss(model=bi_encoder)
    evaluator = RetrieveNgEvaluator(top_k=100)
    print("Start retrieve training...")
    evaluation_epochs = 20
    best_f1 = evaluator(bi_encoder)
    for epoch in tqdm(range(0, epochs, evaluation_epochs), desc="Total training epoch"):
        bi_encoder.fit(
            train_objectives=[(dataloader, loss_function)],
            epochs=evaluation_epochs,
            warmup_steps=100,
            use_amp=True,
            show_progress_bar=True,
        )
        f1 = evaluator(bi_encoder)
        if f1 > best_f1:
            best_f1 = f1
            bi_encoder.save(model_path)


def rerank_train(epochs=60):
    model_path = 'output/rerank_model'
    retrieve_model_path = 'output/retrieve_model'
    bi_encoder = SentenceTransformer(retrieve_model_path)
    if os.path.exists(model_path):
        print("Loading pretrained model from {}".format(model_path))
        cross_encoder = CrossEncoder(model_path, num_labels=1)
    else:
        cross_encoder = CrossEncoder('distilbert-base-uncased', num_labels=1)
    dataloader = get_rerank_train_dataloader(bi_encoder, shuffle=True, batch_size=125)
    evaluator = RerankEvaluator(retrieve_model=bi_encoder, final_k=5)
    print("Start rerank training...")
    cross_encoder.fit(
        train_dataloader=dataloader,
        evaluator=evaluator,
        evaluation_steps=500,
        epochs=epochs,
        warmup_steps=100,
        use_amp=True,
        show_progress_bar=True,
        output_path=model_path,
        save_best_model=True
    )


def classifier_train(epochs=10):
    model_path = 'output/classifier_model'
    model_name = 'roberta-large'
    if os.path.exists(model_path):
        print("Loading pretrained model from {}".format(model_path))
        classifier_model = CrossEncoder(model_path, num_labels=4, max_length=256)
    else:
        classifier_model = CrossEncoder(model_name, num_labels=4, max_length=256)
    dataloader = get_classifier_train_dataloader(shuffle=True, batch_size=64)
    evaluator = ClassifierEvaluator()
    print("Start classifier training...")
    evaluator(classifier_model)
    classifier_model.fit(
        train_dataloader=dataloader,
        evaluator=evaluator,
        evaluation_steps=100,
        epochs=epochs,
        warmup_steps=100,
        use_amp=True,
        show_progress_bar=True,
        output_path=model_path,
        save_best_model=True
    )




import os.path

from sentence_transformers import SentenceTransformer, losses

from dataset import get_train_dataloader
from evaluator import RetrieveNgEvaluator


def retrieve_train(epochs=500):
    model_path = 'output/retrieve_model'
    if os.path.exists(model_path):
        bi_encoder = SentenceTransformer(model_path)
    else:
        bi_encoder = SentenceTransformer('distilbert-base-uncased')
    dataloader = get_train_dataloader(bi_encoder, shuffle=True, batch_size=125)
    loss_function = losses.MultipleNegativesRankingLoss(model=bi_encoder)
    evaluator = RetrieveNgEvaluator(top_k=100)
    print("Start training...")
    evaluation_epochs = 20
    best_f1 = 0
    for epoch in range(0, epochs, evaluation_epochs):
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

#
# def rerank_train():



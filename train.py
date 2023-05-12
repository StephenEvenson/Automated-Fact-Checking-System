from sentence_transformers import SentenceTransformer, losses

from dataset import get_train_dataloader
from evaluator import RetrieveNgEvaluator


def retrieve_train(epochs=1000):
    bi_encoder = SentenceTransformer('distilbert-base-uncased')
    dataloader = get_train_dataloader(bi_encoder, shuffle=True, batch_size=125)
    loss_function = losses.MultipleNegativesRankingLoss(model=bi_encoder)
    evaluator = RetrieveNgEvaluator(top_k=100)
    print("Start training...")
    bi_encoder.fit(
        train_objectives=[(dataloader, loss_function)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=100,
        evaluation_steps=500,
        output_path='output/retrieve_train',
        save_best_model=True,
        use_amp=True,
        show_progress_bar=True,
    )

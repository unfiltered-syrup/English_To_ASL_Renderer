import pandas as pd
from datasets import Dataset
import evaluate
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_pred(batch):
    input_ids = tokenizer(
        batch["text"],
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).input_ids
    out = model.generate(input_ids)
    batch["pred"] = tokenizer.decode(out[0], skip_special_tokens=True)
    return batch

if __name__ == "__main__":
    # 1) Load model & tokenizer
    model     = T5ForConditionalGeneration.from_pretrained("./final_model")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # 2) Load test set
    df = pd.read_csv("test_processed.csv")
    ds = Dataset.from_pandas(df)

    # 3) Predictions
    preds = ds.map(generate_pred, batched=False)

    # 4) Exact Match
    refs = preds["gloss"]
    hyps = preds["pred"]
    exact = sum(p == r for p, r in zip(hyps, refs)) / len(refs)
    print(f"Exact Match: {exact:.4f}")

    # 5) BLEU Score
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(
        predictions=hyps,
        references=[r.split() for r in refs]
    )
    print(f"BLEU Score: {bleu_score['bleu']:.4f}")
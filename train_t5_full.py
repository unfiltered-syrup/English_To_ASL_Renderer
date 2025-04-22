"""
Full pipeline: Data prep → Tokenization → T5‑base fine‑tuning → BLEU evaluation
"""

import re
import pandas as pd
import sacrebleu
import nltk

from datasets import Dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# 1. Basic cleaning
def basic_cleaning(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9:\.\-\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# 2. Load & preprocess
df = pd.read_csv('data/train.csv') \
       .rename(columns={'text':'input_text', 'gloss':'target_text'})
df['input_text']  = df['input_text'].astype(str).apply(basic_cleaning)
df['target_text'] = df['target_text'].astype(str).str.strip().str.upper()

# 3. Train/test split
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = dataset['train'], dataset['test']

# 4. Model & tokenizer
model_name = "t5-base"
tokenizer  = T5TokenizerFast.from_pretrained(model_name)
model      = T5ForConditionalGeneration.from_pretrained(model_name)

# 5. Preprocess for T5
prefix = "translate English to ASL gloss: "
def preprocess(batch):
    inputs = [prefix + t for t in batch['input_text']]
    mi = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch['target_text'],
                           max_length=128, truncation=True, padding="max_length")
    mi["labels"] = labels["input_ids"]
    return mi

train_tok = train_ds.map(preprocess, batched=True,
                         remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(preprocess,  batched=True,
                       remove_columns=eval_ds.column_names)

# 6. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 7. BLEU metric
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    cleaned_labels = []
    for lab in labels:
        seq = [(tok if tok != -100 else tokenizer.pad_token_id) for tok in lab]
        cleaned_labels.append(tokenizer.decode(seq, skip_special_tokens=True))
    bleu = sacrebleu.corpus_bleu(decoded_preds, [cleaned_labels])
    return {"bleu": bleu.score / 100}

# 8. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir                  = "./t5_gloss",
    num_train_epochs            = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    weight_decay                = 0.01,
    save_total_limit            = 2,
    predict_with_generate       = True,
    logging_dir                 = "./logs",
)

# 9. Trainer
trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = eval_tok,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# 10. Run
if __name__ == "__main__":
    nltk.download("punkt")
    print("Starting T5‑base fine‑tuning…")
    trainer.train()
    print("Evaluating…")
    res = trainer.evaluate()
    print(f"→ BLEU score: {res['eval_bleu']:.4f}")
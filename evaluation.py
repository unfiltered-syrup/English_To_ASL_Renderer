import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from train import preprocess_and_gloss

def evaluate_exact(df, true_col='gloss', pred_col='asl_gloss'):
    """
    Exact match evaluation: accuracy, precision, recall, F1 on full-string labels.
    """
    true_labels = df[true_col].astype(str).str.strip().str.upper()
    pred_labels = df[pred_col].astype(str).str.strip().str.upper()

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    return accuracy, precision, recall, f1


def evaluate_jaccard(df, true_col='gloss', pred_col='asl_gloss'):
    """
    Token-level Jaccard similarity evaluation.
    """
    true_tokens = df[true_col].astype(str).str.upper().str.split()
    pred_tokens = df[pred_col].astype(str).str.upper().str.split()

    mlb = MultiLabelBinarizer().fit(true_tokens + pred_tokens)
    true_bin = mlb.transform(true_tokens)
    pred_bin = mlb.transform(pred_tokens)
    jaccard = jaccard_score(true_bin, pred_bin, average='samples')
    return jaccard

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df['asl_gloss'] = df['text'].apply(preprocess_and_gloss)

    # Exact-match metrics
    accuracy, precision, recall, f1 = evaluate_exact(df)
    print("Exact Match Evaluation:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}\n")

    # Jaccard similarity
    jaccard = evaluate_jaccard(df)
    print(f"Token-Level Jaccard Similarity: {jaccard:.4f}")

    # Show mismatches
    mismatches = df[df['gloss'].str.strip().str.upper() != df['asl_gloss'].str.strip().str.upper()]
    if not mismatches.empty:
        print("\nSample Mismatches:")
        print(mismatches[['gloss', 'asl_gloss']].head(10))
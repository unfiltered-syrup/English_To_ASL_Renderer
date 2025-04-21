import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
from train import preprocess_and_gloss

def evaluate_exact(df, true_col='gloss', pred_col='asl_gloss'):
    true_labels = df[true_col].astype(str).str.strip().str.upper()
    pred_labels = df[pred_col].astype(str).str.strip().str.upper()
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='weighted', zero_division=0
    )
    return accuracy, precision, recall, f1

def evaluate_jaccard(df, true_col='gloss', pred_col='asl_gloss'):
    true_tokens = df[true_col].astype(str).str.upper().str.split()
    pred_tokens = df[pred_col].astype(str).str.upper().str.split()
    mlb = MultiLabelBinarizer().fit(true_tokens + pred_tokens)
    true_bin = mlb.transform(true_tokens)
    pred_bin = mlb.transform(pred_tokens)
    return jaccard_score(true_bin, pred_bin, average='samples')

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )

    test_df = test_df.copy()
    test_df['asl_gloss'] = test_df['text'].apply(preprocess_and_gloss)

    # Exact-match metrics on test split
    accuracy, precision, recall, f1 = evaluate_exact(test_df)
    print("Exact Match Evaluation (20% test set):")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}\n")

    # Token-level Jaccard on test split
    jacc = evaluate_jaccard(test_df)
    print(f"Token-Level Jaccard Similarity: {jacc:.4f}\n")

    # Show some mismatches
    mismatches = test_df[
        test_df['gloss'].astype(str).str.strip().str.upper()
        != test_df['asl_gloss'].astype(str).str.strip().str.upper()
    ]
    if not mismatches.empty:
        print("Sample mismatches on test set:")
        print(mismatches[['gloss', 'asl_gloss']].head(10))
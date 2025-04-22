import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def basic_cleaning(text: str) -> str:
    """
    Lowercase, normalize times & preserve DESC- tags,
    hyphens, apostrophes, periods, and colons.
    """
    text = text.lower()
    # Normalize a.m./p.m. into DESC- tags
    text = re.sub(r"\bp\.m\.\b", "desc-p.m.", text)
    text = re.sub(r"\ba\.m\.\b", "desc-a.m.", text)
    # Keep letters, digits, periods, hyphens, apostrophes, and spaces
    text = re.sub(r"[^a-z0-9\.\-'\s:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize, convert possessives, POS-tag, and lemmatize."""
    tokens = word_tokenize(text)
    converted = []
    for t in tokens:
        if t.endswith("'s"):
            base = t[:-2]
            if base: converted.append(base)
            converted.append("x-poss")
        else:
            converted.append(t)
    # POS tag & lemmatize
    tagged = nltk.pos_tag(converted)
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(w, get_wordnet_pos(tag))
        for w, tag in tagged
    ]

def remove_stopwords(tokens: list) -> list:
    """Remove stopwords but keep essential function words."""
    exceptions = {'be', 'at', 'and', 'on', 'in', 'for', 'of', 'the'}
    sw = set(stopwords.words('english')) - exceptions
    return [t for t in tokens if t not in sw]

def preprocess_and_gloss(text: str) -> str:
    """
    Full pipeline:
      1. basic_cleaning
      2. tokenize_and_lemmatize
      3. remove_stopwords
      4. uppercase join
    """
    cleaned   = basic_cleaning(text)
    lems      = tokenize_and_lemmatize(cleaned)
    filtered  = remove_stopwords(lems)
    return " ".join(filtered).upper()

def main():
    df = pd.read_csv('data/train.csv')
    df['asl_gloss'] = df['text'].apply(preprocess_and_gloss)
    print(df[['gloss', 'text', 'asl_gloss']].head(10))

if __name__ == "__main__":
    main()
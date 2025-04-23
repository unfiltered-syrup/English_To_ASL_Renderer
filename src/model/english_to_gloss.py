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
    """
    Map POS tag to first character accepted by WordNetLemmatizer.
    """
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
    Lowercase, normalize times, preserve DESC- tags, hyphens, apostrophes, periods.
    """
    text = text.lower()
    text = re.sub(r"\bp\.m\.\b", "desc-p.m.", text)
    text = re.sub(r"\ba\.m\.\b", "desc-a.m.", text)
    text = re.sub(r"[^a-z0-9\.\-'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> list:
    """
    Tokenize, convert possessives, POS-tag, and lemmatize.
    """
    tokens = word_tokenize(text)
    converted = []
    for token in tokens:
        if token.endswith("'s"):
            base = token[:-2]
            if base:
                converted.append(base)
            converted.append("x-poss")
        else:
            converted.append(token)

    tagged = nltk.pos_tag(converted)
    lemmatizer = WordNetLemmatizer()
    lems = []
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        lems.append(lemmatizer.lemmatize(word, wn_tag))
    return lems

def remove_stopwords(tokens: list) -> list:
    """
    Remove common English stopwords but keep function words used in gloss.
    """
    exceptions = {'be', 'at', 'and', 'on', 'in', 'for', 'of', 'the'}
    stop_words = set(stopwords.words('english')) - exceptions
    return [t for t in tokens if t not in stop_words]

def preprocess_and_gloss(text: str) -> str:
    """
    Full pipeline to produce ASL-style gloss aligned with ground truth.
    """
    cleaned    = basic_cleaning(text)
    lemmatized = tokenize_and_lemmatize(cleaned)
    filtered   = remove_stopwords(lemmatized)
    return " ".join(filtered).upper()


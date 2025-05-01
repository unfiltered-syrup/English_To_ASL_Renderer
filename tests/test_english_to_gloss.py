from src.model.english_to_gloss import *

import nltk
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('stopwords')

def test_get_wordnet_pos():
    assert get_wordnet_pos('JJ') == wordnet.ADJ
    assert get_wordnet_pos('VB') == wordnet.VERB
    assert get_wordnet_pos('NN') == wordnet.NOUN # redundant, but for consistency
    assert get_wordnet_pos('RB') == wordnet.ADV
    assert get_wordnet_pos('XYZ') == wordnet.NOUN


def test_basic_cleaning():
    text = [
        "It's a-hyphenated word.",
        "Symbols like !@#$%^&*()_+ are removed.",
        "Multiple   spaces   are reduced.",
        "ALL CAPS TEXT."
    ]
    expected = [
        "it's a-hyphenated word.",
        "symbols like are removed.",
        "multiple spaces are reduced.",
        "all caps text."
    ]
    'this is a test with 9 00 a.m. and 2 30 p.m..'


    for i in range(len(text)):
        assert basic_cleaning(text[i]) == expected[i], f"Failed basic_cleaning : {text[i]}"


def test_tokenize_and_lemmatize():
    text = [
        "The cat's toys are interesting.",
        "He is running quickly.",
        "The child's book.",
        "I saw three mice.",
    ]
    expected = [
        ['The', 'cat', 'x-poss', 'toy', 'be', 'interest', '.'],
        ['He', 'be', 'run', 'quickly', '.'],
        ['The', 'child', 'x-poss', 'book', '.'],
        ['I', 'saw', 'three', 'mouse', '.']
    ]

    for i in range(len(text)):
        assert tokenize_and_lemmatize(text[i]) == expected[i], f"Failed tokenize_and_lemmatize: {text[i]}"


def test_remove_stopwords():
    tokens = [
        ['the', 'cat', 'is', 'on', 'the', 'mat'],
        ['to', 'be', 'or', 'not', 'to', 'be'],
        ['a', 'few', 'of', 'the', 'reasons']
    ]
    expected = [
        ['the', 'cat', 'on', 'the', 'mat'],
        ['be', 'be'],
        ['of', 'the', 'reasons']
    ]

    for i in range(len(tokens)):
        assert remove_stopwords(tokens[i]) == expected[i], f"Failed remove_stopwords: {tokens[i]}"


def test_preprocess_and_gloss():
    text = [
        "The quick brown fox jumps over the lazy dog at 9:15.",
        "The children's toys are on the table.",
        "It is raining cats and dogs."
    ]
    expected = [
        "THE QUICK BROWN FOX JUMP THE LAZY DOG AT 9 15 .",
        "THE CHILD X-POSS TOY BE ON THE TABLE .",
        "BE RAIN CAT AND DOG ."
    ]

    for i in range(len(text)):
        assert preprocess_and_gloss(text[i]) == expected[i], f"Failed preprocess_and_gloss: {text[i]}"

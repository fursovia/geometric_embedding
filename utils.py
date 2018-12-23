from nltk import wordpunct_tokenize
import pandas as pd


def preprocess_sentence(sentence: str):
    """
    :return: list of words
    """
    tokens = list(filter(str.isalpha, wordpunct_tokenize(sentence.lower())))
    return tokens


def read_sts(path):
    df = pd.read_csv(path, sep='\n', header=None, names=['row'])
    df = pd.DataFrame(df.row.str.split('\t', 6).tolist(), columns=['', 'genre', 'filename',
                                                                   'year', 'score', 'sentence1', 'sentence2'])
    df.drop(df.columns[[0, 1, 2, 3]], inplace=True, axis=1)
    df["score"] = pd.to_numeric(df["score"])
    return df

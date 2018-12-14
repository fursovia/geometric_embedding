# Quote from the paper:
# "We evaluate our model on the STS Benchmark (Cer et al., 2017), a sentence-level semantic similarity
# dataset from SemEval and SEM STS. The goal for a model is to predict a similarity score of two
# sentences given a sentence pair. The evaluation is by the Pearson’s coefficient r between humanlabeled
# similarity (0 - 5 points) and predictions."

from embeddings import preprocess_sentence
from typing import Callable, Union
import os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


class STSDataset:
    def __init__(self,
                 data_type: Union[str, None] = 'test',
                 data_path: Union[str, None] = None,
                 preprocessor: Union[Callable, None] = preprocess_sentence):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = {'train': os.path.join(os.getcwd(), './data/sts_benchmark/sts-train.csv'),
                              'test': os.path.join(os.getcwd(), './data/sts_benchmark/sts-test.csv'),
                              'dev': os.path.join(os.getcwd(), './data/sts_benchmark/sts-dev.csv')}[data_type]
        self.preprocessor = preprocessor
        self.data = []
        self.true_scores = []
        with open(self.data_path) as fin:
            for line in fin:
                items = line.split('\t')
                self.data.append([self.preprocessor(items[5]), self.preprocessor(items[6])])
                self.true_scores.append(float(items[4]))


def evaluate(embedder: Callable[[str], np.array],
             dataset_type: str = 'test',
             similarity_metric: str = 'dot'):
    """
    :param embedder: Callable[str]->np.array
    :param similarity_metric: ['dot', 'cosine']
    :param dataset_type: ['train', 'dev', 'test']
    """
    dataset = STSDataset(data_type=dataset_type)
    model_scores = []

    if similarity_metric == 'dot':
        metric = np.dot
    elif similarity_metric == 'cosine':
        metric = cosine_similarity
    else:
        raise Exception("Metric {} in not defined".format(similarity_metric))

    for sent1, sent2 in dataset.data:
        score = metric(embedder(sent1).reshape(1, -1), embedder(sent2).reshape(1, -1))
        model_scores.append(score[0][0])
    return pearsonr(model_scores, dataset.true_scores)

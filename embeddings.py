from typing import List, Union

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from tensorflow import Tensor

models = ['sentence-transformers/sentence-t5-base',
          'sentence-transformers/sentence-t5-xl',
          'sentence-transformers/sentence-t5-xxl',
          'bert-base-nli-mean-tokens'
          ]


def get_sentences_embeddings(sentences: List[str]) -> List[List[float]]:
    model_name = models[0]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings.tolist()


if __name__ == "__main__":
    l = get_sentences_embeddings(sentences=["I am", "I was"])
    print(l)

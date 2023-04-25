from typing import List
from numpy import dot
from numpy.linalg import norm
import math


def jaccard(sentence1: str, sentence2: str) -> float:
    words1: set = set(sentence1.lower().split())
    words2: set = set(sentence2.lower().split())
    intersection: set = words1.intersection(words2)
    union: set = words1.union(words2)
    # J = |S1 INTERSECT S2|/|S1 UNION S2|
    jaccard_similarity: float = len(intersection) / len(union)
    return jaccard_similarity


def cosine(embedding1: List[float], embedding2: List[float]) -> float:
    cosine_sim = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    return cosine_sim


def manhattan(embedding1: List[float], embedding2: List[float]) -> float:
    manhattan_dist: float = 0
    for i in range(len(embedding1)):
        manhattan_dist += abs(embedding1[i] - embedding2[i])
    return 1 / (1 + manhattan_dist)


def euclidean(embedding1: List[float], embedding2: List[float]) -> float:
    euclidean_dist: float = 0
    for i in range(len(embedding1)):
        euclidean_dist += pow(embedding1[i] - embedding2[i], 2)
    euclidean_dist = math.sqrt(euclidean_dist)
    return 1 / (1 + euclidean_dist)

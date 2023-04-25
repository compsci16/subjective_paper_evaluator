# This is a sample Python script.

# Press the green button in the gutter to run the script.
import csv
from typing import List, Callable, Union
import numpy
import operator
import preprocesor
import embeddings
import math

import similarity


class AnswerGrader:
    def __init__(self, model_answer_points: List[str],
                 similarity_algo: Union[Callable[[List[float], List[float]], float],
                                        Callable[[str, str], float]],
                 maximum_marks, threshold_similarity: float):
        self.model_answer_points: List[str] = model_answer_points
        self.model_answer_embeddings: List[List[float]] = embeddings.get_sentences_embeddings(model_answer_points)
        self.similarity_algo = similarity_algo
        self.threshold_similarity = threshold_similarity
        self.maximum_marks = maximum_marks

    def mark(self, student_answer_points: List[str]) -> int:
        student_answer_embeddings: List[List[float]] = embeddings.get_sentences_embeddings(student_answer_points)
        covered_points_count = 0
        model_ans_points_count = len(self.model_answer_points)
        used_points_indices = set()  # so same point is not checked again from model answer!
        for i in range(len(student_answer_embeddings)):
            student_point_embedding = student_answer_embeddings[i]
            similarities = self.get_similarity_to_all_model_points(student_point_embedding)
            max_index, max_sim = max(enumerate(similarities), key=operator.itemgetter(1))

            # print("Student Point: ", student_answer_points[max_index])
            # print("Matched Model Point ", self.model_answer_points[max_index])
            print("Similarity Computed = ", max_sim)

            if max_sim >= self.threshold_similarity and not (max_index in used_points_indices):
                covered_points_count += 1
                # print(f'Matched: {self.model_answer_points[max_index]}\n{student_answer_points[i]}')
                used_points_indices.add(max_index)

        marks: int = math.ceil(covered_points_count / model_ans_points_count * self.maximum_marks)
        return marks

    def mark_without_embeddings(self, student_answer_points: List[str]):
        covered_points_count = 0
        model_ans_points_count = len(self.model_answer_points)
        used_points_indices = set()  # so same point is not checked again from model answer!
        for i in range(len(student_answer_points)):
            similarities = self.get_similarity_to_all_model_points(student_answer_points[i])
            max_index, max_sim = max(enumerate(similarities), key=operator.itemgetter(1))

            # print("Student Point: ", student_answer_points[max_index])
            # print("Matched Model Point ", self.model_answer_points[max_index])
            print("Similarity Computed = ", max_sim)

            if max_sim >= self.threshold_similarity and not (max_index in used_points_indices):
                covered_points_count += 1
                # print(f'Matched: {self.model_answer_points[max_index]}\n{student_answer_points[i]}')
                used_points_indices.add(max_index)

        marks: int = math.ceil(covered_points_count / model_ans_points_count * self.maximum_marks)
        return marks

    def get_similarity_to_all_model_points(self, student_point_embedding: List[float]) -> List[float]:
        similarities: List[float] = []
        for e in self.model_answer_embeddings:
            similarities.append(self.similarity_algo(student_point_embedding, e))
        return similarities

    def get_similarity_to_all_model_points_2(self, student_point: str) -> List[float]:
        similarities: List[float] = []
        for p in self.model_answer_points:
            similarities.append(self.similarity_algo(student_point, p))
        return similarities


def evaluate(marks_human, marks_ai):
    return numpy.corrcoef(marks_human, marks_ai)[0, 1]


def get_human_ai_scores_for(grader, row) -> (float, float):
    student_answer_points = preprocesor.split_paragraph(row[1])
    ai_score: int = grader.mark(student_answer_points)
    print(f'Human Score = {row[2]}, AI Score = {ai_score}')
    return float(row[2]), ai_score


def view_scores(max_marks: int, sim_algo, threshold: float):
    human_scores = []
    ai_scores = []
    with open('./dataset2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # skip header row
        model_ans_points = preprocesor.split_paragraph(next(csv_reader)[1])
        grader = AnswerGrader(model_ans_points, sim_algo, max_marks, threshold)
        for row in csv_reader:
            h, a = get_human_ai_scores_for(grader, row)
            human_scores.append(h)
            ai_scores.append(a)
    return human_scores, ai_scores


if __name__ == '__main__':
    human, ai = view_scores(4, similarity.cosine, 0.90)
    print(evaluate(human, ai))
    # human, ai = view_scores(4, similarity.manhattan, 0.08)
    # print(evaluate(human, ai))
    # human, ai = view_scores(4, similarity.euclidean, 0.70)
    # print(evaluate(human, ai))
    # human, ai = view_scores(4, similarity.jaccard, 0.15)
    # print(evaluate(human, ai))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

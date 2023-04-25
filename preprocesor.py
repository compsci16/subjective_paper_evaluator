import nltk
import string


def split_paragraph(paragraph):
    sent_text = nltk.sent_tokenize(paragraph)
    return sent_text


def remove_punctuations(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))



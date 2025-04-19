from itertools import chain
from typing import List
import nltk
from nltk.corpus import wordnet

"""
The baseline model is a rule-based model 
"""
class BaselineModel():
    def __init__(self):
        nltk.download("wordnet")

    

    def _find_synonyms(self, word: str) -> List[str]:
        """
        Find synonyms of a word using WordNet
        """
        synonyms = wordnet.synsets(word)
        lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
        return list(lemmas)


if __name__ == "__main__":
    model = BaselineModel()


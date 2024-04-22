
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np 

class NLTKutils:

    @staticmethod
    def tokenise(word: str) -> list[str]:
        return nltk.word_tokenize(word)
    
    @staticmethod
    def stem(word: str) -> str:
        st = PorterStemmer()
        return st.stem(word.lower())
    
    @classmethod
    def bag_of_words(cls, tokenised_sentence: list[str], word_dict: list[str]) -> list[np.float32]:

        sentence_words = [cls.stem(word) for word in tokenised_sentence]
        word_dict = [NLTKutils.stem(word) for word in word_dict]
        bag = np.zeros(len(word_dict), dtype= np.float32)

        for idx, w in enumerate(word_dict):
            if w in sentence_words:
                bag[idx] = 1
        
        return bag
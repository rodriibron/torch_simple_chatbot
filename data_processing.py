
import numpy as np 
import torch 
import torch.nn as nn 
import nltk
from nltk.stem.porter import PorterStemmer
import json
from nltk.corpus import stopwords
from nltk_utils import NLTKutils
from torch.utils.data import Dataset



class DataProcessing(NLTKutils):

    def __init__(self, path: str):

        super(DataProcessing, self).__init__()

        with open(path, "r") as file:
            intents = json.load(file)
            self.data = intents
        
        self.stop_words = set(stopwords.words('english'))
    
    
    def loadData(self) -> tuple[list]:

        all_words = []
        tags = []
        word_tags = []

        for intent in self.data["intents"]:
            tag = intent["tag"]
            tags.append(tag)

            for pattern in intent['patterns']:
                w = NLTKutils.tokenise(pattern)
                all_words.extend(w)
                word_tags.append((w, tag))

        
        all_words = [NLTKutils.stem(w) for w in all_words if w not in self.stop_words]

        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        print(len(word_tags), "patterns")
        print(len(tags), "tags:", tags)
        print(len(all_words), "unique stemmed words:", all_words)

        return all_words, tags, word_tags
    
    def trainingData(self, all_words: list[str], tags: list[str], word_tags: list) -> tuple[np.array]:

        X_train = []
        y_train = []
        for (pattern_sentence, tag) in word_tags:
            # X: bag of words for each pattern_sentence
            bag = self.bag_of_words(pattern_sentence, all_words)
            X_train.append(bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = tags.index(tag)
            y_train.append(label)
        
        return np.array(X_train), np.array(y_train)
    



class ChatDataset(Dataset):

    def __init__(self, X_train, y_train):

        self.X_train = X_train 
        self.y_train = y_train 
        self.n_samples = len(X_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]
    
    def __len__(self):
        return self.n_samples
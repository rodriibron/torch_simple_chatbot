import random
import json

import torch

import json

from nltk_utils import NLTKutils
from data_processing import DataProcessing
from chat_model import NeuralNet, ChatModel


with open("intents.json", "r") as file:
    intents = json.load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainChatbot(model_path: str) -> tuple[NeuralNet, ChatModel]:

    data_process = DataProcessing("intents.json")
    all_words, tags, word_tags = data_process.loadData()
    X_train, y_train = data_process.trainingData(all_words, tags, word_tags)

    input_size= len(X_train[0])
    hidden_size= 8
    num_classes= len(tags)

    PARAMS = {"num_epochs": 1000, "batch_size": 8, "learning_rate": 0.001, "all_words": all_words, "tags": tags}


    model = NeuralNet(input_size= input_size, hidden_size= hidden_size, num_classes= num_classes).to(device)

    chatbot_model = ChatModel(X_train, y_train, input_size= input_size, 
                            hidden_size= hidden_size, output_size=num_classes, **PARAMS)
    chatbot_model.trainModel(model= model, model_path=f"{model_path}")

    return model, chatbot_model

        

def chatbot(model: NeuralNet, model_path: str):
    FILE = f"{model_path}"
    data = torch.load(FILE)

    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model.load_state_dict(model_state)
    model.eval()

    bot_name = "Txin"
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        sentence = NLTKutils.tokenise(sentence)
        X = NLTKutils.bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")


def main(path):
    model, _ = trainChatbot(f"{path}")
    chatbot(model, f"{path}")



if __name__ == "__main__":
    main("data.pth")


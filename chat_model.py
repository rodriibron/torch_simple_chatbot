
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from data_processing import ChatDataset



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        #parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        #layers
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    

class ChatModel(NeuralNet):

    def __init__(self, X_train, y_train, input_size, hidden_size, output_size, **kwargs):
        
        super(ChatModel, self).__init__(input_size, hidden_size, output_size)
        self.X_train = X_train 
        self.y_train = y_train
        self.kwargs = kwargs
        dataset = ChatDataset(X_train, y_train)
        self.train_loader = DataLoader(dataset=dataset, batch_size=kwargs["batch_size"], 
                                       shuffle=True, num_workers=0)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def trainModel(self, model: NeuralNet, save_model=True, model_path= None):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.kwargs["learning_rate"])

        # Train the model
        for epoch in range(self.kwargs["num_epochs"]):
            for (words, labels) in self.train_loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)
                
                # Forward pass
                outputs = model(words)
                # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if (epoch+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{self.kwargs["num_epochs"]}], Loss: {loss.item():.4f}')
            
        
        print(f'final loss: {loss.item():.4f}')

        if save_model:

            data = {
            "model_state": model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.num_classes,
            "all_words": self.kwargs["all_words"],
            "tags": self.kwargs["tags"]
            }

            FILE = f"{model_path}"
            torch.save(data, FILE)

            print(f'training complete. file saved to {FILE}')
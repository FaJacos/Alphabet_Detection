import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#creates network
class network(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(network,self).__init__()
        #This is a 1 layer network
        self.fc1 = nn.Linear(inputSize, 100)
        self.fc2 = nn.Linear(100,outputSize)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
def run():
    #Sets device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Parameters
    inputSize = 784
    numClasses = 10
    learningRate = 0.001
    batchSize = 64 #Batch size fed through the neuro network
    numEpoch = 1 #Number of iterations through the neuro network

    #Loading training data
    trainDataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download = True)
    trainLoader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)  

    #Loading testing data
    testDataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download = True)
    testLoader = DataLoader(dataset = testDataset, batch_size = batchSize, shuffle = True)

    #Initializing network
    model = network(inputSize = inputSize, outputSize = numClasses).to(device)

    #Loss and optimizer
    criteron = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = learningRate)

    #Training network
    for epoch in range(numEpoch):
        for batchId, (data, target) in enumerate(trainLoader):
            #Puts data into GPU
            data = data.to(device = device)
            target = target.to(device = device)
            
            #Reshaping data
            data = data.reshape(data.shape[0],-1)
            
            #Forward
            scores = model(data)
            loss =  criteron(scores,target)
            
            #Backward
            optimizer.zero_grad()
            loss.backward()
            
            #Gradient descent / adam step
            optimizer.step()
            
    def accuracy(loader, model):
        numCorrect = 0
        numSamples = 0
        
        model.eval()
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device = device)
                y = y.to(device = device)
                
                x = x.reshape(x.shape[0],-1)
                scores = model(x)
                _, predictions = scores.max(1)
                numCorrect += (predictions == y).sum()
                numSamples += predictions.size(0)
                
            print(f'Got {numCorrect} / {numSamples} with a accuracy of {((float(numCorrect)/float(numSamples))*100):.2f}')
            
        model.train()

    accuracy(trainLoader, model)
    accuracy(testLoader, model)
            
    torch.save(model.state_dict(), 'modelTrained.pt')
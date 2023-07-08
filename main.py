import torch
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import model
from model import network
import torch.nn.functional as F

#Setup for neuronetwork evaluation
numbers = {'0':'1', '1':'2', '2':'3', '3':'4', '4':'5', '5':'6', '6':'7', '7':'8', '8':'9', '9':'10'} #dictionary for printing
model = network(784,10) #creates a blank network with the same size as the model
model.load_state_dict(torch.load('modelTrained.pt')) #loads the blank network with our trained weights
model.eval() #sets model to evaluation mode

#Evaluates image through the neuronetwork and returns what it means
def eval():
    #reads the image
    image = cv.imread('test7.PNG')
    #changes image to 28x28
    image = cv.resize(image, dsize = (28,28), interpolation = cv.INTER_CUBIC)
    #convert BGR image to Gray scaled image
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    #creates tensor from image
    image = torch.from_numpy(image)
    #change the dtype to float
    image = image.float()
    #view image as the same dimensions as the training data
    image = image.view(1,784)
    
    
    #SUSPECTED ERROR HERE:
    #feeds tensor to model without gradient to improve speed
    with torch.no_grad():
        output = model(image)
        
        #gives probablity and label of output
        probs, label = torch.topk(output, 1)
        #Takes the largest probabilty
        probs = F.softmax(probs, 1)
        #gives prediction
        prediction = output.max(1, keepdim=True)[1]
        
        
        return numbers[str(int(prediction))] + ': ' + '{:.2f}'.format(100*float(probs[0,0])) + '%'
    

print(eval())

import torch
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
import model
from model import network

numbers = {'0':'1', '1':'2', '2':'3', '3':'4', '4':'5', '5':'6', '6':'7', '7':'8', '8':'9', '9':'10'}

model = network(784,10)
model.load_state_dict(torch.load('modelTrained.pt'))
model.eval()

image = cv.imread('test3.PNG')
image = cv.resize(image, dsize = (28,28), interpolation = cv.INTER_CUBIC)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

image = torch.from_numpy(image)
image = image.view(1,784)
image = image.float()

out = model(image)
probs, label = torch.topk(out, 10)
probs = torch.nn.functional.softmax(probs, 1)

pred = out.max(1, keepdim=True)[1]

print(label)
print(pred)
print(probs)
print(numbers[str(int(pred))] + ': ' + '{:.2f}'.format(100*float(probs[0,0])) + '%')




import torch
import random
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class CNN4MNIST(torch.nn.Module):
    def __init__(self):
        super(CNN4MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1,128,2)
        self.conv2 = nn.Conv2d(128,256,3)
        self.conv3 = nn.Conv2d(256,256,2)
        self.conv4 = nn.Conv2d(256,512,3)
        self.conv5 = nn.Conv2d(512,512,2)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print(x.size())
        x = F.relu(self.conv1(x))
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.dropout(x)
        #print(x.size())
        x = F.max_pool2d( F.relu( self.conv4(x)),2 )
        #print(x.size())
        x = F.max_pool2d(F.relu( self.conv5(x) ),2)
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x = F.relu(self.fc1( x ))
        x = F.relu(self.fc2( F.dropout(x) ))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
print(">>>loading model...")
model = torch.load('cnn4mnist.pt')

PATH_TEST = "./data/discrate/"

# load test data
print('>>>loading test data...')
test_data = []
labels = []
fig = plt.figure()
for i,img in  enumerate(os.listdir(PATH_TEST)):
    if img[-3:] in ['jpg']:
        print(">>>reading image",img)
        npimg = plt.imread(PATH_TEST+img).reshape(784,4)
        npimg = np.asarray([  np.average(np.sum(x[:3])) for x in npimg ]).reshape(1,784)
        test_data.append(npimg)
        labels.append(int(img[:1]))
        fig.add_subplot(4,3,i+1)
        plt.imshow(npimg.reshape(28,28),cmap='gray')
plt.show()
fig = plt.figure()
test_data = np.asarray(test_data)
test_data /= 255
test_data -= np.mean(test_data)
test_data = np.asarray([ x.reshape(1,28,28) for x in test_data ])
for i,img in enumerate(test_data):
    fig.add_subplot(4,3,i+1)
    plt.imshow(img.reshape(28,28),cmap='gray')
plt.show()

# start predicting
print(">>>evaluating...")
results = []
batch_size = 5
rounds = int(len(test_data)/batch_size)
base = 0
tipstr = ">>>>eval  " + str(base+batch_size) + "/" + str(len(test_data))
for i in range(rounds):
    print(tipstr,end='\r')
    base = i*batch_size
    x = Variable(torch.from_numpy(test_data[base:base+batch_size].astype(np.float32)), requires_grad=False).cuda()
    dd = model(x).data.cpu().numpy().tolist()
    results.extend(dd)
    tipstr = ">>>>eval  " + str(base+batch_size) + "/" + str(len(test_data))

print(tipstr ,"\n>>>>eval done.")

# save result
results = [ x.index(max(x)) for x in results ]
print(len(results))
results = np.asarray(results)
print(">>>saving results...")
df = pd.DataFrame({'Label':results,'ImageID':labels})
df.to_csv('./data/discrate/results.csv')
print(">>>done.")
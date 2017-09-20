import torch
import random
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class CNN4MNIST(torch.nn.Module):
    def __init__(self):
        super(CNN4MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1,64,5)
        self.conv2 = nn.Conv2d(64,128,3)
        self.conv3 = nn.Conv2d(128,256,3)
        self.conv4 = nn.Conv2d(256,128,3)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #print(x.size())
        x = F.relu(self.conv2(x))
        #print(x.size())
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        #print(x.size())
        x = F.max_pool2d( F.relu( self.conv4(x)),2 )
        #print(x.size())
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

PATH_TEST = "./data/test.csv"

# load test data
print('>>>loading test data...')
test_data=pd.read_csv(PATH_TEST)
test_data /= 255
mean = np.mean(test_data)
test_data -= mean
test_data = np.asarray([ x.reshape(1,28,28) for x in test_data.as_matrix() ])

# start predicting
print(">>>evaluating...")
results = []
batch_size = 400
rounds = int(len(test_data)/batch_size)
base = 0
for i in range(rounds):
    print(">>>>eval",base,"/",len(test_data))
    base = i*batch_size
    x = Variable(torch.from_numpy(test_data[base:base+batch_size].astype(np.float32)), requires_grad=False).cuda()
    dd = model(x).data.cpu().numpy().tolist()
    results.extend(dd)

print(">>>>eval done.")

# save result
# output result
results = [ x.index(max(x)) for x in results ]
print(len(results))
results = np.asarray(results)
print(">>>saving results...")
df = pd.DataFrame({'Label':results})
df.index += 1
df.index.name='ImageId'
df.to_csv('results.csv')
print(">>>done.")
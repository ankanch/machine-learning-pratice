# -*- coding: utf-8 -*-
import torch
import random
import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

dtype = torch.cuda.FloatTensor

# loading training data
print(">>>loading data...")
labeled_images = pd.read_csv("./data/train.csv")
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
print(">>>preprocessing data...")
images = images.astype('float32')
images /= 255
mean = np.mean(images)
images -= mean
images = np.asarray([ x.reshape(1,28,28) for x in images.as_matrix() ])
pm = []
for x  in labels.as_matrix():
    rl = [0,0,0,0,0,0,0,0,0,0]
    rl[x[0]] = 1
    rl = [ np.float32(x) for x in rl ]
    pm.append(rl.copy())
labels = np.asarray(pm)


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

def shuffleCrossVaildationAndTrain(xset,yset,vaild_size=0.2):
    cvsize = int(len(xset)*vaild_size)
    x = 0
    g_list = []
    vaild_set_x = []
    vaild_set_y = []
    train_set_x = []
    train_set_y = []
    while x < cvsize:
        i = random.randint(0,len(xset)-1)
        if i not in g_list:
            g_list.append(i)
            vaild_set_x.append(xset[i])
            vaild_set_y.append(yset[i])
            x+=1
    for i,x in enumerate(yset):
        if i not in g_list:
            train_set_y.append(x)
            train_set_x.append(xset[i])
    return np.asarray(train_set_x),np.asarray(train_set_y),np.asarray(vaild_set_x),np.asarray(vaild_set_y)

cnn4mnist = CNN4MNIST().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn4mnist.parameters())
print(cnn4mnist)
batch_size = 300
cross_vaild = 0.2

epoches = 5
for i in range(epoches):
    ros = "Epoche" + str(i+1) + "/" + str(epoches) + '\t'
    mss = ""
    xtrain,ytrain,xvaild,yvaild = shuffleCrossVaildationAndTrain(images,labels,cross_vaild)
    round_need = int(len(ytrain)/batch_size)
    # train
    for xx in range(round_need):
        # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
        base = xx*batch_size
        x = Variable(torch.from_numpy(xtrain[base:base+batch_size])).cuda()
        y = Variable(torch.from_numpy(ytrain[base:base+batch_size]), requires_grad=False).cuda()

        y_pred = cnn4mnist(x)

        cnn4mnist.zero_grad()
        loss = criterion(y_pred,y)
        mss = ros + str(base) + "/" + str(round_need*batch_size) + "\tloss:" + str(loss.data[0])
        print(mss,end='\r')
        loss.backward()
        optimizer.step() 
    # cross vaildation
    px = Variable(torch.from_numpy(xvaild[:100])).cuda()
    py = Variable(torch.from_numpy(yvaild[:100]), requires_grad=False).cuda()
    yp =  cnn4mnist(px)
    loss = criterion(yp,py)
    print(mss,"\tcross vaildation loss:",loss.data[0])

torch.save(cnn4mnist, 'cnn4mnist.pt')

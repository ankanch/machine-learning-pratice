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
#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 
import numpy as np

# activations

def Sigmoid(x):
    ex = np.exp(x)
    return ex/(ex+1)

def ReLU(x):
    return 0 if x<0 else x

# deveriative of activations

def Sigmoid_(x):
    sx = Sigmoid(x)
    return sx*(1-sx)

def ReLU_(x):
    return 0 if x<0 else 1  

if __name__ == "__main__":
    print("Sigmoid(2)=",Sigmoid(2))
    print("Sigmoid(-2)=",Sigmoid(-2))
    print("ReLU(2)=",ReLU(2))
    print("ReLU(-2)=",ReLU(-2))

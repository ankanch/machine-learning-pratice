#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 
import numpy as np

def Sigmoid(x):
    ex = np.exp(x)
    return ex/(ex+1)

def ReLU(x):
    return 0 if x<0 else x



if __name__ == "__main__":
    print("Sigmoid(2)=",Sigmoid(2))
    print("Sigmoid(-2)=",Sigmoid(-2))
    print("ReLU(2)=",ReLU(2))
    print("ReLU(-2)=",ReLU(-2))

#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 
import numpy as np

def squard_error(inputv,targetv):
    minsv = np.subtract(inputv,targetv)
    return np.sum( np.asarray([ 0.5*np.square(x) for x in minsv ]) )



if __name__ == "__main__":
    print(squard_error([1,2,3,4],[2,3,4,5]))
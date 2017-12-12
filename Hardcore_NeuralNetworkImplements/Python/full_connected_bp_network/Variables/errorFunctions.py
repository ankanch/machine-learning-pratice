#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 
import numpy as np

def squard_error(inputv,targetv):
    """
    return error_list for each neurons and total error

    error_list,total_error
    """
    minsv = np.subtract(inputv,targetv)
    error_list = np.asarray([ 0.5*np.square(x) for x in minsv ])
    return error_list,np.sum( error_list )



if __name__ == "__main__":
    print(squard_error([1,2,3,4],[2,3,4,5]))
#encoding=utf-8
# hardcore backpropagation neural network implements
#               by kanch @ 2017-11-29
#                kanchisme@gmail.com
# 

class Variable:
    """
    """
    value = 0.0
    label = None
    
    def __init__(self,v,l=""):
        self.value = v
        self.label = l
    
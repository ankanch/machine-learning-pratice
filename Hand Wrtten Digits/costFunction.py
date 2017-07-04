
def j_theta(real_output,predict_output,thetas):
    """
    This function is used to compute the cost under given thetas.
    * `real_output` is output in the training set. 
    * `predic_output` is output by the network. 
    * `thetas` is a matrix of all the thetas used in the network.
    """
    m = len(predict_output)
    

if __name__ == "__main__":
    print("test for cost function:")
    print("J_theta:",j_theta([1,0],[1,1],[1,1]))
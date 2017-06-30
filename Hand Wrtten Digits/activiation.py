from math import exp


def sigmoid(x):
    return 1/(1+exp(-1*x))


if __name__ == "__main__":
    print("test for sigmoid function: sigmoid(5)=",sigmoid(5))
    print("test for sigmoid function: sigmoid(1)=",sigmoid(1))
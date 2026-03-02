import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N,D=X.shape
    #define our weights
    w=np.zeros(D)
    #define our biases
    b=0.0
    for _ in range(steps):
        p=_sigmoid(np.dot(X,w)+b)
        error=p-y
        delw=np.dot(X.T,error)/N
        delb=np.mean(error)
       #do an update
        w=w-lr*delw
        b=b-lr*delb
    return w,b

        
        

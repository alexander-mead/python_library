### Activation functions ###

def sigmoid(x):
    from numpy import exp
    return 1./(1.+exp(-x))

def ReLU(x): # Rectified Linear Unit
    from numpy import where
    return where(x<0., 0., x)

def LU(x): # Linear Unit
    return x

def ELU(x): # Exponential Linear Unit
    from numpy import where, exp
    return where(x<0., exp(x)-1., x)

### ###
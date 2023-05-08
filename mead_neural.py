# Third-party imports
import numpy as np

### Activation functions ###

# TODO: Make classes with forward- and backward-propagation methods
# TODO: Automatic weight updates and network training functions

def LU(x): # Linear Unit
    return x

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def zeroed_sigmoid(x):
    return 2.*sigmoid(x)-1.

def tanh(x):
    return np.tanh(x)

def ReLU(x): # Rectified Linear Unit
    #return np.where(x<0., 0., x)
    return np.maximum(0., x)

def leakyReLU(x, alpha=0.1): # Leaky Rectified Linear Unit
    return np.where(x<0., alpha*x, x)

def ELU(x, alpha=1.): # Exponential Linear Unit
    return np.where(x<0., alpha*(np.exp(x)-1.), x)

### ###

### Derivatives of activation functions ###

def d_LU(x):
    return 1.

def d_sigmoid(x):
    return sigmoid(x)*(1.-sigmoid(x))

def d_zeroed_sigmoid(x):
    return 0.5*(1.-zeroed_sigmoid(x)**2)

def d_tanh(x):
    return 1.-tanh(x)**2

def d_ReLU(x):
    return np.where(x<0., 0., 1.)

def d_leakyReLU(x, alpha=0.1):
    return np.where(x<0., alpha, 1.)

def d_ELU(x, alpha=1.):
    return np.where(x<0., alpha*np.exp(x), 1.)

### ###

### Propagation ###

def forward_propagation_three_layer(x, w1, w2, w3, act=LU):
    '''
    Forward propagation through a three-layer neural network
    This is a two-hidden-layer network
    Takes an input 'x' and layer weights w1, w2, w3
    Activation function should be non-linear to make network non-trivial
    Output is y
    '''
    x0 = x               # Input layer
    x1 = act(x0.dot(w1)) # First hidden layer
    x2 = act(x1.dot(w2)) # Second hidden layer
    x3 = x2.dot(w3)      # Output layer (no non-linearity)
    y = x3               # Output prediction
    return y, [x0, x1, x2, x3]

def backward_propagation_three_layer(dL, x0, x1, x2, w1, w2, w3, dact=d_LU):
    '''
    Backward propagation through a three-layer neural network
    This is a two-hidden-layer network
    Used to determine weight updates
    Takes in dL = dL/dy as well as neuron values (x0, x1, x2) and weights (w1, w2, w3)
    Return is dL/dw for each weight matrix
    NOTE: w1 is not used in computations, which is a bit odd
    '''
    grad_x3 = dL                         # dL/dx3 (all dL/dx are vectors)
    grad_w3 = x2.T.dot(grad_x3)          # dL/dw3 (all dL/dw are matrices)
    grad_x2 = grad_x3.dot(w3.T)          # dL/dx2
    grad_w2 = x1.T.dot(grad_x2*dact(x2)) # dL/dw2; note element-wise product with the activation derivative
    grad_x1 = grad_x2.dot(w2.T)          # dL/dx1
    grad_w1 = x0.T.dot(grad_x1*dact(x1)) # dL/dw1; note element-wise product with the activation derivative
    return [grad_w1, grad_w2, grad_w3]

def forward_propagation(x, ws, act=LU):
    '''
    Forward propagation through a fully connected neural network
    Takes an input 'x' and a list of layer weight matrices 'ws' (n)
    This is an n-1 hidden-layer network
    Activation function should be non-linear to make network non-trivial
    Output is the value from the final layer and all the neuron values (n+1)
    '''
    x_prev = x; xs = [x]
    for i, w in enumerate(ws):
        if i != len(ws)-1:
            x_new = act(x_prev.dot(w))
        else:
            x_new = x_prev.dot(w) # No non-linearity with final layer
        xs.append(x_new)
        x_prev = x_new
    y = x_new
    return y, xs

def backward_propagation(dL, xs, ws, dact=d_LU):
    '''
    Backward propagation through a three-layer neural network to determine weight updates
    Input dL = dL/dy as well as neuron values (xs; n+1) and weights (ws; n)
    This is an n-1 hidden-layer network
    Return is dL/dw for each weight matrix
    NOTE: First layer weights are not used in computations, which is a bit odd
    '''
    n = len(ws); grad_ws = []
    for i in range(n):
        if i == 0:
            grad_x = dL                     # First neuron gradient is dL/dx[-1]
            grad_w = xs[-2-i].T.dot(grad_x) # First weight gradient has no non-linearity
        else:
            grad_x = grad_x.dot(ws[0-i].T)                 # dL/dx are all vectors
            grad_w = xs[-2-i].T.dot(grad_x*dact(xs[-1-i])) # dL/dw are matrices 
        grad_ws.append(grad_w)
    grad_ws.reverse() # Reverse list to get first-layer weight gradients first
    return grad_ws

def forward_timestep(i, x0, x1, x2, W0, W1, act):
    from numpy import matmul # TODO: Replace with dot
    x1[i] = act(matmul(W0[i], x0[i]))
    x2[i] = matmul(W1[i], x1[i]) # No non-linear function to output layer

def forward_timestep_lag(i, x0, x1, x2, W0, W1, l01, l12, act):
    from numpy import matmul # TODDO: Replace with dot
    x1[i] = act(matmul(W0[i], x0[i-l01]))
    x2[i] = matmul(W1[i], x1[i-l12])

def forward_timestep_lag_corrected(i, x0, x1, x2, W0, W1, l01, l12, act):
    from numpy import matmul # TODO: Replace with dot
    x1[i] = act(matmul(W0[i], x0[i-l01]+(x0[i-l01]-x0[i-l01-1])*l01))
    x2[i] = matmul(W1[i], x1[i-l12]+(x1[i-l12]-x1[i-l12-1])*l12)

def backward_timestep(i, x0, x1, x2, g0, g1, g2, W0, W1, dact):
    from numpy import matmul # TODO: Replace with dot
    g1[i] = matmul(g2[i], W1[i])
    g0[i] = matmul(g1[i]*dact(x1[i]), W0[i])

def backward_timestep_lag(i, x0, x1, x2, g0, g1, g2, W0, W1, l01, l12, dact):
    from numpy import matmul # TODO: Replace with dot
    g1[i] = matmul(g2[i-l12], W1[i])
    g0[i] = matmul(g1[i-l01]*dact(x1[i]), W0[i])

def backward_timestep_lag_corrected(i, x0, x1, x2, g0, g1, g2, W0, W1, l01, l12, dact):
    from numpy import matmul # TODO: Replace with dot
    g1[i] = matmul(g2[i-l12]+(g2[i-l12]-g2[i-l12-1])*l12, W1[i])
    g0[i] = matmul((g1[i-l01]+(g1[i-l01]-g1[i-l01-1])*l01)*dact(x1[i-2*(l01+l12)]), W0[i])

### ###

### Optimization ###

def GradientDescent(grad, x, alpha=1e-3, **kwargs):
    '''
    The standard gradient descent algorithm, often called SGD
    '''
    xn = x-alpha*grad(x, **kwargs) # Standard gradient descent
    return xn

def GradientMomentumDescent(grad, x, v, alpha=1e-3, rho=0.99, **kwargs):
    '''
    Gradient descent with momentum, often SGDm
    Note that rho=0 corresponds to zero momentum, while rho=1 corresponds to SGD
    This definition of rho is different to some in the literature: rho' = 1-rho
    '''
    vn = v*(1.-rho)-alpha*grad(x, **kwargs)
    xn = x+vn
    return xn, vn

def Nesterov(grad, x, v, alpha=1e-3, rho=0.99, **kwargs):
    '''
    Similar to SGDm but gradient term is evaluated as if a step had already been taken
    This leads to faster convergence in practice
    '''
    vn = v*(1.-rho)-alpha*grad(x+v*(1.-rho), **kwargs)
    xn = x+vn
    return xn, vn

def ViscousDescent(grad, x, v, alpha=1e-3, rho=0.99, **kwargs):
    '''
    Similar to gradient descent with momentum but with a v^2 friction term, rather than v
    This leads to a velocity-unit depdence of the parameter rho, which is annoying
    '''
    from numpy.linalg import norm
    vn = v*(1.-rho*norm(v))-alpha*grad(x, **kwargs)
    xn = x+vn
    return xn, vn

def AdaGrad(grad, x, g2_sum, alpha=1e-3, eps=1e-7, **kwargs):
    grad = grad(x, **kwargs)
    g2_sumn = g2_sum+grad**2 # Accumulate sum of gradient-squares in each direction; returns a vector
    xn = x-alpha*grad/(np.sqrt(g2_sumn)+eps)
    return xn, g2_sumn

def RMSProp(grad, x, g2, alpha=1e-3, rho=0.99, eps=1e-7, **kwargs):
    grad = grad(x, **kwargs)
    g2n = rho*g2+(1.-rho)*grad**2 # Accumulates a sum of gradient-squares that decays; returns a vector
    xn = x-alpha*grad/(np.sqrt(g2n)+eps)
    return xn, g2n

def Adam(grad, x, m1, m2, t, alpha=1e-3, beta1=0.99, beta2=0.99, eps=1e-7, unbiasing=True, **kwargs):
    '''
    Famous Adam algoirthm, a combination of AdaGrad/RMSProp and momentum
    Parameter beta1 controls friction term while beta2 controls gradient penalty
    '''
    grad = grad(x, **kwargs)
    m1n = beta1*m1+(1.-beta1)*grad    # Sort of like velocity/momentum/friction part
    m2n = beta2*m2+(1.-beta2)*grad**2 # Gradient penalty
    if unbiasing and t != 0: # This avoids problems with division by ~0 with m2 term
        u1 = m1n/(1.-beta1**t)
        u2 = m2n/(1.-beta2**t)
    else:
        u1 = m1n
        u2 = m2n
    xn = x-alpha*u1/(np.sqrt(u2)+eps)
    return xn, m1n, m2n

### ###

### Loss ###

def softmax(x, T):
    '''
    Converts any set of scores x into something that could be interpreted as a probability.
    The vector of scores, x, could initially be positive or negative; ordering is preserved.
    T is an effective 'temperature', which governs the relative contributions of different x
    '''
    ex = np.exp(x/T)
    return ex/ex.sum()

### ###
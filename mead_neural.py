### Activation functions ###

# TODO: Make classes with forward- and backward-propagation methods

def LU(x): # Linear Unit
    return x

def sigmoid(x):
    from numpy import exp
    return 1./(1.+exp(-x))

def zeroed_sigmoid(x):
    return 2.*sigmoid(x)-1.

def tanh(x):
    from numpy import tanh as nptanh
    return nptanh(x)

def ReLU(x): # Rectified Linear Unit
    from numpy import where
    return where(x<0., 0., x)

def leakyReLU(x, alpha=0.1): # Leaky Rectified Linear Unit
    from numpy import where
    return where(x<0., alpha*x, x)

def ELU(x, alpha=1.): # Exponential Linear Unit
    from numpy import where, exp
    return where(x<0., alpha*(exp(x)-1.), x)

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
    from numpy import where
    return where(x<0., 0., 1.)

def d_leakyReLU(x, alpha=0.1):
    from numpy import where
    return where(x<0., alpha, 1.)

def d_ELU(x, alpha=1.):
    from numpy import where, exp
    return where(x<0., alpha*exp(x), 1.)

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
    from numpy import sqrt
    grad = grad(x, **kwargs)
    g2_sumn = g2_sum+grad**2 # Accumulate sum of gradient-squares in each direction; returns a vector
    xn = x-alpha*grad/(sqrt(g2_sumn)+eps)
    return xn, g2_sumn

def RMSProp(grad, x, g2, alpha=1e-3, rho=0.99, eps=1e-7, **kwargs):
    from numpy import sqrt
    grad = grad(x, **kwargs)
    g2n = rho*g2+(1.-rho)*grad**2 # Accumulates a sum of gradient-squares that decays; returns a vector
    xn = x-alpha*grad/(sqrt(g2n)+eps)
    return xn, g2n

def Adam(grad, x, m1, m2, t, alpha=1e-3, beta1=0.99, beta2=0.99, eps=1e-7, unbiasing=True, **kwargs):
    '''
    Famous Adam algoirthm, a combination of AdaGrad/RMSProp and momentum
    Parameter beta1 controls friction term while beta2 controls gradient penalty
    '''
    from numpy import sqrt
    grad = grad(x, **kwargs)
    m1n = beta1*m1+(1.-beta1)*grad    # Sort of like velocity/momentum/friction part
    m2n = beta2*m2+(1.-beta2)*grad**2 # Gradient penalty
    if unbiasing and t != 0: # This avoids problems with division by ~0 with m2 term
        u1 = m1n/(1.-beta1**t)
        u2 = m2n/(1.-beta2**t)
    else:
        u1 = m1n
        u2 = m2n
    xn = x-alpha*u1/(sqrt(u2)+eps)
    return xn, m1n, m2n

### ###

### Loss ###

def softmax(x):
    '''
    Converts any set of scores x into something that could be interpreted as a probability.
    The vector of scores, x, could initially be positive or negative; ordering is preserved.
    '''
    from numpy import exp
    ex = exp(x)
    return ex/ex.sum()

### ###
### Activation functions ###

def sigmoid(x):
    from numpy import exp
    return 1./(1.+exp(-x))

def ReLU(x): # Rectified Linear Unit
    from numpy import where
    return where(x<0., 0., x)

def leakyReLU(x, alpha=0.1): # Leaky Rectified Linear Unit
    from numpy import where
    return where(x<0., alpha*x, x)

def LU(x): # Linear Unit
    return x

def ELU(x, alpha=1.): # Exponential Linear Unit
    from numpy import where, exp
    return where(x<0., alpha*(exp(x)-1.), x)

### ###

### Optimization ###

def GradientDescent(grad, x, alpha=1e-3, **kwargs):
    xn = x-alpha*grad(x, **kwargs) # Standard gradient descent
    return xn

def GradientMomentumDescent(grad, x, v, alpha=1e-3, rho=0.99, **kwargs):
    vn = rho*v-alpha*grad(x, **kwargs) # Momentum term, rho is like friction
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
    from numpy import sqrt
    grad = grad(x, **kwargs)
    m1n = beta1*m1+(1.-beta1)*grad    # Sort of like velocity/momentum
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
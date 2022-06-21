import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad

def mean_anomaly(nu, e, approximate=False):
    '''
    Compute the mean anomaly corresponding to the true anomaly via integration
    '''
    if approximate: # Taylor expansion for low e
        result = nu-2.*e*np.sin(nu)
    else:
        integrand = lambda x: 1./(1.+e*np.cos(x))**2
        integral, _ = quad(integrand, 0., nu)
        result = ((1.-e**2)**1.5)*integral
    return result

def true_anomaly(M, e, approximate=False):
    '''
    Compute the true anomaly from the mean anomaly using root finding
    '''
    #nu = M+2.*e*np.sin(M) # Approximate numerical inversion for small e
    f = lambda nu: mean_anomaly(nu, e, approximate)-M
    nu = fsolve(f, M) # Initial guess that nu = M
    return nu[0]
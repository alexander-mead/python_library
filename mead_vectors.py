# Third-party imports
import numpy as np


def unit(x):
    '''
    Create a unit vector from a non-unit vector
    '''
    return x/np.linalg.norm(x)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    '''
    Check if a square array/matrix is symmetric
    '''
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_antisymmetric(a, rtol=1e-05, atol=1e-08):
    '''
    Check if a square array/matrix is antisymmetric
    '''
    return np.allclose(a, -a.T, rtol=rtol, atol=atol)


def check_antisymmetric(a, rtol=1e-05, atol=1e-08):
    '''
    Check if a square array/matrix is Hermitian
    '''
    return np.allclose(a, np.conj(a.T), rtol=rtol, atol=atol)

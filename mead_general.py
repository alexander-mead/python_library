### This set of functions use only basic Python ###

def opposite_side(left_or_right):
    if left_or_right == 'left':
        out = 'right'
    elif left_or_right == 'right':
        out = 'left'
    else:
        raise ValueError('Input should be left or right')
    return out

def number_name(n):
    if n == int(1e1):
        return 'ten'
    elif n == int(1e2):
        return 'hundred'
    elif n == int(1e3):
        return 'thousand'
    elif n == int(1e4):
        return 'ten thousand'
    elif n == int(1e5):
        return 'hundred thousand'
    elif n == int(1e6):
        return 'million'
    elif n == int(1e9):
        return 'billion'
    else:
        raise ValueError('Integer does not appear to have a name')
        #return None

def file_length(fname):
    '''
    Count the number of lines in a file
    https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
    '''
    with open(fname) as f:
        i = 0
        for i, _ in enumerate(f):
            pass
    return i+1

def mrange(a, b=None):
    '''
    Mead range is more sensible than range, goes (1, 2, ..., n)
    I hate the inbuilt Python 'range' with such fury that it frightens me
    '''
    if b is None:
        return range(a+1)
    else:
        return range(a, b+1)

def is_float_close_to_integer(x):
    '''
    Checks if float is close to an integer value
    '''
    from math import isclose
    return isclose(x, int(x))

### ###

### These functions operate on collections (lists, tuples, sets, dictionaries) ###

def create_unique_list(list_with_duplicates):
    '''
    Takes a list that may contain duplicates and returns a new list with the duplicates removed
    TODO: Check that the ordering is preserved. Is the first occurance kept and later ones discarded?
    '''
    return list(dict.fromkeys(list_with_duplicates))

def remove_list_from_list(removal_list, original_list):
    '''
    Remove items in 'removal_list' if they occur in 'original_list'
    '''
    for item in removal_list:
        if item in original_list:
            original_list.remove(item)

### ###

### This set of functions use numpy ###

def arange(min, max):
    '''
    Sensible arange function
    I hate the inbuilt numpy one with such fury that it frightens me
    '''
    from numpy import arange
    return arange(min, max+1)

def logspace(xmin, xmax, nx):
    '''
    Return a logarithmically spaced range of numbers
    Numpy version is specifically base10, which is insane since log spacing is independent of base
    '''
    from numpy import logspace, log10
    return logspace(log10(xmin), log10(xmax), nx)

def is_power_of_two(x):
    '''
    True if arguement is a perfect power of two
    '''
    return is_power_of_n(x, 2)

def is_power_of_n(x, n):
    '''
    Checks if x is a perfect power of n (e.g., 32 = 2^5)
    '''
    from mead_special_functions import logn
    lg = logn(x, n)
    return is_float_close_to_integer(lg)

def is_perfect_square(x):
    '''
    Checks if argument is a perfect square (e.g., 16 = 4^2)
    '''
    from numpy import sqrt
    root = sqrt(x)
    return is_float_close_to_integer(root)

def is_perfect_triangle(x):
    '''
    Checks if argument is a perfect triangle number (e.g., 1, 3, 6, 10, ...)
    '''
    from numpy import sqrt
    n = 0.5*(sqrt(1.+8.*x)-1.)
    return is_float_close_to_integer(n)

def print_array_attributes(x):
    '''
    Print useful array attributes
    '''
    print('Array attributes')
    print('ndim:', x.ndim)
    print('shape:', x.shape)
    print('size:', x.size)
    print('dtype:', x.dtype)
    print('')

def print_full_array(xs):
    '''
    Print full array(-like) to screen with indices
    '''
    for ix, x in enumerate(xs):
        print(ix, x)

def nans(shape, **kwargs):
    '''
    Initialise an array of nans
    '''
    from numpy import empty, nan
    return nan*empty(shape, **kwargs)

def remove_nans(x):
    '''
    Remove nans from array x
    '''
    from numpy import isnan
    return x[~isnan(x)]

def array_values_at_indices(array, list_of_array_positions):
    '''
    Returns values of the array at the list of array position integers
    '''
    if len(array.shape) == 1:
        return array[list_of_array_positions]
    elif len(array.shape) == 2:
        ix, iy = zip(*list_of_array_positions)
        result = array[ix, iy]
        return result
    else:
        ValueError('Error, this only works in either one or two dimensions at the moment')
        return None

# use numpy deg2rad() or radians()
#def degrees_to_radians(theta):
#    from numpy import pi
#    return theta*pi/180.

# use numpy rad2deg or degrees()
#def radians_to_degrees(theta):
#    from numpy import pi
#    return theta*180./pi

# A sum function that returns nan if any of the values to be summed is a nan
# This should be the default behaviour of the np.sum function
#def nansum(a, **kwargs):
#    from numpy import isnan, nan, nansum
#    if isnan(a).any():
#        return nan
#    else:
#        return nansum(a, **kwargs)

### ###

### matplotlib ###

def seq_color(i, n, cmap):
    '''
    Sequential colors
    '''
    return cmap(i/(n-1))

def colour(i):
    '''
    Default colours (C0, C1, C2, ...)
    '''
    return 'C%d'%(i)

### ###
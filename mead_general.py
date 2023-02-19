import numpy as np

### Functions ###

### High level ###

def module_info(mod):
    print(mod.__name__, mod.__version__, mod.__file__)

def none_or_string(value):
    if value == 'None':
        result = None
    else:
        result = value
    return result

### ###

### This of functions use only basic Python ###

def periodic_integer(i, n):
    '''
    Ensure that integers can only exist on the range [0, 1, ..., n-1]
    Solution from: https://stackoverflow.com/questions/43827464
    '''
    return ((i%n)+n)%n

def periodic_float(x, L):
    '''
    Ensures that the float x can only exist on the interval [0, L)
    NOTE: This is just the modulus operation, but I always forget this
    '''
    return x%L

def opposite_side(left_or_right):
    '''
    Returns the opposite of the strings: 'left' or 'right'
    '''
    if left_or_right == 'left':
        out = 'right'
    elif left_or_right == 'right':
        out = 'left'
    else:
        raise ValueError('Input should be either \'left\' or \'right\'')
    return out

def number_name(n):
    '''
    The standard name for 10^n
    e.g., 10^2 = hundred
    e.g., 10^9 = billion
    '''
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

def mrange(a, b=None, step=None):
    '''
    Mead range is more sensible than range, goes (1, 2, ..., n)
    e.g., list(range(4)) = [1, 2, 3, 4]
    e.g., list(range(2, 4)) = [2, 3, 4]
    I hate the inbuilt Python 'range' with such fury that it frightens me
    NOTE: I don't think I can call this range since that would overwrite internal Python version
    TODO: Include step properly, what is this (start, stop[, step]) square-bracket thing?
    TODO: Note that Python range() is actually a class and returns an iterable
    '''
    if step is None:
        if b is None:
            return range(a+1)
        else:
            return range(a, b+1)
    else:
        if b is None:
            raise ValueError('If a step is specified then you must also specify start (a) and stop (b)')
        else:
            return range(a, b+1, step)

def is_float_close_to_integer(x):
    '''
    Checks if float is close to an integer value
    '''
    from math import isclose
    return isclose(x, int(x))

def bin_edges_for_integers(a, b=None):
    '''
    Defines a set of bin edges for histograms of integer data
    e.g., bin_edges_for_integers(1, 5) = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5 so that values are at centres
    e.g., bin_edges_for_integers(a) assumes a is array and goes from min-0.5 to max+0.5 in steps of unity
    '''
    if b is None:
        bin_edges = list(mrange(min(a), max(a)+1))
    else:
        bin_edges = list(mrange(a, b+1))
    bin_edges = [bin_edge-0.5 for bin_edge in bin_edges] # Centre on integers
    return bin_edges

def key_from_value(dict, value):
    '''
    Get a dictionary key from the value
    From: https://stackoverflow.com/questions/8023306
    This is often a bad idea, and is not really how dictionary should be used
    '''
    return list(dict.keys())[list(dict.values()).index(value)]

### ###

### These functions operate on collections (lists, tuples, sets, dictionaries) ###

def count_entries_of_nested_list(listOfElem):
    ''' Get number of elements in a nested list'''
    count = 0
    for elem in listOfElem: # Iterate over the list
        if type(elem) == list: # Check if type of element is list
            count += count_entries_of_nested_list(elem) # Recursive call
        else:
            count += 1
    return count

def create_unique_list(list_with_duplicates):
    '''
    Takes a list that may contain duplicates and returns a new list with the duplicates removed
    '''
    #return list(set(list_with_duplicates)) # NOTE: This does not preserve order
    return list(dict.fromkeys(list_with_duplicates))

def remove_list_from_list(removal_list, original_list):
    '''
    Remove items in 'removal_list' if they occur in 'original_list'
    NOTE: This only removes the first occurance of something in a list, care if repeated entries
    '''
    new_list = original_list.copy()
    for item in removal_list:
        if item in new_list:
            new_list.remove(item)
    return new_list

def remove_multiple_elements_from_list(original_list, indices):
    '''
    Remove multiple elements from a list by providing a list of indices
    '''
    new_list = original_list.copy()
    for index in sorted(indices, reverse=True):
        del new_list[index]
    return new_list

def second_largest(numbers):
    '''
    Returns the second-largest entry in collection of numbers
    '''
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

### ###

### This set of functions use numpy ###

def is_array_monotonic(x:np.array) -> bool:
    '''
    Returns True iff the array contains monotonically increasing values
    '''
    return np.all(np.diff(x) > 0.)

def is_array_linear(x:np.array, atol=1e-8) -> bool:
    '''
    Returns True iff the array is linearly spaced
    '''
    return np.isclose(np.all(np.diff(x)-np.diff(x)[0]), 0., atol=atol)

def arange(min:float, max:float, dtype=None) -> np.ndarray:
    '''
    Sensible arange function for producing integers from min to max inclusive
    I hate the inbuilt numpy one with such fury that it frightens me
    TODO: Include step properly, what is this ([start,] stop[, step], ...) square-bracket thing?
    '''
    return np.arange(min, max+1, dtype=dtype)#, like=like)

def linspace_step(start:int, stop:int, step):
    '''
    Create a linear-spaced array going from start->stop (inclusive) via step
    The end point is included only if it falls exaclty on a step
    TODO: Check this actually works with float division issues
    '''
    num = 1+int((stop-start)/step)
    new_stop = start+(num-1)*step
    return np.linspace(start, new_stop, num)

def logspace(xmin, xmax, nx):
    '''
    Return a logarithmically spaced range of numbers
    Numpy version is specifically base10, which is insane since log spacing is independent of base
    '''
    return np.logspace(np.log10(xmin), np.log10(xmax), nx)

def is_power_of_n(x, n):
    '''
    Checks if x is a perfect power of n (e.g., 32 = 2^5)
    '''
    lg = np.logn(x, n)
    return is_float_close_to_integer(lg)

def is_power_of_two(x):
    '''
    True if argument is a perfect power of two
    '''
    return is_power_of_n(x, 2)

def is_perfect_square(x):
    '''
    Checks if argument is a perfect square (e.g., 16 = 4^2)
    '''
    root = np.sqrt(x)
    return is_float_close_to_integer(root)

def is_perfect_triangle(x):
    '''
    Checks if argument is a perfect triangle number (e.g., 1, 3, 6, 10, ...)
    '''
    n = 0.5*(np.sqrt(1.+8.*x)-1.)
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
    print()

def print_full_array(xs, title=None):
    '''
    Print full array(-like) to screen with indices
    # TODO: Look at pprint module
    '''
    if title is not None:
        print(title)
    for ix, x in enumerate(xs):
        print(ix, x)

def array_of_nans(shape, **kwargs):
    '''
    Initialise an array of nans
    '''
    return np.nan*np.empty(shape, **kwargs)

def remove_nans_from_array(x):
    '''
    Remove nans from array x
    '''
    return x[~np.isnan(x)]

def array_contains_nan(array):
    '''
    Returns True if the array contains any nan values
    '''
    return np.isnan(array.sum())

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

def print_array_statistics(x):
    '''
    Print useful array statistics
    '''
    print('Array statistics')
    n = x.size
    print('size:', n)
    print('sum:', x.sum())
    print('min:', x.min())
    print('max:', x.max())
    mean = x.mean()
    print('mean:', mean)
    std = x.std()
    var = std**2
    std_bc = std*np.sqrt(n/(n-1))
    var_bc = std_bc**2
    print('std:', std)
    print('std (Bessel corrected):', std_bc)
    print('variance:', var)
    print('variance (Bessel corrected):', var_bc**2)
    print('<x^2>:', mean**2+var)
    print()

def standardize_array(x):
    '''
    From: https://towardsdatascience.com/pca-with-numpy-58917c1d0391
    This function standardizes an array, its substracts mean value, 
    and then divide the standard deviation.
        x: array 
        return: standardized array
    NOTE: In sklearn the 'StandardScaler' exists to do exactly this
    '''
    rows, columns = x.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for col in range(columns):
        mean = x[:, col].mean(); std = x[:, col].std()
        tempArray = np.empty(0)
        for element in x[:, col]:
            tempArray = np.append(tempArray, (element-mean)/std)
        standardizedArray[:, col] = tempArray
    return standardizedArray

def covariance_matrix(sigmas, R):
    '''
    Creates an nxn covariance matrix from a correlation matrix
    Covariance is matrix multiplication of S R S where S = diag(sigmas)
    @params
        sigmas - sigmas for the diagonal
        R - correlation matrix (nxn)
    TODO: Could save memory by having the sigmas run the diagonal of the correlation matrix
    '''
    S = np.diag(sigmas)
    cov = np.matmul(np.matmul(S, R), S)
    return cov

def find_closest_index_value(x:float, arr:np.array):
    '''
    Find the index, value pair of the closest values in array 'arr' to value 'x'
    '''
    index = (np.abs(arr-x)).argmin()
    return index, arr[index]

### ###

### matplotlib ###

def seq_color(i, n, cmap='viridis'):
    '''
    Sequential colors from i=0 to i=n-1 to select n colors from cmap
    '''
    from matplotlib.pyplot import get_cmap
    if isinstance(cmap, str):
        cmap_here = get_cmap(cmap)
    else:
        cmap_here = cmap
    return cmap_here(i/(n-1))

def colour(i):
    '''
    Default colours (C0, C1, C2, ...)
    '''
    return 'C%d'%(i)

def plot_curve_with_error_region(x, y, axis=0, alpha_fill=0.2, label=None, **kwargs):
    '''
    Plot a curve with a transparent error region behind it
    The curve is taken to be the mean of the day (array y along 'axis')
    The extent of the error region is taken to be the standard deviation (array y along 'axis')
    @params
        x (n array): x positions of the data
        y (nxm array): y possitions of the data
    '''
    from matplotlib.pyplot import plot, fill_between
    mean = y.mean(axis=axis)
    std = y.std(axis=axis)
    fill_between(x, mean-std, mean+std, alpha=alpha_fill, **kwargs)
    plot(x, mean, label=label, **kwargs)

### ###
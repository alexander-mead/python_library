import numpy as np
import matplotlib.pyplot as plt

# See scikit-image for lots of functions

### Filters ###

# Sobel edge-detection filter in the x direction
# y-direction filter is the transpose of this
# This approximates the numerical gradient of the image
Sobel_filter = np.array([
    [1., 0., -1.],
    [2., 0., -2.],
    [1., 0., -1.],
])

### ###

### Functions ###

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def invsigmoid(x):
    return np.log(x/(1.-x))

### ###

### Utility ###

def convert_8bit_to_float(X):
    return X.astype(float)/255

def convert_float_to_8bit(X):
    return (X*255).astype(np.uint8)

def convert_to_greyscale(X):
    #R = 0.299; G = 0.587; B = 0.144 # RGB weights for optimal visual greyscale
    R = 0.2126; G = 0.7152; B = 0.0722
    #R = 1./3.; G = 1./3.; B = 1./3.
    return R*X[:, :, 0]+G*X[:, :, 1]+B*X[:, :, 2]

def image_properties(X):
    '''
    Print some useful image properties
    '''
    print('Image shape:', X.shape)
    print('Data type:', X.dtype)
    nr = X.shape[0]; nc = X.shape[1] # Numbers of rows and columns
    print('Number of rows (y dimension):', nr)
    print('Number of columns (x dimension):', nc)
    print('Total number of pixels:', nr*nc)
    print('Total number of megapixels:', nr*nc/1e6)
    print('Number of colour channels:', X.shape[2])
    print('Example pixel value(s):', X[0, 0, :])
    print()

def plot_color_image(X, **kwargs):
    plt.figure(**kwargs)
    plt.imshow(X)
    plt.xticks([]); plt.yticks([])
    plt.show()

def plot_greyscale_image(X, vmin=None, vmax=None, **kwargs):
    plt.figure(**kwargs)
    plt.imshow(X, cmap='gray', vmin=vmin, vmax=vmax)
    plt.xticks([]); plt.yticks([])
    plt.show()

def plot_color_histograms(X, bins=64):
    '''
    Histograms of RGB values
    '''
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        plt.hist(np.ravel(X[:, :, i]), bins=bins, density=True, color=color, alpha=0.5)
    plt.xlabel('RGB value')
    plt.xlim((0, 255))
    plt.yticks([])
    plt.show()

### ###

### Processing ###

def zero_filter(f):
    '''
    Make a filter sum to zero by filling zeros with negative values
    '''
    weight = f.sum()
    num_zeros = f.size-np.count_nonzero(f)
    fill = -weight/num_zeros
    #print(weight, num_zeros, fill, f.size, np.count_nonzero(f))
    F = np.where(f==0., fill, f)
    return F

def normalize_filter(f):
    '''
    Make a filter sum to unity
    '''
    return f/f.sum()

def segmented_image(X, n, random_state=None, random_sample_size=None):
    '''
    Take an RGB image and break into 'n' distinct colour regions
    Uses k-means clustering to choose colours for the segmentation
    @params
        X: image (nr, nc, 3)
        n: Number of clusters
        random_state: Random state for the k-means algorithm
            should be less than the number of pixels
        random_sample_size: Number of pixels to sample for means
    '''
    from numpy.random import choice
    from sklearn.cluster import KMeans
    nr, nc, _ = X.shape
    if random_sample_size is None:
        Xflat = X.reshape(nr*nc, 3)
    else:
        ir = choice(X.shape[0], random_sample_size)#, replace=True)
        ic = choice(X.shape[1], random_sample_size)#, replace=True)
        Xflat = X[ir, ic, :]
    kmeans = KMeans(n_clusters=n, random_state=random_state).fit(Xflat)
    Y = kmeans.cluster_centers_[kmeans.predict(X.reshape(nr*nc, 3))]
    Y = Y.reshape(nr, nc, 3)/255. # Need to divide by 255 since means are non-integer
    return Y

def Sobel_edge(X, **kwargs):
    from scipy.signal import convolve2d
    dx = convolve2d(X, Sobel_filter, **kwargs)
    dy = convolve2d(X, Sobel_filter.T, **kwargs)
    return np.sqrt(dx**2+dy**2)

def mean_filter(npix):
    F = np.ones((npix, npix))
    return normalize_filter(F)

def Gaussian_filter(npix, spix):
    F = np.empty((npix, npix))
    if npix%2 == 0:
        raise ValueError('Gaissiam filter must be an odd number of pixels')
    mid = (npix-1)//2
    for ir in range(npix):
        for ic in range(npix):
            ix = ir-mid; iy = ic-mid
            F[ir, ic] = np.exp(-0.5*(ix**2+iy**2)/spix**2)
    return normalize_filter(F)

# def mean_smoothing(X, npix=3, **kwargs):
#     from scipy.signal import convolve2d
#     F = normalize_filter(np.ones((npix, npix)))
#     return convolve2d(X, F, **kwargs)

# def Gaussian_smoothing(X, npix=3, spix=1, **kwargs):
#     from scipy.signal import convolve2d
#     F = Gaussian_filter(npix, spix)
#     return convolve2d(X, F, **kwargs)

def smooth(X, npix=3, spix=1, method='Gaussian', **kwargs):
    from scipy.signal import convolve2d
    if method == 'Gaussian':
        F = Gaussian_filter(npix, spix)
    elif method == 'mean':
        F = mean_filter(npix)
    else:
        raise ValueError('Smoothing method not recognized')
    return convolve2d(X, F, **kwargs)

### ###
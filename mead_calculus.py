### Differentiation ###

def derivative(f, x, dx=1e-3):
    '''
    Calculate the numerical derivative of f(x) at the point x: df/dx
    NOTE: scipy.misc.derivative
    '''
    hdx = dx/2.
    df = f(x+hdx)-f(x-hdx) # Two-sided difference in numerator
    return df/dx


def log_derivative(f, x, dx=1e-3):
    '''
    Calculate the logarithmic derivative of f(x) at the point x: dln(f)/dln(x)
    NOTE: scipy.misc.derivative
    '''
    from numpy import log
    hdx = dx/2.
    dlnf = log(f(x+hdx)/f(x-hdx)) # Two-sided difference in numerator
    dlnx = log((x+hdx)/(x-hdx)) # Two-sided (is this necessary?)
    #dlnx = log(1.+dx/x) # Using this is probably fine; for dx<<x they are equal
    return dlnf/dlnx


def gradient(f, x, dx=1e-3):
    '''
    Calculates the gradient of scalar function f(x) defined for vector x
    Returns a vector of the gradients of f(x) at the point x
    TODO: dx could also be a vector
    TODO: Loop is probably very slow here
    '''
    from numpy import empty, zeros
    df = empty(len(x))
    for i, _ in enumerate(x):
        hdx = zeros(len(x))
        hdx[i] = dx/2.
        df[i] = (f(x+hdx)-f(x-hdx))/dx
    return df


def derivative_from_samples(x, xs, fs):
    '''
    Calculates the derivative of the function f(x) which is sampled as fs at values xs
    Approximates the function as quadratic using the samples and Legendre polynomials
    Args:
        x: Point at which to calculate the derivative
        xs: Sample locations
        fs: Value of function at sample locations
    '''
    from mead_general import find_closest_index_value
    from scipy.interpolate import lagrange
    ix, _ = find_closest_index_value(xs, x)
    if ix == 0:
        (imin, imax) = (0, 1) if x < xs[0] else (0, 2)
    elif ix == len(xs)-1:
        nx = len(xs)
        (imin, imax) = (nx-2, nx-1) if x > xs[-1] else (nx-3, nx-1)
    else:
        imin, imax = ix-1, ix+1
    poly = lagrange(xs[imin:imax+1], fs[imin:imax+1])
    return poly.deriv()(x)

### ###

### Integration ###

def integrate_quad_log(func,a,b,\
                       args=(),\
                       full_output=0,\
                       epsabs=1.49e-08,\
                       epsrel=1.49e-08,\
                       limit=50,\
                       points=None,\
                       weight=None,\
                       wvar=None,\
                       wopts=None,\
                       maxp1=50,\
                       limlst=50):
    '''
    A routine to integrate in log space. 
    This may actually be pretty useless... not sure. Should do speed tests
    TODO: Surely can use *args and **kwargs here. This is ugly as fuck.
    '''
    from numpy import log, exp
    from scipy import integrate
    loga=log(a); logb=log(b)
    ans=integrate.quad(lambda x, *args: exp(x)*func(exp(x), *args), loga, logb,\
                       args=args,\
                       full_output=full_output,\
                       epsabs=epsabs,\
                       epsrel=epsrel,\
                       limit=limit,\
                       points=points,\
                       weight=weight,\
                       wvar=wvar,\
                       wopts=wopts,\
                       maxp1=maxp1,\
                       limlst=limlst)
    return ans


def integrate_rectangular(fx, x):
    '''
    A very simple rectangular integration that assumes equal sized bins
    This is zeroth order - worse than trapezium rule - but consistent for some binned data
    '''
    dx = x[1]-x[0]
    return sum(fx)*dx


def trapz2d(F, x, y):
    '''
    Two-dimensional trapezium rule
    First integrates along x for each y, and then y
    '''
    from numpy import trapz
    Fmid = trapz(F, x)
    return trapz(Fmid, y)

### ###
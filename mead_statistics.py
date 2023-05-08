# Third-party imports
import numpy as np
from scipy.integrate import quad

### Continuous probability distributions ###

def normalisation(f, x1, x2, *args):
    '''
    Calculate the normalisation of a probability distribution via integration
    f: f(x) probability distribution function
    x1, x2: Limits over which to normalise (could be -inf to +inf)
    *args: Arguments to be passed to function
    '''
    norm, _ = quad(lambda x: f(x, *args), x1, x2)
    return norm

def cumulative(x, f, x1, *args):
    '''
    Compute the cumulative distribution function via integration
    '''
    C, _ = quad(lambda x: f(x,*args), x1, x)
    return C
cumulative = np.vectorize(cumulative) # Vectorise to work with array 'x'

def draw_from_distribution(x, Cx):
    '''
    Draw a random number given an cumulative probability function
    x: array of x values
    C(x): array for cumulative probability
    '''
    r = np.random.uniform(Cx[0], Cx[-1])
    xi = np.interp(r, Cx, x)
    return xi

def moment(n, f, x1, x2, *args):
    '''
    Compute the n-th moment of a continuous distribution via integration
    n: order of moment
    f: f(x)
    x1, x2: low and high limits
    *args: arguments to be passed to function
    '''
    norm = normalisation(f, x1, x2, *args)
    m, _ = quad(lambda x: (x**n)*f(x, *args)/norm, x1, x2)
    return m

def mean(f, x1, x2, *args):
    '''
    Computes the mean of a continuous distribution via integration
    '''
    return moment(1, f, x1, x2, *args)

def variance(f, x1, x2, *args):
    '''
    Compute the variance of a continuous distribution via integration
    '''
    m1 = moment(1, f, x1, x2, *args)
    m2 = moment(2, f, x1, x2, *args)
    return m2-m1**2

### ###

### Drawing random numbers from continuous distributions ###

def draw_from_1D(n, f, x1, x2, nx, *args):
    '''
    Draw random numbers from a continuous 1D distribution
    n: number of draws to make from f
    f: f(x) array to draw from
    x1, x2: limits on x axis
    nx: number of points to use along x axis
    '''
    x = np.linspace(x1, x2, nx)
    C = cumulative(x, f, x1, *args)
    xi = np.zeros(n)
    for i in range(n):
        xi[i] = draw_from_distribution(x, C)
    return xi

def draw_from_2D(n, f, x1, x2, nx, y1, y2, ny):
    '''
    Draw random numbers from a 2D distribution
    n - number of draws to make from f
    f - f(x,y) to draw from
    x1, x2 - limits on x axis
    y1, y2 - limits on y axis
    nx, ny - number of points along x and y axes
    '''
    # Pixel sizes in x and y
    dx = (x2-x1)/np.real(nx)
    dy = (y2-y1)/np.real(ny)

    # Linearly spaced arrays of values corresponding to pixel centres
    x = np.linspace(x1+dx/2., x2-dx/2., nx)
    y = np.linspace(y1+dy/2., y2-dy/2., ny)

    # Make a grid of xy coordinate pairs
    xy = np.array(np.meshgrid(x, y))
    xy = xy.reshape(2, nx*ny) # Reshape the grid (2 here coresponds to 2 coordinates: x, y)
    xy = np.transpose(xy).tolist() # Convert to a long list
    
    # Make array of function values corresponding to the xy coordinates
    X, Y = np.meshgrid(x, y)
    z = f(X, Y)      # Array of function values
    z = z.flatten() # Flatten array to create a long list of function values
    z = z/sum(z)    # Force normalisation

    # Make a list of integers linking xy coordiantes to function values
    i = list(range(z.size)) 

    # Make the random choices with probabilties proportional to the function value
    # The integer chosen by this can then be matched to the xy coordiantes
    j = np.random.choice(i,n,replace=True,p=z) 
    
    # Now match the integers to the xy coordinates
    xs = []
    ys = []
    for i in range(n):
        xi, yi = xy[j[i]]
        xs.append(xi)
        ys.append(yi)

    # Random numbers for inter-pixel displacement
    dxs = np.random.uniform(-dx/2., dx/2., n)
    dys = np.random.uniform(-dy/2., dy/2., n)

    # Apply uniform-random displacement within a pixel
    xs = xs+dxs
    ys = ys+dys
        
    return xs, ys

### ###

### Other ###

def correlation_matrix(cov):
    '''
    Calculate a correlation matrix from a covariance matrix
    '''
    shape = cov.shape
    n = shape[0]
    if n != shape[1]:
        raise TypeError('Input covariance matrix must be square')
    cor = np.empty_like(cov)
    
    for i in range(n):
        for j in range(n):
            cor[i, j] = cov[i, j]/np.sqrt(cov[i, i]*cov[j, j]) 
            
    return cor

### ###

### Integer distributions ###

def central_condition_Poisson(n, lam, p):
    '''
    Probability mass function for a Poisson distribution affected by the central condition
    n: p(n); n is an integer
    lam: mean value for the underlying Poisson distribution (not the mean value of this distribution)
    p: probability of hosting a central galaxy
    '''
    from scipy.stats import poisson
    PMF = poisson.pmf(n, lam)
    PMF = np.where(n == 0, p*PMF+1.-p, p*PMF)
    return PMF

def expectation_integer_distribution(p, f, nmax, *args):
    '''
    The expectation value of some function of an integer probability distribuion
    P(n, *args): Probability distribution (mass) function
    f(n): Function over which to compute expectation value (e.g., f(n)=n is mean; f(n)=n^2 is second moment)
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    ns = np.arange(nmax+1)
    ps = p(ns, *args)
    return np.sum(f(ns)*ps)

def moment_integer_distribution(p, pow, nmax, *args):
    '''
    Compute the moment of an integer distribution via direct summation
    p(n, *args): Probability distribution (mass) function
    pow: Order for moment (0 - normalisation; 1 - mean; 2 - second moment)
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    return expectation_integer_distribution(p, lambda n: n**pow, nmax, *args)

def sum_integer_distribution(p, nmax, *args):
    '''
    Compute the sum of probabilities for integer distribution via direct summation (should be unity)
    p(n, *args): Probability distribution (mass) function
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    return moment_integer_distribution(p, 0, nmax, *args)

def mean_integer_distribution(p, nmax, *args):
    '''
    Compute the mean value of an integer distribution via direct summation
    p(n, *args): Probability distribution (mass) function
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    return moment_integer_distribution(p, 1, nmax, *args)

def variance_integer_distribution(p, nmax, *args):
    '''
    Compute the variance of an integer distribution via direct summation
    p(n, *args): Probability distribution (mass) function
    nmax: Maximum n to compute sum
    *args: to be passed to p(n, *args)
    '''
    mom1 = moment_integer_distribution(p, 1, nmax, *args)
    mom2 = moment_integer_distribution(p, 2, nmax, *args)
    return mom2-mom1**2

### ###

### MCMC ###

def Gibbs_sampling(conditionals, start, n_chains=5, n_points=int(1e3), burn_frac=0.5):
    '''
    Simple Gibbs sampling with m chains of length n
    conditionals: sampling function from conditional probability distributions
        The i-th conditional distribution is supposed to be P(x_i | x_{-i}; y)
        However, the function call is P(x) with array x
        The i-th element of x is supposed to be the random variable 
        with all other components of x being held fixed (conditioned upon)
    start: starting location in parameter space
    n_chains: Number of independent chains
    n_points: Number of points per chain
    burn_frac: Fraction of the beginning of the chain to remove
    '''
    chains = []
    for _ in range(n_chains):
        x = start; xs = []
        for _ in range(n_points):
            for i, conditional in enumerate(conditionals):
                x[i] = conditional(x)
            xs.append(x.copy())
        chains.append(np.array(xs))
    chains = burn_chains(chains, burn_frac=burn_frac)
    return chains

def MCMC_sampling(proposal, f, start, n_chains=5, n_points=int(1e3), burn_frac=0.5):
    '''
    Simple MCMC with m chains of length n
    proposal: p(x) function to sample from proposal distribution
    f: f(x) target function
    start: starting location in parameter space
    n_chains: Number of independent chains
    n_points: Number of points per chain
    burn_frac: Fraction of the beginning of the chain to remove
    '''
    chains = []
    for _ in range(n_chains):
        x = np.copy(start); p = f(x)
        xs = []; x_old = x; p_old = p
        for _ in range(n_points):
            x_new = proposal(x_old)            # Sample from the proposal
            p_new = f(x_new)                   # New probability
            acceptance = min(p_new/p_old, 1)   # Acceptance probability
            accept = np.random.uniform(0., 1.) # Accept or reject
            if accept < acceptance:
                x_old = x_new; p_old = p_new
            #if x_old != start: xs.append(x_old) # Avoid adding the first sample?
            xs.append(x_old)
        chains.append(np.array(xs))
    chains = burn_chains(chains, burn_frac=burn_frac)
    return chains

def HMC_sampling(lnf, dlnf, start, n_chains=5, n_points=int(1e3), burn_frac=0.5, M=1., dt=0.1, T=1., debug=False, verbose=False):
    '''
    Hamiltonian Monte Carlo with m chains of length n
    lnf: ln(f(x)) natural logarithm of the target function
    dlnf: grad[ln(f(x)] gradient of the natural logarithm of the target function
    start: starting location in parameter space
    n_chains: Number of independent chains
    n_points: Number of points per chain
    burn_frac: Fraction of the beginning of the chain to remove
    M: Mass for the 'particles' TODO: Make matrix
    dt: Time-step for the particles
    T: Integration time per step for the particles
    '''
    # Functions for leap-frog integration
    def leap_frog_step(x, p, dlnf, M, dt):
        p_half = p+0.5*dlnf(x)*dt
        x_full = x+p_half*dt/M
        p_full = p_half+0.5*dlnf(x_full)*dt
        return x_full, p_full
    def leap_frog_integration(x_init, p_init, dlnf, M, dt, T):
        N_steps = int(T/dt)
        x, p = np.copy(x_init), np.copy(p_init)
        for _ in range(N_steps):
            x, p = leap_frog_step(x, p, dlnf, M, dt)
        return x, p
    def Hamiltonian(x, p, lnf, M):
        T = 0.5*np.dot(p, p)/M
        V = -lnf(x)
        return T+V
    # MCMC step
    chains = []
    for j in range(n_chains):
        x_old = np.copy(start); xs = []; n_accepted = 0
        for i in range(n_points):
            p_old = np.random.normal(0., 1., size=x_old.size) # Randomize momentum each go
            if i == 0: H_old = 0.
            if debug: print('Prev sample:', i, x_old, p_old, H_old)
            x_new, p_new = leap_frog_integration(x_old, p_old, dlnf, M, dt, T)
            H_new = Hamiltonian(x_new, p_new, lnf, M)
            if debug: print('Next sample:', i, x_new, p_new, H_new)
            acceptance = 1. if (i == 0) else min(np.exp(H_old-H_new), 1)
            accept = np.random.uniform(0., 1.) < acceptance # Accept or reject
            if debug: print('Acceptance:', acceptance, accept)
            if accept: x_old = x_new; H_old = H_new; n_accepted += 1
            xs.append(x_old)
        chains.append(np.array(xs))
        if verbose: print('Chain: %d; acceptance fraction: %1.2f'%(j, n_accepted/n_points))
    chains = burn_chains(chains, burn_frac=burn_frac)
    if debug: exit()
    return chains

def HMC_sampling_torch(lnf, start, n_chains=5, n_points=int(1e3), burn_frac=0.5, M=1., dt=0.1, T=1., verbose=False):
    '''
    Hamiltonian Monte Carlo with m chains of length n
    lnf: ln(f(x)) natural logarithm of the target function
    start: starting location in parameter space
    n_chains: Number of independent chains
    n_points: Number of points per chain
    burn_frac: Fraction of the beginning of the chain to remove
    M: Mass for the 'particles' TODO: Make matrix
    dt: Time-step for the particles
    T: Integration time per step for the particles
    '''
    import torch as tc
    # Functions for leap-frog integration
    def get_gradient(x, lnf):
        x = x.detach()
        x.requires_grad_()
        lnf(x).backward()
        dlnfx = x.grad
        x = x.detach()
        return dlnfx
    def forward_Euler_step(x, p, lnf, M, dt):
        dlnfx = get_gradient(x, lnf)
        x_full = x+p*dt/M
        p_full = p+dlnfx*dt
        return x_full, p_full
    def leap_frog_step(x, p, lnf, M, dt):
        dlnfx = get_gradient(x, lnf)
        p_half = p+0.5*dlnfx*dt
        x_full = x+p_half*dt/M
        dlnfx = get_gradient(x_full, lnf)
        p_full = p_half+0.5*dlnfx*dt
        return x_full, p_full
    def leap_frog_integration(x_init, p_init, lnf, M, dt, T, method='leap-frog'):
        N_steps = int(T/dt)
        x, p = tc.clone(x_init), tc.clone(p_init)
        step = leap_frog_step if method=='leap-frog' else forward_Euler_step
        for _ in range(N_steps):
            x, p = step(x, p, lnf, M, dt)
        return x, p
    def Hamiltonian(x, p, lnf, M):
        T = 0.5*tc.dot(p, p)/M
        V = -lnf(x)
        return T+V
    # MCMC step
    chains = []; n = len(start)
    for j in range(n_chains):
        x_old = tc.clone(start); xs = []; n_accepted = 0
        for i in range(n_points):
            p_old = tc.normal(0., 1., size=(n,))
            if i == 0: H_old = 0.
            x_new, p_new = leap_frog_integration(x_old, p_old, lnf, M, dt, T)
            H_new = Hamiltonian(x_new, p_new, lnf, M)
            acceptance = 1. if (i == 0) else min(tc.exp(H_old-H_new), 1.) # Acceptance probability
            accept = tc.rand((1,)) < acceptance
            if accept: x_old = x_new; H_old = H_new; n_accepted += 1
            xs.append(x_old)
        chains.append(tc.stack(xs))
        if verbose: print('Chain: %d; acceptance fraction: %1.2f'%(j, n_accepted/n_points))
        chains = burn_chains(chains, burn_frac=burn_frac)
    return chains

def burn_chains(chains, burn_frac=0.5):
    '''
    Remove the first fraction of a chain as burn in
    '''
    n = len(chains[0])
    burned_chains = [chain[int(n*burn_frac):] for chain in chains]
    return burned_chains

def Gelman_Rubin_statistic(chains, verbose=False):
    '''
    Calculate the Gelman-Rubin statistic
    chains: A list of equal-length MCMC chains
    '''
    # Initial information
    m = len(chains); n = len(chains[0])
    d = 1 if (len(chains[0].shape) == 1) else chains[0].shape[1]
    if verbose:
        print('Number of chains:', m)
        print('Number of parameters:', d)
        print('Chain lengths:', n)

    # Calculate the mean and variance for each chain
    chain_means = []; chain_variances = []
    for i, chain in enumerate(chains):
        if len(chain) != n: raise ValueError('All chains must be the same length')
        chain_mean = chain.mean(axis=0)       # Sample mean of chain i
        chain_var = chain.var(axis=0, ddof=1) # Sample variance of chain i
        if verbose: print('Chain:', i, 'mean:', chain_mean, 'variance:', chain_var)
        chain_means.append(chain_mean); chain_variances.append(chain_var)
    chain_means = np.array(chain_means); chain_variances = np.array(chain_variances)

    # Calculate the intra-chain variances; this is s^2; underestimates variance
    B = n* np.atleast_1d(chain_means).var(axis=0, ddof=1) # Sample variance of sample means from chains
    overall_variance = np.atleast_1d(chain_variances).mean(axis=0)
    sigma2_hat = overall_variance*(n-1)/n+B/n # This overestimates variance
    R = np.sqrt(sigma2_hat/overall_variance)  # Gelman-Rubin statistic
    if verbose: print('Gelman-Rubin statistics:', R)
    return R

### ###

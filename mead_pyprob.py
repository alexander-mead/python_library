import numpy as np
import matplotlib.pyplot as plt

def load_training_log(training_logfile):
    data = np.loadtxt(training_logfile, skiprows=1, delimiter=',', converters={8: lambda _: 0.})
    return data

def plot_learning_curves(data, **kwargs):
    plt.figure(**kwargs)
    plt.plot(data[:, 1], data[:, 3], label='training')
    if data[0, 4] != 0.:
        plt.plot(data[:, 1], data[:, 4], label='validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def assign_trace_positions(traces):
    '''
    To keep track of the order in which traces were generated
    This is useful for tracking positions of samples from multiple MCMC chains
    '''
    for i, trace in enumerate(traces):
        trace.position = i

def join_lists_of_empiricals(list_of_empericals):
    '''
    Take two 'empircal distributions' and join them together.
    # TODO: Delete, probably unnecessary as one line
    '''
    from pyprob.distributions import Empirical
    return Empirical(concat_empiricals=list_of_empericals)

def burn_chains(chains, f=0.5):
    '''
    Burn a certain fraction from the beginning of an MCMC chain
    '''
    new_chains = []
    for chain in chains:
        min_index = int(f*chain.length)
        max_index = chain.length
        chain = chain.thin(max_index-min_index, min_index=min_index, max_index=max_index)
        new_chains.append(chain)
    return new_chains

def Gelman_Rubin_statistics(chains, params, verbose=True):
    '''
    Calculate the Gelman-Rubin statistic
    chains: A list of equal-length MCMC traces
    '''
    # Initial information
    m = len(chains)
    n = len(chains[0])
    d = len(params)
    if verbose:
        print('Number of chains:', m)
        print('Number of parameters:', d)
        print('Chain lengths:', n, '\n')
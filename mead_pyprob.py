# Standard imports
import sys
import os

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# PyProb imports
import pyprob
from pyprob.distributions import Empirical

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

def get_Jeffrys_posterior(model, data, data_error, num_traces_posterior, num_traces_per_go, 
    obs_name='observations', 
    suppress_printing=True, 
    verbose=True
    ):

    num_of_goes = num_traces_posterior//num_traces_per_go
    if verbose:
        print('Number of posterior traces:', num_traces_posterior)
        print('Number of traces per go:', num_traces_per_go)
        print('Number of goes:', num_of_goes)
    observational_posteriors = []; observation_values = []; ESS = []
    for i in range(num_of_goes):
        obs = np.random.normal(loc=data, scale=data_error) # Draw observations from their distribution
        observation_values.extend(obs)
        if i == 1: 
            if verbose: print('First observation:', obs)
            if suppress_printing:
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w') # Suppress printing
        observational_posterior = model.posterior(
            num_traces=num_traces_per_go,
            inference_engine=pyprob.InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK,
            observe={obs_name: obs},
            )
        ESS.append(observational_posterior._effective_sample_size)
        observational_posterior = observational_posterior.resample(num_samples=num_traces_per_go) # NOTE: Must resample here using importance weights
        observational_posteriors.append(observational_posterior) 
    posterior = Empirical(concat_empiricals=observational_posteriors) # Join all posterior distributions together
    if suppress_printing:
        sys.stdout = old_stdout # Re-enable printing
    return posterior
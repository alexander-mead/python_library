import matplotlib.pyplot as plt

def triangle_plots(dicts_of_samples, params, labels,
    truths = None,
    fig_size = 3.,
    hist_bins = 'auto',
    hist_density = True,
    hist_alpha = 0.7,
    scatter_alpha = 0.1,
    scatter_size = 5.,
    ):
    '''
    Makes a triangle plot
    params:
    dicts_of_samples: List of dictionaries of samples (e.g., dict['x'] = [1., 1.1., 1.3, ...])
    params: List of names of parameters to plot (dictionary keys)
    labels: List of axis labels corresponding to parameters
    truths: List of true values of the parameters TODO: Option for None
    '''
    n = len(params)
    plt.subplots(figsize=(n*fig_size, n*fig_size))
    iplot = 0
    for ir, (param_r, label_r) in enumerate(zip(params, labels)):
        for ic, (param_c, label_c) in enumerate(zip(params, labels)):
            iplot += 1
            if ir == ic:
                plt.subplot(n, n, iplot)
                if truths is not None:
                    plt.axvline(truths[ir], color='black', ls='--', alpha=0.7, label='Truth')
                for dict_of_samples in dicts_of_samples:
                    plt.hist(dict_of_samples[param_r], 
                        bins=hist_bins, density=hist_density, alpha=hist_alpha,
                    )
                plt.xlabel(label_r) if ic==n-1 else plt.gca().set_xticklabels([])
                plt.yticks([])
                if iplot == 1: plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
            elif ir > ic:
                plt.subplot(n, n, iplot)
                if truths is not None:
                    plt.plot([truths[ic]], [truths[ir]], color='black', marker='x', alpha=0.7, label='Truth')
                for dict_of_samples in dicts_of_samples:
                    plt.scatter(dict_of_samples[param_c], dict_of_samples[param_r], 
                            alpha=scatter_alpha, s=scatter_size,
                    )
                plt.xlabel(label_c) if ir==n-1 else plt.gca().set_xticklabels([])
                plt.ylabel(label_r) if ic==0 else plt.gca().set_yticklabels([])
    #plt.tight_layout()
    plt.show()
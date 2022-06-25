import matplotlib.pyplot as plt

# def triangle_plots(latents, truths, labels, dict_of_samples,
#     fig_size = 3.,
#     hist_bins = 'auto',
#     hist_density = True,
#     hist_alpha = 0.7,
#     scatter_alpha = 0.1,
#     scatter_size = 5.,
#     ):
#     '''
#     Makes a triangle plot
#     latents: List of 
#     truths: List of true values of the data
#     labels: List of axis labels
#     dict_of_samples: Dictionary
#     '''

#     n = len(latents)
#     if n != len(truths) or n != len(labels):
#         raise ValueError('Error {latents, truths, ids, labels} must all be lists of the same length')
#     plt.subplots(figsize=(n*fig_size, n*fig_size))
#     iplot = 0
#     #nsamples = 2#len(dict_of_samples[latents[0]])
#     nsamples = len(dict_of_samples[latents[0]])
#     for ir, (latent_r, true_r, label_r) in enumerate(zip(latents, truths, labels)):
#         for ic, (latent_c, true_c, label_c) in enumerate(zip(latents, truths, labels)):
#             iplot += 1
#             if ir == ic:
#                 plt.subplot(n, n, iplot)
#                 plt.axvline(true_r, color='black', ls='--', alpha=0.7, label='Truth')
#                 for isample in range(nsamples):
#                     plt.hist(dict_of_samples[latent_r][isample], 
#                         bins=hist_bins, density=hist_density, alpha=hist_alpha,
#                     )
#                 plt.xlabel(label_r) if ic==n-1 else plt.gca().set_xticklabels([])
#                 plt.yticks([])
#                 if iplot == 1: plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
#             elif ir > ic:
#                 plt.subplot(n, n, iplot)
#                 plt.plot([true_c], [true_r], color='black', marker='x', alpha=0.7, label='Truth')
#                 for isample in range(nsamples):
#                     plt.scatter(dict_of_samples[latent_c][isample], dict_of_samples[latent_r][isample], 
#                             alpha=scatter_alpha, s=scatter_size,
#                     )
#                 plt.xlabel(label_c) if ir==n-1 else plt.gca().set_xticklabels([])
#                 plt.ylabel(label_r) if ic==0 else plt.gca().set_yticklabels([])
#     #plt.tight_layout()
#     plt.show()

def triangle_plots(latents, truths, labels, dicts_of_samples,
    fig_size = 3.,
    hist_bins = 'auto',
    hist_density = True,
    hist_alpha = 0.7,
    scatter_alpha = 0.1,
    scatter_size = 5.,
    ):
    '''
    Makes a triangle plot
    latents: List of 
    truths: List of true values of the data
    labels: List of axis labels
    dict_of_samples: Dictionary of samples (e.g., dict['x'] = [1., 1.1., 1.3, ...])
    '''

    n = len(latents)
    if n != len(truths) or n != len(labels):
        raise ValueError('Error {latents, truths, ids, labels} must all be lists of the same length')
    plt.subplots(figsize=(n*fig_size, n*fig_size))
    iplot = 0
    for ir, (latent_r, true_r, label_r) in enumerate(zip(latents, truths, labels)):
        for ic, (latent_c, true_c, label_c) in enumerate(zip(latents, truths, labels)):
            iplot += 1
            if ir == ic:
                plt.subplot(n, n, iplot)
                plt.axvline(true_r, color='black', ls='--', alpha=0.7, label='Truth')
                for dict_of_samples in dicts_of_samples:
                    plt.hist(dict_of_samples[latent_r], 
                        bins=hist_bins, density=hist_density, alpha=hist_alpha,
                    )
                plt.xlabel(label_r) if ic==n-1 else plt.gca().set_xticklabels([])
                plt.yticks([])
                if iplot == 1: plt.legend(loc='upper left', bbox_to_anchor=(1., 1.))
            elif ir > ic:
                plt.subplot(n, n, iplot)
                plt.plot([true_c], [true_r], color='black', marker='x', alpha=0.7, label='Truth')
                for dict_of_samples in dicts_of_samples:
                    plt.scatter(dict_of_samples[latent_c], dict_of_samples[latent_r], 
                            alpha=scatter_alpha, s=scatter_size,
                    )
                plt.xlabel(label_c) if ir==n-1 else plt.gca().set_xticklabels([])
                plt.ylabel(label_r) if ic==0 else plt.gca().set_yticklabels([])
    #plt.tight_layout()
    plt.show()
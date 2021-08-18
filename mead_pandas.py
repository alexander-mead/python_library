import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def data_head(df, comment, verbose=False):
    '''
    Utility function for writing out subset of dataframe with comment
    TODO: This does not seem to be necessary
    '''
    if verbose:
        print(comment)
        print(df.head(15))
        print()

def column_statistics(df, column, condition=None):
    '''
    Computes useful summary statistics of one pandas column (called feature here)
    TODO: How about df.describe() or df.agg()?
    '''
    if condition is None:
        d = df[column]
    else:
        d = df[column].loc[condition]
    print('Feature:', column)
    print('Number:', len(d))
    print('Sum:', d.sum())
    print('Mean:', d.mean())
    print('Median:', d.median())
    print('Standard deviation:', d.std())
    print()

def unique_column_entries(df, column, max=15, normalize=False): # Unique values in columns
    '''
    Write out useful information about the number of unique entries in a column
    '''
    entries = df[column].unique()
    number_of_different_entries = len(entries)
    if number_of_different_entries < max:
        print(column, ':', number_of_different_entries)
        print(df[column].value_counts(dropna=False, normalize=normalize))

def drop_column_name_prefix(df, prefix):
    df.columns = df.columns.str.replace(prefix, '')
    return df

def _add_jitter(values, std):
    # Add jitter to values
    # TODO: Set a seed
    return values+np.random.normal(0., std, values.shape)

def _calculate_minimum_difference(series):
    # Calculate the minimum of the differences between values in a column
    arr = series.to_numpy()
    b = np.diff(np.sort(arr))
    return b[b>0].min()

def _jitter_data(df, feature_row, feature_col, fsig):
    # Apply a jitter
    if fsig == 0.:
        data_row = df[feature_row]
        data_col = df[feature_col]
    else:
        std_row = fsig*_calculate_minimum_difference(df[feature_row])
        std_col = fsig*_calculate_minimum_difference(df[feature_col])
        data_row = _add_jitter(df[feature_row], std_row)
        data_col = _add_jitter(df[feature_col], std_col)
    return (data_row, data_col)

def feature_triangle(df, label, features, continuous_label=False, 
                                          kde=True, 
                                          alpha=1., 
                                          figsize=(10,10), 
                                          jitter=0.
                                        ):
    '''
    Triangle plot of list of features split by some characteristic. Histogram distributions along diagonal, correlations off diagonal.
    df - pandas data frame
    label - string, name of one column, usually the (discrete) label you are interested in predicting (e.g., species) 
    features - list of strings corresponding to feature columns (e.g., petal length, petal width)
    TODO: Seed random numbers so they are the same for each jitter
    '''

    # Initialise the plot
    sns.set_theme(style='ticks')
    n = len(features)

    # Figure out the hue of bars/points
    if label is None:
        hue_hist = None
        hue_scat = None
    else:
        if continuous_label:
            hue_hist = None
        else:
            hue_hist = df[label]
        hue_scat = df[label]

    # Make the big plot
    _, axs = plt.subplots(n, n, figsize=figsize)
    i = 0
    for irow in range(n):
        for icol in range(n):
            i += 1 # Add one to the plot number
            plt.subplot(n, n, i)
            feature_row = features[irow]; feature_col = features[icol]
            if icol > irow:
                axs[irow, icol].axis('off') # Ignore upper triangle
                continue
            if (icol == irow):
                sns.histplot(df, x=feature_col, 
                                 hue=hue_hist, 
                                 stat='density', 
                                 bins='auto', 
                                 legend=(not continuous_label and (icol==0)), 
                                 kde=kde,
                                )
            else:
                data_row, data_col = _jitter_data(df, feature_row, feature_col, jitter)
                sns.scatterplot(x=data_col, y=data_row,
                                hue=hue_scat,
                                alpha=alpha,
                                legend=None,
                               )
            # x-axis labels
            if irow == n-1: 
                plt.xlabel(feature_col)
            else:
                plt.xlabel(None)
                plt.tick_params(axis='x', which='major', bottom=True, labelbottom=False)

            # y-axis labels
            if (icol == irow):
                plt.ylabel(None)
                plt.tick_params(axis='y', which='major', left=False, labelleft=False)
            elif (icol == 0):
                plt.ylabel(feature_row)
            else:
                plt.ylabel(None)
                plt.tick_params(axis='y', which='major', left=True, labelleft=False)

    plt.tight_layout()
    return plt

def feature_scatter_triangle(df, label, features, continuous_label=False, 
                                                    alpha=1., 
                                                    figsize=(10,10), 
                                                    jitter=0.
                                                    ):
    '''
    Triangle plot of list of features split by some characteristic. Histogram distributions along diagonal, correlations off diagonal.
    df - pandas data frame
    label - string, name of one column, usually the (discrete) label you are interested in predicting (e.g., species) 
    features - list of strings corresponding to feature columns (e.g., petal length, petal width)
    '''

    # Initialise the plot
    sns.set_theme(style='ticks')
    n = len(features)-1

    # Figure out the hue of bars/points
    if label is None:
        hue_scat = None
    else:
        hue_scat = df[label]

    # Make the big plot
    _, axs = plt.subplots(n, n, figsize=figsize)
    i = 0
    for irow in range(n):
        for icol in range(n):
            i += 1 # Add one to the plot number
            if icol > irow:
                axs[irow, icol].axis('off') # Ignore upper triangle
                continue
            plt.subplot(n, n, i)
            feature_col = features[icol]; feature_row = features[irow+1]
            data_row, data_col = _jitter_data(df, feature_row, feature_col, jitter)
            sns.scatterplot(x=data_col, y=data_row,
                            hue=hue_scat,
                            alpha=alpha,
                            legend=(not continuous_label and (icol==0)),
                            )

            # x-axis labels
            if (irow == n-1): 
                plt.xlabel(feature_col)
            else:
                plt.xlabel(None)
                plt.tick_params(axis='x', which='major', bottom=True, labelbottom=False)

            # y-axis labels
            if (icol == 0):
                plt.ylabel(feature_row)
            else:
                plt.ylabel(None)
                plt.tick_params(axis='y', which='major', left=True, labelleft=False)

    plt.tight_layout()
    return plt

# Correlation matrix
def correlation_matrix(df, columns, figsize=(7,7), mask_diagonal=True, mask_upper_triangle=True):
    '''
    Create a plot of the correlation matrix for (continous) columns (features) of dataframe (df)
    '''
    # Calculate correlation coefficients
    corr = df[columns].corr() 
    if mask_diagonal and mask_upper_triangle:
        corr.drop(labels=columns[0], axis=0, inplace=True)  # Remove first row
        corr.drop(labels=columns[-1], axis=1, inplace=True) # Remove last column

    # Create mask
    mask = np.zeros_like(corr, dtype=bool) 
    if mask_upper_triangle and mask_diagonal:
        mask[np.triu_indices_from(mask, k=1)] = True # k=1 does diagonal offset from centre
    elif mask_upper_triangle:
        mask[np.triu_indices_from(mask, k=1)] = True
    elif mask_diagonal:
        mask[np.diag_indices_from(mask)] = True

    # Make the plot
    plt.style.use('seaborn-white') 
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(corr, vmin=-1., vmax=1., cmap=cmap, mask=mask, linewidths=.5,
                    annot=True,
                    square=True,
                    cbar=False,
                )
    g.set_yticklabels(labels=g.get_yticklabels(), va='center') # Centre y-axis ticks
    #return plt # TODO: Does this need to return anything?

def swarmplot(df, columns, id_=None, hue=None, order=None, 
        hue_order=None, dodge=False, orient=None, color=None, palette=None, 
        size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs):

    # Make the id_vars column correctly if there are Nones
    # TODO: There must be a one-line solution
    if id_ is None and hue is None:
        id_vars = None
    elif hue is None:
        id_vars = [id_]
    elif id_ is None:
        id_vars = [hue]
    else:
        id_vars = [id_, hue]

    # Make the melted data frame columns correctly if there are Nones
    # TODO: There must be a one-line solution
    if id_vars is None:
        new_columns = columns
    else:
        new_columns = columns+id_vars

    # Make the simple data frame, then melt it from wide form to long form, then make swarmplot
    simple_df = df[new_columns]
    df_melted = pd.melt(simple_df, id_vars=id_vars, value_vars=None, var_name='var', value_name='value')
    sns.swarmplot(data=df_melted, x='var', y='value', hue=hue, order=order, 
        hue_order=hue_order, dodge=dodge, orient=orient, color=color, palette=palette, 
        size=size, edgecolor=edgecolor, linewidth=linewidth, ax=ax, **kwargs)

def lineswarm(ax, data, columns, line_alpha=0.1):

    # TODO: Add all variables
    #sns.swarmplot(data=data, x='variable', y='value', ax=ax)
    swarmplot(data, columns)
    n = len(columns)

    # Now connect the dots
    locs = []
    for idx in range(n):
        locs.append(ax.get_children()[idx].get_offsets())

    for j in range(n-1):
        for i in range(locs[0].shape[0]):
            x = [locs[j][i, 0], locs[j+1][i, 0]]
            y = [locs[j][i, 1], locs[j+1][i, 1]]
            ax.plot(x, y, color='black', alpha=line_alpha)


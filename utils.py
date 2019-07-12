from string import ascii_lowercase

import numpy as np
from numpy import in1d

import matplotlib.pyplot as plt
from matplotlib import rcParams

def apply_sublabels(axs, x=-50, y=0, size='medium', weight='bold', ha='right', va='top', prefix='', postfix='', invert_color_inds=[], bg=None):
    '''
    Applys panel labels (a, b, c, ... ) in order to the axis handles stored in the list axs
    
    Most of the function arguments should be self-explanatory
    
    invert_color_inds, specifies which labels should use white text, which is useful for darker pcolor plots
    '''
    
    if bg is not None:
        bbox_props = dict(boxstyle="round,pad=0.1", fc=bg, ec="none", alpha=0.9)
    else:
        bbox_props = None
    
    # If using latex we need to manually insert the \textbf command
    if rcParams['text.usetex'] and weight == 'bold':
        prefix  = '\\textbf{' + prefix
        postfix = postfix + '}'
    
    for n, ax in enumerate(axs):
        if in1d(n, invert_color_inds):
            color='w'
        else:
            color='k'
        
        ax.annotate(prefix + ascii_lowercase[n] + postfix,
                    xy=(0, 1),
                    xytext=(x, y),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size=size,
                    color=color,
                    weight=weight,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    bbox=bbox_props)

"""
Plotting utilities
"""


def initialize(plt):
    """
    Initialize the plot configuration
    """

    plt.rc('font', **{'family'     : 'sans-serif',
                  'sans-serif' : ['Helvetica'],
                  'size': 20
                  })

    plt.rc('figure', **{'autolayout' : True,
                    'figsize'    : (14, 12)
                    })

    plt.rc('xtick', labelsize='medium', direction='in', top=True)
    plt.rc('ytick', labelsize='medium', direction='in', right=True)
    plt.rc('axes',  labelsize='medium', grid=True)
    plt.rc('axes',  titlesize='medium')
    plt.rc('legend', fontsize='medium')
    plt.rc('image',  cmap='gray')


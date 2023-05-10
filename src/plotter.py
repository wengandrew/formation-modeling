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

    plt.rc('xtick', labelsize='medium',
                    direction='in',
                    top=True)

    plt.rc('xtick.major', width=1.0)
    plt.rc('xtick.minor', visible=True)

    plt.rc('ytick', labelsize='medium',
                    direction='in',
                    right=True)
    plt.rc('ytick.major', width=1.0)
    plt.rc('ytick.minor', visible=True)

    plt.rc('axes',  labelsize='medium',
                    grid=True,
                    linewidth=1.0)

    plt.rc('legend', fontsize='medium',
                     frameon=False)

    plt.rc('image',  cmap='gray')


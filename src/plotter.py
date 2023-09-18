"""
Plotting utilities
"""


def initialize(plt, style='default'):
    """
    Initialize the plot configuration

    Style: 'default', 'ieee'
    """

    if style == 'default':
        font = 'Helvetica'
    elif style == 'ieee':
        font = 'Times New Roman'

    plt.rc('font', **{'family'     : 'serif',
                  'serif' : [font],
                  'size': 20
                  })

    # Latex font formatting (dejavusans, dejavuserif, cm, stix, stixsans)
    plt.rcParams['mathtext.fontset'] = 'cm'

    plt.rc('figure', **{'autolayout' : True,
                    'figsize'    : (14, 12)
                    })

    plt.rc('lines', linewidth=2)

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


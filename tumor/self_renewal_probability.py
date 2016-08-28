""" Functions needed to calculate the probability that a cycling cell divides in the next unit of time or becomes quiescent
"""

import numpy as np
from scipy.spatial.distance import cdist


def number_cells_within_range(query_cell, cells, signaling_range):
    """
    Computes the number of cells that lie within a certain distance of a given query cell.
    Biologically, this is implemented by feedback signals (e.g. biomolecules that are secreted and later absorbed).
    """

    if len(cells) == 0:
        return 0
    else:
        query_cell_position = np.array([query_cell.position])
        cell_positions = np.array([cell.position for cell in cells])
        query_cell_distances = cdist(query_cell_position, cell_positions).ravel()
        return len(query_cell_distances[query_cell_distances < signaling_range])


def W_negative_base(NC, NQ, parameterValues):
    """
    A negative feedback function of the number of cycling and quiescent cells in the neighborhood of a given cell
    """

    negative_gain_C = 0.0
    negative_n_C = 1  # should be non-zero
    negative_n_Q = 2  # should be non-zero

    ss = np.power(negative_gain_C*NC, negative_n_C) + np.power(parameterValues['negative_gain_Q']*NQ, negative_n_Q)

    return 1.0/(1.0 + ss)


def W_negative(random_C_cell, C_cells, Q_cells, parameterValues):
    """
    Level of negative feedback a random cycling cell receives
    """

    NC = number_cells_within_range(random_C_cell, C_cells, parameterValues['signaling_range_negative'])
    NQ = number_cells_within_range(random_C_cell, Q_cells, parameterValues['signaling_range_negative'])

    return W_negative_base(NC, NQ, parameterValues)


def W_positive_base(N_C_wt, N_C_mut, N_Q_wt, N_Q_mut, parameterValues):
    """
    A positive feedback function of the number of various types of cells in the neighborhood of a given cell
    """

    positive_gain_C_wt = 0
    positive_gain_C_mut = 0
    positive_n_C = 1  # should be non-zero

    positive_n_Q = 2  # should be non-zero

    ss = (np.power(positive_gain_C_wt*N_C_wt + positive_gain_C_mut*N_C_mut, positive_n_C) +
          np.power(parameterValues['positive_gain_Q_wt']*N_Q_wt + parameterValues['positive_gain_Q_mut']*N_Q_mut, positive_n_Q))

    return ss/(1.0 + ss)


def W_positive(random_C_cell, C_cells, Q_cells, parameterValues):
    """
    Level of positive feedback a random cycling cell receives
    """

    signaling_range_positive = 4

    C_wildtype_cells = [cell for cell in C_cells if cell.heritable_state == 'wild-type']
    C_mutant_cells = [cell for cell in C_cells if cell.heritable_state == 'mutant']
    Q_wildtype_cells = [cell for cell in Q_cells if cell.heritable_state == 'wild-type']
    Q_mutant_cells = [cell for cell in Q_cells if cell.heritable_state == 'mutant']

    N_C_wt = number_cells_within_range(random_C_cell, C_wildtype_cells, signaling_range_positive)
    N_Cstar = number_cells_within_range(random_C_cell, C_mutant_cells, signaling_range_positive)
    N_Q_wt = number_cells_within_range(random_C_cell, Q_wildtype_cells, signaling_range_positive)
    N_Q_mut = number_cells_within_range(random_C_cell, Q_mutant_cells, signaling_range_positive)

    return W_positive_base(N_C_wt, N_Cstar, N_Q_wt, N_Q_mut, parameterValues)


def self_renewal_probability(random_C_cell, C_cells, Q_cells, parameterValues):
    """
    Probability that a random cycling cell divides after one unit of time versus becoming quiescent
    """

    return (parameterValues['self_renewal_probability_max'] *
            W_negative(random_C_cell, C_cells, Q_cells, parameterValues) *
            W_positive(random_C_cell, C_cells, Q_cells, parameterValues))


def plot_self_renewal_probability():
    """
    Visualize how the self-renewal probability of a given (cycling) cell depends upon the number of cycling and quiescent cells in its vicinity
    """

    def self_renewal_probability(x, y):
        from read import read_into_dict
        parameterValues = read_into_dict('parameterValues.in')
        self_renewal_probability_max = parameterValues['self_renewal_probability_max']
        return self_renewal_probability_max * W_positive_base(0, 0, x, y, parameterValues) * W_negative_base(0, x+y, parameterValues)

    def print_self_renewal_probabilities(x, y):
        print 'number of wild-type quiescent cells = ' + str(x)
        print 'number of mutant quiescent cells = ' + str(y)
        print 'self-renewal probability = ' + str(self_renewal_probability(x, y))

    # how much does the self-renewal probability change if we replace one wild-type Q-cell with a mutant Q-cell?
    print_self_renewal_probabilities(2, 0)
    print_self_renewal_probabilities(1, 1)
    print
    print_self_renewal_probabilities(1, 0)
    print_self_renewal_probabilities(0, 1)

    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import pyplot as plt

    fig = plt.figure(facecolor='white')
    ax = fig.gca(projection='3d', zlim=(0, 1))

    xx = np.linspace(0, 4)
    yy = np.linspace(0, 4)
    XX, YY = np.meshgrid(xx, yy)
    # noinspection PyUnresolvedReferences
    surf = ax.plot_surface(XX, YY, self_renewal_probability(XX, YY), rstride=2, cstride=2, cmap=cm.RdPu, linewidth=1, antialiased=True)

    ax.zaxis.set_major_locator(LinearLocator(6))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title('self renewal probability\n' + '(assumes equal +ve and -ve feedback range)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('number of local quiescent WT cells')
    ax.set_ylabel('number of local quiescent MUTANT cells')
    ax.view_init(elev=30, azim=-60)
    ax.dist = 10
    plt.show()


if __name__ == '__main__':

    plot_self_renewal_probability()

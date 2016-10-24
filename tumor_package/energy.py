""" Functions to compute the total energy of a tumor """

import numpy as np
from scipy.spatial.distance import pdist


def sum_cell_cell_energies(cell_cell_distances):
    """Given a list of cell-cell distances, compute the corresponding

    cell-cell interaction energies and sum those cell-cell interaction
    energies.  Cell-cell interaction energy is strongly repulsive at
    short distances, weakly attractive at intermediate distances, and
    zero at long distances
    """

    Va = 120.
    Vc = 12.
    aa = 1
    bb = 1.2
    cc = 2
    nn = 2

    cc_dist1 = cell_cell_distances[(cell_cell_distances < aa)]
    energy1 = Va * np.sum(np.power(aa / cc_dist1, nn))
    energy1 -= cc_dist1.size * Vc

    cc_dist2 = cell_cell_distances[(~(cell_cell_distances < aa)) &
                                   (cell_cell_distances < bb)]
    energy2 = -cc_dist2.size * Vc

    cc_dist3 = cell_cell_distances[(~(cell_cell_distances < bb)) &
                                   (cell_cell_distances < cc)]

    energy3 = Vc * np.sum(cc_dist3 - bb) / (cc - bb)
    energy3 -= cc_dist3.size * Vc

    return energy1 + energy2 + energy3


def calculate_tumor_energy(cells):
    """Brute-force computation of tumor energy by summing the cell-cell

    interaction energies of all unique cell pairs
    """

    if len(cells) == 0:
        print('this function should not be called with no cells')
        exit(1)
    elif len(cells) == 1:
        return 'undefined'
    else:
        cell_positions = np.array([cell.position for cell in cells])
        # much faster than looping over all unique cell pairs
        cell_cell_distances = pdist(cell_positions)
        return sum_cell_cell_energies(cell_cell_distances)

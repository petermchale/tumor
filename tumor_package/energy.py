""" Functions to compute the total energy of a tumor """

import numpy as np
from scipy.spatial.distance import pdist
import numexpr as ne


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

    cc_dist1 = cell_cell_distances[ne.evaluate('cell_cell_distances < aa')]
    # putting sum outside ne.evaluate (instead of inside it) handles the case when cc_dist1 is empty
    energy1 = np.sum(ne.evaluate('Va * (aa / cc_dist1)**nn'))
    cc_dist1_size = cc_dist1.size
    energy1 = ne.evaluate('energy1 - cc_dist1_size * Vc')

    cc_dist2 = cell_cell_distances[
        ne.evaluate('(~(cell_cell_distances < aa)) & (cell_cell_distances < bb)')]
    cc_dist2_size = cc_dist2.size
    energy2 = ne.evaluate('-cc_dist2_size * Vc')

    cc_dist3 = cell_cell_distances[
        ne.evaluate('(~(cell_cell_distances < bb)) & (cell_cell_distances < cc)')]
    # putting sum outside ne.evaluate handles the case when cc_dist3 is empty
    energy3 = np.sum(ne.evaluate('Vc * (cc_dist3 - bb) / (cc - bb)'))
    cc_dist3_size = cc_dist3.size
    energy3 = ne.evaluate('energy3 - cc_dist3_size * Vc')

    return ne.evaluate('energy1 + energy2 + energy3')


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

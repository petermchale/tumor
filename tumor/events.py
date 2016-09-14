""" These functions are needed to update a tumor object at every step of the stochastic process """


import copy

import numpy as np
from scipy.spatial.distance import cdist

from self_renewal_probability import self_renewal_probability
from energy import calculate_tumor_energy, sum_cell_cell_energies


def time_until_next_event(prng, net_rate):
    """
    Compute the time at which the next event is due to occur.
    All events are modeled as Poisson processes with exponentially distributed waiting times.
    Units of time are chosen such that the average time it takes for a given cell to divide is 1.
    """

    return prng.standard_exponential() / net_rate


def execute_divisionQuiescence_event(prng, cells, old_tumor_energy, parameterValues):
    """
    Randomly choose between dividing a random cell versus making it quiescent.
    This function modifies the list of cells in place, but updates the tumor energy using a functional programming style,
    i.e. the tumor energy is passed in, updated, and then passed out using a return statement.
    """

    # when parent is cloned (cell division), displace parent and daughter this distance from one another
    distance_between_parent_and_daughter = 1.5

    C_cells = [cell for cell in cells if cell.cell_cycle_state == 'cycling']
    random_C_cell = prng.choice(C_cells)

    Q_cells = [cell for cell in cells if cell.cell_cycle_state == 'quiescent']

    if prng.uniform(0, 1) < self_renewal_probability(random_C_cell, C_cells, Q_cells, parameterValues):
        original_parent_position = random_C_cell.position

        # randomly displace parent cell
        theta_parent = prng.uniform(0, 2*np.pi)
        random_C_cell.position = np.add(original_parent_position,
                                        0.5 * distance_between_parent_and_daughter * np.array([np.cos(theta_parent), np.sin(theta_parent)]))

        # clone parent cell to generate a daughter cell
        daughter_C_cell = copy.deepcopy(random_C_cell)

        # randomly displace daughter cell
        theta_daughter = theta_parent + np.pi
        daughter_C_cell.position = np.add(original_parent_position,
                                          0.5 * distance_between_parent_and_daughter * np.array([np.cos(theta_daughter), np.sin(theta_daughter)]))

        daughter_C_cell.ID = max([cell.ID for cell in cells]) + 1
        cells.append(daughter_C_cell)

        def update_tumor_energy():
            """
            This updates only the cell-cell interaction energies that actually changed, which scales linearly with the number of cells.
            That is, it does not wastefully re-compute *all* cell-cell interaction energies,
            which would scale quadratically with the number of cells.
            """

            parent_position = np.array([original_parent_position])
            daughter1_position = np.array([random_C_cell.position])
            daughter2_position = np.array([daughter_C_cell.position])

            # use 'is not' not '!=' to check object identity:
            other_cells = [cell for cell in cells if (cell is not random_C_cell) and (cell is not daughter_C_cell)]
            other_positions = np.array([cell.position for cell in other_cells])

            parent_other_distances = cdist(parent_position, other_positions).ravel()
            parent_other_energy = sum_cell_cell_energies(parent_other_distances)

            daughter1_other_distances = cdist(daughter1_position, other_positions).ravel()
            daughter1_other_energy = sum_cell_cell_energies(daughter1_other_distances)

            daughter2_other_distances = cdist(daughter2_position, other_positions).ravel()
            daughter2_other_energy = sum_cell_cell_energies(daughter2_other_distances)

            daughter1_daughter2_distance = cdist(daughter1_position, daughter2_position).ravel()
            daughter1_daughter2_energy = sum_cell_cell_energies(daughter1_daughter2_distance)

            return (old_tumor_energy - parent_other_energy + daughter1_other_energy
                    + daughter2_other_energy + daughter1_daughter2_energy)

        if len(cells) > 2:
            return update_tumor_energy()
        elif len(cells) == 2:
            return calculate_tumor_energy(cells)
        else:
            print('cell division event cannot produce 1 or 0 cells')
            exit()

    else:
        random_C_cell.cell_cycle_state = 'quiescent'
        return old_tumor_energy


def execute_reactivation_event(prng, cells):
    """
    Randomly choose between reactivating a random quiescent cell (i.e. make it a cycling cell) versus leaving it alone.
    This function modifies the list of cells in place.
    """

    Q_cells = [cell for cell in cells if cell.cell_cycle_state == 'quiescent']
    random_Q_cell = prng.choice(Q_cells)

    # mu = calculate_mu(random_Q_cell, C_cells, Q_cells)
    mu = 0

    if prng.uniform(0, 1) < float(mu):
        random_Q_cell.cell_cycle_state = 'cycling'


def execute_relaxation_event(prng, cells, old_tumor_energy, randomDisplacement, randomIndex):
    """
    Randomly choose a cell, propose to randomly displace it, and accept the proposed move with a
    probability that depends upon how much and direction in which the tumor energy is changed (Metropolis algorithm).
    This function modifies the list of cells in place, but updates the tumor energy using a functional programming style,
    i.e. the tumor energy is passed in, updated, and then passed out using a return statement.
    """

    try:
        old_cell = cells[randomIndex.next(len(cells))]
    except IndexError:
        old_cell = prng.choice(cells)
    old_position_array = np.array([old_cell.position])

    try:
        random_displacement = randomDisplacement.next()
    except IndexError:
        random_displacement = randomDisplacement.new(prng)
    new_position = np.add(old_cell.position, random_displacement)
    new_position_array = np.array([new_position])

    def update_tumor_energy():
        """
        This updates only the cell-cell interaction energies that actually changed, which scales linearly with the number of cells.
        That is, it does not wastefully re-compute *all* cell-cell interaction energies, which would scale quadratically with the number of cells.
        """

        other_cells = [cell for cell in cells if cell is not old_cell]
        other_positions_array = np.array([cell.position for cell in other_cells])

        old_other_distances = cdist(old_position_array, other_positions_array).ravel()
        old_other_energy = sum_cell_cell_energies(old_other_distances)

        new_other_distances = cdist(new_position_array, other_positions_array).ravel()
        new_other_energy = sum_cell_cell_energies(new_other_distances)

        return old_tumor_energy - old_other_energy + new_other_energy

    new_tumor_energy = update_tumor_energy()
    delta_energy = new_tumor_energy - old_tumor_energy

    def accept_move():
        old_cell.position = new_position
        return new_tumor_energy

    def reject_move():
        return old_tumor_energy

    epsilon = 1e-8  # should be small compared to energy scales but large compared to round-off error
    if abs(delta_energy) < epsilon:
        return accept_move()
    elif delta_energy < -epsilon:
        return accept_move()
    else:
        if prng.uniform(0, 1) < np.exp(-delta_energy):
            return accept_move()
        else:
            return reject_move()


def execute_death_event(prng, cells, old_tumor_energy):
    """
    Randomly choose a cell and remove it from the population
    """

    dead_cell = prng.choice(cells)
    dead_position = np.array([dead_cell.position])
    cells.remove(dead_cell)

    def update_tumor_energy():
        """
        This updates only the cell-cell interaction energies that actually changed, which scales linearly with the number of cells.
        That is, it does not wastefully re-compute *all* cell-cell interaction energies, which would scale quadratically with the number of cells.
        """

        other_positions = np.array([cell.position for cell in cells])

        dead_other_distances = cdist(dead_position, other_positions).ravel()
        dead_other_energy = sum_cell_cell_energies(dead_other_distances)

        return old_tumor_energy - dead_other_energy

    if len(cells) > 1:
        return update_tumor_energy()
    else:
        return 'undefined'


""" Classes that define a simulated tumor. """

import numpy as np

from energy import calculate_tumor_energy
import events


class Cell(object):
    """
    Represents a cell in a tumor
    """

    def __init__(self, position, ID, cell_cycle_state, heritable_state):
        self.position = np.array(position)
        self.ID = ID  # unique to each cell
        self.cell_cycle_state = cell_cycle_state  # cycling or quiescent
        self.heritable_state = heritable_state  # wild-type or mutant

    def __str__(self):
        x, y = self.position
        return '{:<5d}'.format(self.ID) + '{:<15}'.format(self.cell_cycle_state) + '{:<10.4f}'.format(x) + \
               '{:<10.4f}'.format(y) + '{:<15}'.format(self.heritable_state)


class Time_Course(object):
    """
    Contains a uniform grid of time points and the states of the tumor at those time points
    """

    def __init__(self, number_C_cells, number_Q_cells, number_of_states):

        self.number_of_states = number_of_states

        time_point_spacing = 0.02
        self.time_points = [ii * time_point_spacing for ii in xrange(self.number_of_states)]
        self.time_points = np.array(self.time_points, dtype=float)

        self.number_C_cells = -np.ones(self.number_of_states, dtype=int)
        self.number_C_cells[0] = number_C_cells

        self.number_Q_cells = -np.ones(self.number_of_states, dtype=int)
        self.number_Q_cells[0] = number_Q_cells

        self.number_of_initialized_states = 1

    def next_time(self):

        return self.time_points[self.number_of_initialized_states]

    def update(self, new_time, old_number_C_cells, old_number_Q_cells):
        """initialize all uninitialized states at time points before given time"""

        while self.number_of_initialized_states < self.number_of_states:
            if new_time > self.time_points[self.number_of_initialized_states]:
                self.number_C_cells[self.number_of_initialized_states] = old_number_C_cells
                self.number_Q_cells[self.number_of_initialized_states] = old_number_Q_cells
                self.number_of_initialized_states += 1
            else:
                break

    def pad_with_extinct_states(self):

        self.number_C_cells[self.number_of_initialized_states:] = np.zeros(self.number_of_states - self.number_of_initialized_states, dtype=int)
        self.number_Q_cells[self.number_of_initialized_states:] = np.zeros(self.number_of_states - self.number_of_initialized_states, dtype=int)

    def end_time_point(self):

        return self.time_points[-1]


class Random_Index(object):
    """
    Generate a large number of integers to index into a cell list of unknown length.
    Faster to generate these random numbers in advance than on the fly.
    """

    def __init__(self, prng, relaxation_rate_per_cell):

        self._min_number_of_cells = 1
        self._max_number_of_cells = 300

        # the following number should be much larger than the relaxation rate per cell, nu:
        # the average time until a population of N cells produces one additional cell is 1/N
        # the average number of relaxation events a single cell undergoes during this time interval is nu*(1/N)
        # therefore, the average number of times that a relaxation event occurs in a population of N cells is:
        # N*nu*(1/N) = nu
        # This assumes no death and no negative feedback,
        # both of which act to keep population at size N for longer than 1/N.
        self._number_indices = int(100 * relaxation_rate_per_cell)

        self._randomNumbers = np.zeros((self._max_number_of_cells, self._number_indices), dtype=np.int)
        self._index = np.zeros(self._max_number_of_cells, dtype=np.int)

        # noinspection PyArgumentList
        for number_of_cells in xrange(self._min_number_of_cells, self._max_number_of_cells):
            self._randomNumbers[number_of_cells] = prng.randint(0, number_of_cells, self._number_indices)

    def next(self, number_of_cells):
        """Fetch the next pre-computed random index that lies between 0 (inclusive) and number_of_cells (exclusive)"""

        ii = self._index[number_of_cells]
        self._index[number_of_cells] += 1
        return self._randomNumbers[number_of_cells, ii]


class Random_Displacement(object):
    """
    Generate a large number of random cell displacements.
    Faster to generate these random numbers in advance than on the fly.
    """

    def __init__(self, prng, sample_size):

        self._mean = np.array([0, 0], dtype=float)
        sigma = 0.02
        self._covariance = sigma*np.identity(2)
        self._randomNumbers = prng.multivariate_normal(self._mean, self._covariance, int(sample_size))
        self._index = 0

    def next(self):
        """
        Fetch next pre-computed random displacement
        """

        ii = self._index
        self._index += 1
        return self._randomNumbers[ii]

    def new(self, prng):
        """
        Generate a random displacement de novo
        """

        return prng.multivariate_normal(self._mean, self._covariance)


def unpack(positions):
    """
    Take a (n X 2) numpy array of n cell positions in 2D and return two (n X 1) numpy arrays representing the x and y coordinates
    """

    try:
        x, y = zip(*positions)  # http://hangar.runway7.net/python/packing-unpacking-arguments
        return x, y
    except ValueError:
        if not positions:  # positions is empty
            return [], []


class Extinction(Exception):
    """Raise when there are no more cells left"""


class Tumor(object):
    """
    Represents the state of a collection of cells (a tumor).
    Each cell can switch between a cycling (dividing) and quiescent (non-dividing) state and die (be removed from the simulation).
    Space is explicitly modeled by assigning an energy to each cell pair,
    such that it is energetically unfavorable for cells to be within one cell diameter of one another,
    cells at intermediate distances are attracted to one another, and cells far from one another exert no effect on each other.
    """

    def __init__(self, initialCondition, parameterValues, random_seed=None):

        self._prng = np.random.RandomState(random_seed)
        self.time_elapsed = 0.0
        self.number_of_steps = 0

        self.cells = []
        ID = 0

        if ('C_cells_x' in initialCondition) and ('C_cells_y' in initialCondition):
            for position in zip(initialCondition['C_cells_x'], initialCondition['C_cells_y']):
                self.cells.append(Cell(position, ID, 'cycling', 'wild-type'))
                ID += 1

        if ('C*_cells_x' in initialCondition) and ('C*_cells_y' in initialCondition):
            for position in zip(initialCondition['C*_cells_x'], initialCondition['C*_cells_y']):
                self.cells.append(Cell(position, ID, 'cycling', 'mutant'))
                ID += 1

        if ('Q_cells_x' in initialCondition) and ('Q_cells_y' in initialCondition):
            for position in zip(initialCondition['Q_cells_x'], initialCondition['Q_cells_y']):
                self.cells.append(Cell(position, ID, 'quiescent', 'wild-type'))
                ID += 1

        if ('Q*_cells_x' in initialCondition) and ('Q*_cells_y' in initialCondition):
            for position in zip(initialCondition['Q*_cells_x'], initialCondition['Q*_cells_y']):
                self.cells.append(Cell(position, ID, 'quiescent', 'mutant'))
                ID += 1

        self.energy = calculate_tumor_energy(self.cells)

        self._randomDisplacement = Random_Displacement(self._prng, 1e7)
        self.relaxation_rate_per_cell = 500
        self._randomIndex = Random_Index(self._prng, self.relaxation_rate_per_cell)

        self.time_course = Time_Course(self.number_of_C_Cells(), self.number_of_Q_Cells(), int(parameterValues['number_of_states']))

        self._parameterValues = parameterValues

    def _step(self):
        """
         Execute a single step of the stochastic process
        """

        if len(self.cells) == 0:
            raise Extinction('time = ' + str(self.time_elapsed) + '; no more cells left')

        self._old_number_C_cells = self.number_of_C_Cells()
        self._old_number_Q_cells = self.number_of_Q_Cells()

        death_rate_per_cell = 0.1

        NN = self._old_number_C_cells + self._old_number_Q_cells
        divisionQuiescence_rate = float(self._old_number_C_cells)
        reactivation_rate = float(self._old_number_Q_cells)
        relaxation_rate = float(NN) * self.relaxation_rate_per_cell
        death_rate = float(NN) * death_rate_per_cell
        net_rate = divisionQuiescence_rate + reactivation_rate + relaxation_rate + death_rate

        uniform_random_number = self._prng.uniform(0, 1)
        if uniform_random_number < divisionQuiescence_rate/net_rate:
            self.energy = events.execute_divisionQuiescence_event(self._prng, self.cells, self.energy, self._parameterValues)
        elif uniform_random_number < (divisionQuiescence_rate + reactivation_rate)/net_rate:
            events.execute_reactivation_event(self._prng, self.cells)
        elif uniform_random_number < (divisionQuiescence_rate + reactivation_rate + relaxation_rate)/net_rate:
            if len(self.cells) > 1:
                self.energy = events.execute_relaxation_event(self._prng, self.cells, self.energy, self._randomDisplacement, self._randomIndex)
        else:
            self.energy = events.execute_death_event(self._prng, self.cells, self.energy)

        self.time_elapsed += events.time_until_next_event(self._prng, net_rate)
        self.number_of_steps += 1

    def next_frame(self):
        """
         Evolve the system until it has passed the next 'grid time point'
        """

        while self.time_elapsed < self.time_course.next_time():
            self._step()
        self.time_course.update(self.time_elapsed, self._old_number_C_cells, self._old_number_Q_cells)

    def generate_time_course(self):
        """
         Sample the stochastic growth of the tumor at evenly spaced time points.
         Alternatively, one could generate a time course by recording each change of tumor state,
         e.g. recording the time and new state whenever the cycling or quiescent cell number changes
        """

        while self.time_elapsed < self.time_course.end_time_point():
            try:
                self.next_frame()
            except Extinction as exception:  # http://www.scipy-lectures.org/intro/language/exceptions.html
                print exception.message
                self.time_course.pad_with_extinct_states()
                break

        return self.time_course

    def C_cell_positions(self):

        return unpack([cell.position for cell in self.cells if cell.cell_cycle_state == 'cycling'])

    def Q_cell_positions(self):

        return unpack([cell.position for cell in self.cells if cell.cell_cycle_state == 'quiescent'])

    def mutant_cell_positions(self):

        return unpack([cell.position for cell in self.cells if cell.heritable_state == 'mutant'])

    def number_of_C_Cells(self):

        return len([cell for cell in self.cells if cell.cell_cycle_state == 'cycling'])

    def number_of_Q_Cells(self):

        return len([cell for cell in self.cells if cell.cell_cycle_state == 'quiescent'])

    def cell_pair_energy(self):

        nn = len(self.cells)
        number_cell_pairs = 0.5*nn*(nn-1)
        if number_cell_pairs > 0:
            return self.energy/float(number_cell_pairs)
        else:
            return 'undefined'

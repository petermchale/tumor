""" Simulate the stochastic evolution of a tumor """

import numpy as np

from read import read_into_dict
from tumor import Tumor


def generate_tumor_growth_trajectories_base(initialCondition, parameterValues, number_realizations, random_seed=None, output_directory_name='./'):
    """
    Generate many time courses of tumor growth and save data
    """

    prng = np.random.RandomState(random_seed)
    random_seed_array = prng.randint(0, 1000, number_realizations)
    number_of_states = int(parameterValues['number_of_states'])
    time_points = -np.ones((number_realizations, number_of_states), dtype=float)
    number_C_cells = -np.ones((number_realizations, number_of_states), dtype=int)
    number_Q_cells = -np.ones((number_realizations, number_of_states), dtype=int)
    from write import cleanUp_createFile, append_log_file
    flog = cleanUp_createFile(output_directory_name + 'tumor.log', 'a')
    for realization in range(number_realizations):
        tumor = Tumor(initialCondition, parameterValues, random_seed=random_seed_array[realization])
        time_course = tumor.generate_time_course()
        time_points[realization, :] = time_course.time_points
        number_C_cells[realization, :] = time_course.number_C_cells
        number_Q_cells[realization, :] = time_course.number_Q_cells
        append_log_file(flog, realization, time_course)
    print('finished generating time courses\n')

    # type '!unzip -l data.npz' in the Python Console to see individual files in zip archive:
    np.savez(output_directory_name + parameterValues['data_file_name'],
             time_points=time_points,
             number_C_cells=number_C_cells,
             number_Q_cells=number_Q_cells)


def generate_tumor_growth_trajectories(number_realizations, random_seed=None):
    """
    Read in initial conditions & parameter values, and generate many time courses of tumor growth and save data
    """

    initialCondition = read_into_dict('initialCondition.in')
    parameterValues = read_into_dict('parameterValues.in')

    generate_tumor_growth_trajectories_base(initialCondition, parameterValues, number_realizations, random_seed)


if __name__ == '__main__':

    generate_tumor_growth_trajectories(number_realizations=1, random_seed=2)

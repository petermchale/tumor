""" Simulate the stochastic evolution of a tumor """

from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from read import read_into_dict
from tumor import Tumor


def animate_tumor_growth(number_of_frames, random_seed=None, run_mode='plot animation'):
    """
    Generate (and optionally save) a movie of the simulated growth of a tumor containing cycling (green) and quiescent (red) cells,
    each of which is either wild-type (indicated by absence of a star) or mutant (indicated by a star).
    """

    initialCondition = read_into_dict('initialCondition.in')
    parameterValues = read_into_dict('parameterValues.in')
    tumor = Tumor(initialCondition, parameterValues, random_seed=random_seed)

    fig = plt.figure(figsize=(7, 7), facecolor='w')
    half_width_box = 12
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-half_width_box, half_width_box), ylim=(-half_width_box, half_width_box))

    cell_markersize = 12
    star_markersize = 10
    C_cell_positions, = ax.plot([], [], 'go', ms=cell_markersize)
    Q_cell_positions, = ax.plot([], [], 'ro', ms=cell_markersize)
    mutant_cell_positions, = ax.plot([], [], 'w*', ms=star_markersize, markeredgewidth=0.0)

    # transform=ax.transAxes tells the Text object
    # to use the coordinate system of the ax object, in which
    # (0,0) is bottom left of the axes and (1,1) is top right of the axes.
    numberSteps_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    numberCCells_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    numberQCells_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
    tumorEnergy_text = ax.text(0.02, 0.10, '', transform=ax.transAxes)
    cellPairEnergy_text = ax.text(0.02, 0.05, '', transform=ax.transAxes)

    def create_initial_frame():

        C_cell_positions.set_data([], [])
        Q_cell_positions.set_data([], [])
        mutant_cell_positions.set_data([], [])

        time_text.set_text('')
        numberCCells_text.set_text('')
        numberQCells_text.set_text('')
        numberSteps_text.set_text('')
        tumorEnergy_text.set_text('')
        cellPairEnergy_text.set_text('')

        return (C_cell_positions, Q_cell_positions, mutant_cell_positions, time_text,
                numberCCells_text, numberQCells_text, numberSteps_text, tumorEnergy_text, cellPairEnergy_text)

    # noinspection PyUnusedLocal
    def create_frame(frame_number):

        try:
            tumor.next_frame()
        except IndexError:
            time_elapsed_string = 'time elapsed = ' + str(tumor.time_elapsed)
            end_point_string = 'last time point = ' + str(tumor.time_course.end_time_point())
            raise IndexError('\n' + time_elapsed_string + '\n' + end_point_string)

        C_cell_positions.set_data(tumor.C_cell_positions())
        Q_cell_positions.set_data(tumor.Q_cell_positions())
        mutant_cell_positions.set_data(tumor.mutant_cell_positions())

        time_text.set_text('time = %.4f' % tumor.time_elapsed)
        numberCCells_text.set_text('number of cycling cells = %d' % tumor.number_of_C_Cells())
        numberQCells_text.set_text('number of quiescent cells = %d' % tumor.number_of_Q_Cells())
        numberSteps_text.set_text('number of steps = %d' % tumor.number_of_steps)
        if tumor.energy != 'undefined':  # use '==' when comparing values and 'is' when comparing identities
            tumorEnergy_text.set_text('tumor energy = %.4f' % tumor.energy)
        else:
            tumorEnergy_text.set_text('tumor energy = undefined')
        if tumor.cell_pair_energy() != 'undefined':
            cellPairEnergy_text.set_text('average cell-cell energy = %.4f' % tumor.cell_pair_energy())
        else:
            cellPairEnergy_text.set_text('average cell-cell energy = undefined')

        return (C_cell_positions, Q_cell_positions, mutant_cell_positions, time_text,
                numberCCells_text, numberQCells_text, numberSteps_text, tumorEnergy_text, cellPairEnergy_text)

    animation_reference = \
        animation.FuncAnimation(fig, create_frame, frames=number_of_frames, interval=1, blit=True, init_func=create_initial_frame, repeat=False)

    if run_mode == 'save animation':
        # requires that user has installed ffmpeg, e.g. using homebrew on Mac OS X
        plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
        animation_reference.save('tumor.mp4', writer=animation.FFMpegWriter(fps=30), dpi=200)
    elif run_mode == 'plot animation':
        fig.tight_layout()
        plt.show()
    else:
        print '\'' + run_mode + '\'', 'is an invalid run mode'
        print 'exiting'
        exit()


def generate_tumor_growth_trajectories(number_realizations, random_seed=None):
    """
    Generate many time courses of tumor growth and save data
    """

    prng = np.random.RandomState(random_seed)
    random_seed_array = prng.randint(0, 1000, number_realizations)
    parameterValues = read_into_dict('parameterValues.in')
    number_of_states = parameterValues['number_of_states']
    time_points = -np.ones((number_realizations, number_of_states), dtype=float)
    number_C_cells = -np.ones((number_realizations, number_of_states), dtype=float)
    number_Q_cells = -np.ones((number_realizations, number_of_states), dtype=float)
    initialCondition = read_into_dict('initialCondition.in')
    from write import cleanUp_createFile, append_log_file
    flog = cleanUp_createFile('tumor.log', 'a')
    for realization in xrange(number_realizations):
        tumor = Tumor(initialCondition, parameterValues, random_seed=random_seed_array[realization])
        time_course = tumor.generate_time_course()
        time_points[realization, :] = time_course.time_points
        number_C_cells[realization, :] = time_course.number_C_cells
        number_Q_cells[realization, :] = time_course.number_Q_cells
        append_log_file(flog, realization, time_course)
    print 'finished generating time courses\n'

    # type '!unzip -l data.npz' in the Python Console to see individual files in zip archive:
    np.savez(parameterValues['data_file_name'],
             time_points=time_points,
             number_C_cells=number_C_cells,
             number_Q_cells=number_Q_cells)


if __name__ == '__main__':

    animate_tumor_growth(400, run_mode='plot animation')

    # generate_tumor_growth_trajectories(3, random_seed=2)

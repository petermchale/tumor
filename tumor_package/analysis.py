""" Analyze a tumor growth simulation """

from matplotlib import pyplot as plt
import numpy as np


def analyze_tumor_growth_trajectories(data_file_name):
    """
    Read in data for a large number of tumors and analyze their statistics
    """

    # From the documentation: "For ``.npz`` files,
    # the returned instance of NpzFile class
    # must be closed to avoid leaking file descriptors."
    with np.load(data_file_name) as data_set:
        time_points = data_set['time_points']
        number_C_cells = data_set['number_C_cells']
        number_Q_cells = data_set['number_Q_cells']

    total_number_cells = number_C_cells + number_Q_cells

    time_at_which_to_plot_histogram = 10
    index_at_which_to_plot_histogram = (np.abs(time_points[0, :] - time_at_which_to_plot_histogram)).argmin()
    time_at_which_to_plot_histogram = time_points[0, index_at_which_to_plot_histogram]

    total_number_cells_at_time_point = total_number_cells[:, index_at_which_to_plot_histogram]
    number_C_cells_at_time_point = number_C_cells[:, index_at_which_to_plot_histogram]
    number_Q_cells_at_time_point = number_Q_cells[:, index_at_which_to_plot_histogram]

    def plot_figure(number_cells, number_cells_at_time_point, cell_type='', bin_width=5, step=4):

        # set up figure
        number_subplot_rows = 1
        number_subplot_columns = 2
        fig = plt.figure(figsize=(14, 7), facecolor='w')

        def thin_out(data):
            """get every step'th row of data"""

            return data[1::step, :]

        # select a small number of representative time courses
        selected_time_points = thin_out(time_points)
        selected_number_cells = thin_out(number_cells)

        x_max = max(time_points.flatten())
        y_max = max(total_number_cells.flatten()) + 2*bin_width

        # plot time courses
        ax = fig.add_subplot(number_subplot_rows, number_subplot_columns, 1, xlim=(0, x_max), ylim=(-1, y_max))
        ax.plot(selected_time_points.T, selected_number_cells.T, linewidth=1)
        ax.set_xlabel('time (cell cycles)')
        ax.set_ylabel('number of ' + cell_type + ' cells per tumor')
        ax.plot([time_at_which_to_plot_histogram, time_at_which_to_plot_histogram], [-1, y_max], linestyle='--', color='black')

        # plot distribution of number of cells at chosen time point, conditioned on tumor being large enough
        ax = fig.add_subplot(number_subplot_rows, number_subplot_columns, 2, xlim=(0, 1), ylim=(-1, y_max))
        bin_edges = np.arange(0.5, y_max, bin_width)
        counts = np.histogram(number_cells_at_time_point, bin_edges)[0]
        counts = np.array(counts)
        probabilities = counts/float(sum(counts))
        rectangle_height = 0.7 * (bin_edges[1] - bin_edges[0])
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        rectangle_bottoms = bin_centers - 0.5*rectangle_height
        rectangle_widths = probabilities
        ax.barh(rectangle_bottoms, rectangle_widths, height=rectangle_height, color='red')
        ax.set_xlabel('probability')
        ax.set_ylabel('number of ' + cell_type + ' cells per tumor, \ngiven that tumor is larger than its initial size')

    plot_figure(number_C_cells, number_C_cells_at_time_point, cell_type='cycling')
    # plot_figure(number_Q_cells, number_Q_cells_at_time_point, cell_type='quiescent')
    # plot_figure(total_number_cells, total_number_cells_at_time_point)

    plt.show()

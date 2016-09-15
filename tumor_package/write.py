"""Functions for writing diagnostic information to a log file during the course of many simulations"""

import os
import subprocess


def clean_up(filename):
    """if a file exists with the given name, then delete it"""

    if os.path.exists(filename):  # then move to trash
        posix_path = os.path.abspath(filename)
        applescript = 'tell app "Finder" to move ("' + posix_path + '" as POSIX file) to trash'
        subprocess.call(['osascript', '-e', applescript])


def cleanUp_createFile(filename, mode):
    """create a file with the given name to be written to according to the given mode"""

    clean_up(filename)

    fout = open(filename, mode)
    if os.path.isfile(filename):
        print(filename + " created")

    return fout


def append_log_file(flog, realization, time_course):
    """ Write diagnostic information to a log file after each tumor realization"""

    realization_string = 'realization = ' + str(realization)
    time_course_string = 'time course = ' + str(time_course)
    flog.write(realization_string + '\t' + time_course_string + '\n')
    flog.flush()  # flush the program buffer
    os.fsync(flog.fileno())  # flush the OS buffer
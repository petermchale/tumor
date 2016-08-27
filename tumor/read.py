""" Functions for parsing an input file into a Python dictionary """

import numpy as np


def try_float(element):
    """ try to convert string input to a float, otherwise return as a stripped string"""

    try:
        return float(element)
    except ValueError:
        element_stripped = element.strip()
        # print 'cannot convert \'' + element_stripped + '\' to float'
        return element_stripped


def parse_nameValuePair(name_value_pair_str):
    """take an input string of the form 'x=y' and return x and y"""

    name, value = name_value_pair_str.split('=')

    if '[' in value and ']' in value:
        value = value.strip().strip('[]')
        return [name.strip(), np.array([try_float(vv) for vv in value.split(',')])]
    else:
        return [name.strip(), try_float(value)]


def add_to_dictionary(line, parameterValue_dict):
    """ parse a line of the form 'x=y' and add it to the given dictionary """

    name, value = parse_nameValuePair(line)
    parameterValue_dict[name] = value
    return parameterValue_dict


def read_into_dict(file_name):
    """ read an 'input' file containing lines of the form 'x=y' into a dictionary """

    lines = open(file_name, 'r').readlines()

    # parse name-value pairs and add to a dictionary
    parameterValue_dict = {}
    for line in lines:
        parameterValue_dict = add_to_dictionary(line, parameterValue_dict)

    return parameterValue_dict

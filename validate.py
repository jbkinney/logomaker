from __future__ import division, print_function
import numpy as np
import pandas as pd
import ast
import inspect
import re
import sys
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

# Need for testing colors
import color
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

import warnings

# Revise warning output to just show warning, not file name or line number
def _warning(message, category = UserWarning, filename = '', lineno = -1):
    print('Warning: ' + str(message), file=sys.stderr)

# Comment this line if you want to see line numbers producing warnings
warnings.showwarning = _warning

# Comment this line if you don't want to see warnings multiple times
#warnings.simplefilter('always', UserWarning)


def _try_some_code(code_lines, **kwargs):
    """
    Returns True if any of the supplied lines of code do not throw an error.
    """
    is_valid = False
    for code_line in code_lines:
        try:
            exec(code_line)
            is_valid = True
        except:
            pass
    return is_valid

#
# Parameter specifications
#

# Valid values for matrix_type and logo_type
LOGOMAKER_TYPES = {'counts', 'probability', 'enrichment', 'energy',
                   'information'}

params_with_float_values = {
    'xtick_anchor'
}

# Names of numerical parameters that must be > 0
params_greater_than_0 = {
    'energy_gamma',
    'width',
    'xtick_spacing'
}

# Names of numerical parameters that must be >= 0
params_greater_or_equal_to_0 = {
    'pseudocount',
    'edgewidth',
    'highlight_edge_width',
    'baseline_width',
    'vline_width',
    'hpad',
    'vpad'
}

# Names of numerical parameters in the interval [0,1]
params_between_0_and_1 = {
    'alpha',
    'boxalpha',
    'highlight_alpha',
    'highlight_box_alpha',
    'below_alpha',
    'below_shade',
    'occurance_threshold',
}

# Names of parameters allowed to take on a small number of specific values
params_with_values_in_dict = {
    'matrix_type': LOGOMAKER_TYPES,
    'logo_type': LOGOMAKER_TYPES,
    'information_units': ['bits', 'nats'],
    'stack_order': ['big_on_top', 'small_on_top', 'fixed'],
    'axes_style': ['classic', 'naked', 'everything', 'rails',
                   'light_rails', 'vlines'],
    'enrichment_logbase': [2, np.e, 10]
}

# Names of parameters whose values are True or False
params_with_boolean_values = {
    'enrichment_centering',
    'uniform_stretch',
    'first_position_is_1',
    'use_transparency',
    'below_flip',
    'show_vlines',
    'show_binary_yaxis',
    'draw_now'
}

# Names of parameters whose values are strings
params_with_string_values = {
    'max_stretched_character',
    'highlight_sequence',
    'font_family',
    'font_style',
    'font_file',
    'energy_units',
    'xlabel',
    'ylabel',
    'title',
    'save_to_file'
}

# Names of parameters whose values specify a numerical interval
params_that_specify_intervals = {
    'use_position_range',
    'xlim',
    'ylim'
}

# Names of parameters whose values are ordered numerical arrays
params_that_are_ordered_arrays = {
    'positions',
    'xticks',
    'yticks'
}

# Names of parameters that specify color schemes
params_that_specify_colorschemes = {
    'colors',
    'edgecolors',
    'boxcolors',
    'highlight_colors'
}

# Names of parameters that specify colors:
params_that_specify_colors = {
    'vline_color'
}

# Names of parameters that cannot have None value
params_that_cant_be_none = {
    'pseudocount',
    'energy_gamma',
    'enrichment_logbase',
    'information_units',
    'alpha',
    'edgewidth',
    'hpad',
    'vpad',
    'axes_style',
    'stack_order',
    'baseline_width',
    'vline_width',
    'vline_color',
    'width',
    'draw_now'
}
# Do not allow any boolean parameters to be None
params_that_cant_be_none = \
    params_that_cant_be_none.union(params_with_boolean_values)

#
# Primary validation function
#


def validate_parameter(name, user, default):
    """
    Validates any parameter passed to make_logo or Logo.__init__.
    If user is valid of parameter name, silently returns user.
    If user is invalid, issues warning and retuns default instead
    """

    # Skip if value is none
    if user is None:
        if name in params_that_cant_be_none:
            raise ValueError("Parameter '%s' cannot be None.")
        else:
            value = user

    # Skip these two specific parameters,
    # Which can't be validated except in the context of other parameters
    elif name in ['background', 'font_weight']:
        value = user

    elif name in params_with_values_in_dict:
        value = _validate_in_set(name, user, default,
                                 params_with_values_in_dict[name])

    elif name in params_with_boolean_values:
        value = _validate_bool(name, user, default)

    elif name in params_with_string_values:
        value = _validate_str(name, user, default)

    elif name in params_with_float_values:
        value = _validate_float(name, user, default)

    elif name in params_greater_than_0:
        value = _validate_float(name, user, default,
                                greater_than=0.0)

    elif name in params_greater_or_equal_to_0:
        value = _validate_float(name, user, default,
                                greater_than_or_equal_to=0.0)

    elif name in params_between_0_and_1:
        value = _validate_float(name, user, default,
                                greater_than_or_equal_to=0.0,
                                less_than_or_equal_to=1.0)

    elif name in params_that_specify_intervals:
        value = _validate_array(name, user, default, length=2)

    elif name in params_that_are_ordered_arrays:
        value = _validate_array(name, user, default, increasing=True)

    elif name in params_that_specify_colorschemes:
        value = _validate_colorscheme(name, user, default)

    elif name in params_that_specify_colors:
        value = _validate_color(name, user, default)

    elif name == 'matrix':
        value = validate_mat(user)

    elif name == 'figsize':
        value = _validate_array(name, user, default, length=2)

    elif name == 'rcparams':
        if type(user)==dict:
            value = user
        else:
            message = "rcparams = %s is not a dictionary. Using % instead." \
            % (repr(user),repr(default))
            warnings.warn(message, UserWarning)

    else:
        warnings.warn("'%s' parameter not validated." % name, UserWarning)
        value = user

    return value

#
# Private validation functions
#


def _validate_float(name,
                    user,
                    default,
                    greater_than=-np.Inf,
                    greater_than_or_equal_to=-np.Inf,
                    less_than=np.Inf,
                    less_than_or_equal_to=np.Inf,
                    in_set=None):
    """ Validates a floating point parameter. """

    # Test whether parameter can be interpreted as a float
    try:
        value = float(user)

    except ValueError:
        value = default
        message = "Cannot interpret value %s for parameter '%s' as float. " +\
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Test inequalities
    if not value > greater_than:
        value = default
        message = "Value %s for parameter '%s' is not greater than %s. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(greater_than),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif not value >= greater_than_or_equal_to:
        value = default
        message = "Value %s for parameter '%s' is not greater or equal to %s." + \
                  " Using default value %s instead."
        message = message % (repr(user), name, repr(greater_than_or_equal_to),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif not value < less_than:
        value = default
        message = "Value %s for parameter '%s' is not less than %s. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(less_than),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif not value <= less_than_or_equal_to:
        value = default
        message = "Value %s for parameter '%s' is not less or equal to %s. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(less_than_or_equal_to),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif (in_set is not None) and not (value in in_set):
        value = default
        message = "Value %s for parameter '%s' is not within the set. " + \
                  "of valid values %s. Using default value %s instead."
        message = message % (repr(user), name, repr(in_set),
                             repr(default))
        warnings.warn(message, UserWarning)

    return value


def _validate_bool(name, user, default):
    """ Validates a floating point parameter. """

    # Test whether parameter is already a boolean
    # (not just whether it can be interpreted as such)
    if isinstance(user, bool):
        value = user

    # If not, return default value and raise warning
    else:
        value = default
        message = "Parameter '%s' assigned a non-boolean value. " +\
                  "Using default value %s instead."
        message = message % (name, repr(default))
        warnings.warn(message, UserWarning)

    return value


def _validate_in_set(name, user, default, in_set):
    """ Validates a parameter with a finite number of valid values. """

    # If user is valid, use that
    if user in in_set:
        value = user

    # If user value is not valid, set to default and issue warning
    else:
        value = default
        message = "Invalid value %s for parameter '%s'. " + \
                           "Using default value %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_str(name, user, default):
    """ Validates a string parameter. """

    # Test whether parameter can be interpreted as a string
    try:
        value = str(user)

    # If user value is not valid, set to default and issue warning
    except ValueError:
        value = default
        message = "Cannot interpret value %s for parameter '%s' as string. " +\
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_array(name, user, default, length=None, increasing=False):
    """ Validates an array of numbers. """

    try:
        if length is not None:
            assert len(user) == length

        for i in range(len(user)):
            user[i] = float(user[i])

        if increasing:
            for i in range(1, len(user)):
                assert user[i - 1] < user[i]

        value = np.array(user).copy()

    except AssertionError:
        value = default
        message = "Improper value %s for parameter '%s'. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)
    # Return valid value to user
    return value

def _validate_colorscheme(name, user, default):
    """ Tests whether user input can be interpreted as a colorschme. """

    # Check whether any of the following lines of code execute without error
    code_lines = [
        'color.color_scheme_dict[user]',
        'plt.get_cmap(user)',
        'to_rgb(user)',
        'assert(user in ["none", "random"])',
        'expand_color_dict(user)'
    ]

    # Test lines of code
    is_valid = False
    for code_line in code_lines:
        try:
            exec(code_line)
            is_valid = True
        except:
            pass

    # If so, then colorscheme is valid
    if is_valid:
        value = user

    # Otherwise, use default colorscheme
    else:
        value = default
        message = "Improper value %s for parameter '%s'. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_color(name, user, default):
    """ Tests whether user input can be interpreted as an RGBA color. """

    # Check whether any of the following lines of code execute without error
    try:
        to_rgba(user)
        is_valid = True
    except ValueError:
        is_valid = False

    # If so, then colorscheme is valid
    if is_valid:
        value = user

    # Otherwise, use default colorscheme
    else:
        value = default
        message = "Improper value %s for parameter '%s'. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def validate_mat(matrix):
    '''
    Runs assert statements to verify that df is indeed a motif dataframe.
    Returns a cleaned-up version of df if possible
    '''

    # Copy and preserve logomaker_type
    try:
        mat_type = matrix.logomaker_type
    except:
        mat_type = None
    matrix = matrix.copy()

    assert type(matrix) == pd.core.frame.DataFrame, 'Error: df is not a dataframe'
    cols = matrix.columns

    for i, col_name in enumerate(cols):
        # Ok to have a 'pos' column
        if col_name=='pos':
            continue

        # Convert column name to simple string if possible
        assert isinstance(col_name,basestring), \
            'Error: column name %s is not a string'%col_name
        new_col_name = str(col_name)

        # If column name is not a single chracter, try extracting single character
        # after an underscore
        if len(new_col_name) != 1:
            new_col_name = new_col_name.split('_')[-1]
            assert (len(new_col_name)==1), \
                'Error: could not extract single character from colum name %s'%col_name

        # Make sure that colun name is not a whitespace character
        assert re.match('\S',new_col_name), \
            'Error: column name "%s" is a whitespace charcter.'%repr(col_name)

        # Set revised column name
        matrix.rename(columns={col_name:new_col_name}, inplace=True)

    # If there is a pos column, make that the index
    if 'pos' in cols:
        matrix.set_index('pos', drop=True, inplace=True)

    # Remove name from index column
    matrix.index.names = [None]

    # Alphabetize character columns
    char_cols = list(matrix.columns)
    char_cols.sort()
    matrix = matrix[char_cols]
    matrix.logomaker_type = mat_type

    # Return cleaned-up df
    return matrix

def validate_probability_mat(matrix):
    '''
    Verifies that the df is indeed a probability motif dataframe.
    Returns a normalized and cleaned-up version of df if possible
    '''

    # Validate as motif
    matrix = validate_mat(matrix)

    # Validate df values as info values
    assert (all(matrix.values.ravel() >= 0)), \
        'Error: not all values in df are >=0.'

    # Normalize across columns
    matrix.loc[:, :] = matrix.values / matrix.values.sum(axis=1)[:, np.newaxis]

    # Label matrix type
    matrix.logomaker_type = 'probability'

    return matrix
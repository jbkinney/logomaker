from __future__ import division, print_function
import numpy as np
import pandas as pd
import ast
import inspect
import re
import sys
import numbers
import matplotlib.pyplot as plt
import numbers
import pdb
import os

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.lines import Line2D
from character import font_manager

# Need for testing colors
import color
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

import warnings

# Specifies IUPAC string transformations
iupac_dict = {
    'A': 'A',
    'C': 'C',
    'G': 'G',
    'T': 'T',
    'R': 'AG',
    'Y': 'CT',
    'S': 'GC',
    'W': 'AT',
    'K': 'GT',
    'M': 'AC',
    'B': 'CGT',
    'D': 'AGT',
    'H': 'ACT',
    'V': 'ACG',
    'N': 'ACGT'
}

# Revise warning output to just show warning, not file name or line number
def _warning(message, category = UserWarning, filename = '', lineno = -1):
    print('Warning: ' + str(message), file=sys.stderr)

# Comment this line if you want to see line numbers producing warnings
warnings.showwarning = _warning

# Comment this line if you don't want to see warnings multiple times
# warnings.simplefilter('always', UserWarning)


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
LOGOMAKER_TYPES = {'counts', 'probability', 'enrichment', 'information'}

# Names of parameters that can take on any float value
params_with_float_values = {
    'xtick_anchor',
    'xtick_rotation',
    'ytick_rotation',
    'character_zorder',
    'highlight_zorder',
    'fullheight_zorder',
    'baseline_zorder',
    'vline_zorder',
    'scalebar_x',
    'scalebar_ymin',
    'scalebar_textrotation',
}

# Names of numerical parameters that must be > 0
params_greater_than_0 = {
    'dpi',
    'xtick_spacing',
    'max_positions_per_line',
}

# Names of numerical parameters that must be >= 0
params_greater_or_equal_to_0 = {
    'pseudocount',
    'counts_threshold',
    'character_edgewidth',
    'character_boxedgewidth',
    'highlight_edgewidth',
    'highlight_boxedgewidth',
    'fullheight_edgewidth',
    'fullheight_boxedgewidth',
    'max_alpha_val',
    'hpad',
    'vpad',
    'gridline_width',
    'baseline_width',
    'vline_width',
    'xtick_length',
    'ytick_length',
    'scalebar_length',
    'scalebar_linewidth',
}

# Names of numerical parameters in the interval [0,1]
params_between_0_and_1 = {
    'character_alpha',
    'character_edgealpha',
    'character_boxalpha',
    'character_boxedgealpha',
    'highlight_alpha',
    'highlight_edgealpha',
    'highlight_boxalpha',
    'highlight_boxedgealpha',
    'fullheight_alpha',
    'fullheight_edgealpha',
    'fullheight_boxalpha',
    'fullheight_boxedgealpha',
    'below_shade',
    'below_alpha',
    'width',
    'fullheight_width',
    'vsep',
    'fullheight_vsep',
    'gridline_alpha',
    'baseline_alpha',
}

# Names of parameters allowed to take on a small number of specific values
params_with_values_in_dict = {
    'matrix_type': LOGOMAKER_TYPES,
    'logo_type': LOGOMAKER_TYPES,
    'background_mattype': ['counts', 'probability'],
    'enrichment_logbase': [2, np.e, 10],
    'information_units': ['bits', 'nats'],
    'sequence_type': ['dna', 'DNA',
                      'rna', 'RNA',
                      'protein', 'PROTEIN'],
    'stack_order': ['big_on_top',
                    'small_on_top',
                    'fixed_going_up',
                    'fixed_going_down'],
    'axes_type': ['classic',
                  'naked',
                  'everything',
                  'rails',
                  'vlines',
                  'scalebar'],
    'gridline_axis': ['x', 'y', 'both'],
}

# Names of parameters whose values are True or False
params_with_boolean_values = {
    'center_columns',
    'draw_now',
    'use_transparency',
    'below_flip',
    'uniform_stretch',
    'show_gridlines',
    'show_baseline',
    'show_binary_yaxis',
    'left_spine',
    'right_spine',
    'top_spine',
    'bottom_spine',
    'use_tightlayout',
    'show_position_zero',
    'remove_flattened_characters',
    'csv_delim_whitespace',
    'highlight_bgconsensus',
    'negate_matrix',
    'show_scalebar',
}

# Names of parameters whose values are strings
params_with_string_values = {
    'meme_motifname',
    'save_to_file',
    'characters',
    'ignore_characters',
    'highlight_sequence',
    'max_stretched_character',
    'style_sheet',
    'xtick_format',
    'xlabel',
    'ytick_format',
    'ylabel',
    'title',
    'csv_delimiter',
    'scalebar_text',
    'scalebar_texthalignment',
    'scalebar_textvalignment'
}

# Names of parameters whose values specify a numerical interval
params_that_specify_intervals = {
    'position_range',
    'xlim',
    'ylim'
}

# Names of parameters whose values are ordered numerical arrays
params_that_are_ordered_arrays = {
    'xticks',
    'yticks'
}

# Names of parameters that specify color schemes
params_that_specify_colorschemes = {
    'character_colors',
    'character_edgecolors',
    'character_boxcolors',
    'character_boxedgecolors',
    'highlight_colors',
    'highlight_edgecolors',
    'highlight_boxcolors',
    'highlight_boxedgecolors',
    'fullheight_colors',
    'fullheight_edgecolors',
    'fullheight_boxcolors',
    'fullheight_boxedgecolors',
}

# Names of parameters that specify colors:
params_that_specify_colors = {
    'gridline_color',
    'baseline_color',
    'vline_color',
    'scalebar_color',
}


# Names of parameters that specify fontsize
params_that_specify_FontProperties = {
    'font_file': 'fname',
    'font_family': 'family',
    'font_weight': 'weight',
    'font_style': 'style',

    'axes_fontfile': 'fname',
    'axes_fontfamily': 'family',
    'axes_fontweight': 'weight',
    'axes_fontstyle': 'style',
    'axes_fontsize': 'size',

    'tick_fontfile': 'fname',
    'tick_fontfamily': 'family',
    'tick_fontweight': 'weight',
    'tick_fontstyle': 'style',
    'tick_fontsize': 'size',

    'label_fontfile': 'fname',
    'label_fontfamily': 'family',
    'label_fontweight': 'weight',
    'label_fontstyle': 'style',
    'label_fontsize': 'size',

    'title_fontfile': 'fname',
    'title_fontfamily': 'family',
    'title_fontweight': 'weight',
    'title_fontstyle': 'style',
    'title_fontsize': 'size',
}

params_that_specify_linestyles = {
    'gridline_style',
    'baseline_style',
    'vline_style',
}

# Names of parameters that cannot have None value
params_that_cant_be_none = {
    'pseudocount',
    'enrichment_logbase',
    'center_columns',
    'information_units',
    'draw_now',
    'colors',
    'alpha',
    'edgecolors',
    'edgealpha',
    'edgewidth',
    'boxcolors',
    'boxalpha',
    'boxedgecolors',
    'boxedgealpha',
    'boxedgewidth',
    'stack_order',
    'use_transparency',
    'below_shade',
    'below_alpha',
    'below_flip',
    'hpad',
    'vpad',
    'width',
    'uniform_stretch',
    'axes_type',
    'rcparams',
    'xtick_anchor',
    'use_tightlayout',
    'show_position_zero',
    'highlight_bgconsensus',
    'negate_matrix'
}

# Parameters that specify tick labels
params_that_specify_ticklabels = {
    'xticklabels',
    'yticklabels',
}

# Parameters that specify file names
params_that_specify_filenames = {
    'fasta_file',
    'meme_file',
    'sequences_csvfile',
    'background_seqcsvfile',
    'matrix_csvfile',
    'background_matcsvfile'
}

# Parameters that specify dictionaries
params_that_specify_dicts = {
    'rcparams',
    'csv_kwargs',
    'background_csvkwargs'
}

# Names of parameters to leave for later validatation
params_for_later_validation = {
    'background',
    'ct_col',
    'background_ctcol',
    'seq_col',
    'background_seqcol',
    'csv_index_col',
    'csv_header',
    'csv_usecols',
}


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
            raise ValueError("Parameter '%s' cannot be None." % name)
        else:
            value = user

    # Special case: enrichment_logbase: validate as string
    elif name == 'enrichment_logbase':
        str_to_num_dict = {'2': 2, '10': 10, 'e': np.e}
        if isinstance(user, str):
            user = str_to_num_dict[user]
        value = _validate_in_set(name, user, default,
                               params_with_values_in_dict[name])

    #  If value is in a set
    elif name in params_with_values_in_dict:
        value = _validate_in_set(name, user, default,
                                 params_with_values_in_dict[name])

    # If value is boolean
    elif name in params_with_boolean_values:
        value = _validate_bool(name, user, default)

    # If value is str
    elif name in params_with_string_values:
        value = _validate_str(name, user, default)

    # If value is float
    elif name in params_with_float_values:
        value = _validate_number(name, user, default)

    # If value is float > 0
    elif name in params_greater_than_0:
        value = _validate_number(name, user, default,
                                 greater_than=0.0)

    # If value is float >= 0
    elif name in params_greater_or_equal_to_0:
        value = _validate_number(name, user, default,
                                 greater_than_or_equal_to=0.0)

    # If value is float in [0,1]
    elif name in params_between_0_and_1:
        value = _validate_number(name, user, default,
                                 greater_than_or_equal_to=0.0,
                                 less_than_or_equal_to=1.0)

    # If value is an interval
    elif name in params_that_specify_intervals:
        value = _validate_array(name, user, default, length=2)

    # If value is an ordered array
    elif name in params_that_are_ordered_arrays:
        value = _validate_array(name, user, default, increasing=True)

    # If value specifies a color scheme
    elif name in params_that_specify_colorschemes:
        value = _validate_colorscheme(name, user, default)

    # If value specifies a color
    elif name in params_that_specify_colors:
        value = _validate_color(name, user, default)

    # If value specifies FontProperties object
    elif name in params_that_specify_FontProperties:
         passedas = params_that_specify_FontProperties[name]
         value = _validate_FontProperties_parameter(name, user, default,
                                                    passedas=passedas)

    # If value specifies a linestyle
    elif name in params_that_specify_linestyles:
        value = _validate_linestyle(name, user, default)

    # If value specifies ticklabels
    elif name in params_that_specify_ticklabels:
        value = _validate_ticklabels(name, user, default)

    # If value specifies a filename
    elif name in params_that_specify_filenames:
        value = _validate_filename(name, user, default)

    # If value specifies a dicitionary
    elif name in params_that_specify_dicts:
        value = _validate_dict(name, user, default)

    # Special case: shift_first_position_to
    elif name == 'shift_first_position_to':
        value = _validate_number(name, user, default, is_int=True)

    # Special case: max_positions_per_line
    elif name in {'max_positions_per_line', 'meme_motifnum'}:
        value = _validate_number(name, user, default, is_int=True,
                                 greater_than=0)

    # Special case: iupac_string
    elif name == 'iupac_string':
        value = _validate_iupac(name, user, default)

    # Special case: matrix
    elif name == 'matrix':
        value = validate_mat(user, allow_nan=True)

    # Special case: figsize
    elif name == 'figsize':
        value = _validate_array(name, user, default, length=2)

    # Special case: vline_positions
    elif name == 'vline_positions':
        value = _validate_array(name, user, default)

    # Special case: fullheight
    elif name == 'fullheight':

        # Verify is a dictionary
        value = _validate_fullheight(name, user, default)

    # Parameters left for validation later on
    elif name in params_for_later_validation:
        value = user

    # Otherwise, warn if parameter passed through all filters
    else:
        warnings.warn("'%s' parameter not validated." % name, UserWarning)
        value = user

    return value

#
# Private validation functions
#


def _validate_number(name,
                     user,
                     default,
                     is_int=False,
                     greater_than=-np.Inf,
                     greater_than_or_equal_to=-np.Inf,
                     less_than=np.Inf,
                     less_than_or_equal_to=np.Inf,
                     in_set=None):
    """ Validates a floating point parameter. """

    # Test whether parameter can be interpreted as a float
    try:
        # If converting to int
        if is_int:
            value = int(user)

        # Otherwise, if converting to float
        else:
            value = float(user)

    except (ValueError, TypeError):
        value = default
        message = "Cannot interpret value %s for parameter '%s' as number. " +\
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
    """ Validates a boolean parameter parameter. """

    # Convert to bool if string is passed
    if isinstance(user, basestring):
        if user == 'True':
            user = True
        elif user == 'False':
            user = False
        else:
            user = default
            message = "Parameter '%s', if string, must be " + \
                      "'True' or 'False'. Using default value %s instead."
            message = message % (name, repr(default))
            warnings.warn(message, UserWarning)

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

    # Otherwise, if user is string, try evaluating as literal
    elif isinstance(user, basestring):
        try:
            tmp = ast.literal_eval(user)
            if tmp in in_set:
                value = tmp

        except:
            value = default
            message = "Invalid value %s for parameter '%s'. " + \
                      "Using default value %s instead."
            message = message % (repr(user), name, repr(default))
            warnings.warn(message, UserWarning)

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

def _validate_iupac(name, user, default):
    """ Validates an IUPAC string """

    message = None

    # Check that user input is a string
    if not isinstance(user, basestring):
        value = default
        message = "Value %s for parameter '%s' is not a string. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))

    # Make sure string has nonzero length
    elif len(user) == 0:
        value = default
        message = "String %s, set for parameter '%s', is empty. " + \
                  "Using default value %s instead."
        message = message % (repr(user), name, repr(default))

    # Make sure string contains valid characters
    elif not set(list(user.upper())) <= set(iupac_dict.keys()):
        value = default
        message = "String %s, set for parameter '%s', contains " + \
                  "invalid characters. Using default value %s instead."
        message = message % (repr(user), name, repr(default))

    # Make sure string is all capitals
    elif any([c == c.lower() for c in list(user)]):
        value = user.upper()
        message = "String %s, set for parameter '%s', contains lowercase " + \
                  "characters. Using capitalized characters instead."
        message = message % (repr(user), name)

    # If all tests pass, use user input
    else:
        value = user

    if message is not None:
        warnings.warn(message, UserWarning)

    return value


def _validate_filename(name, user, default):
    """ Validates a string parameter. """

    # Test whether file exists and can be opened
    message = None
    try:

        if not os.path.isfile(user):
            value = default
            message = "File %s passed for parameter '%s' does not exist. " +\
                      "Using default value %s instead."
            message = message % (repr(user), name, repr(default))

        elif open(user, 'r'):
            value = user
        else:
            value = default
            message = "File %s passed for parameter '%s' cannot be opened." + \
                      " Using default value %s instead."
            message = message % (repr(user), name, repr(default))

    except (ValueError,TypeError):
        value = default
        if message is None:
            message = "Value %s passed for parameter '%s' is invalid." + \
                      " Using default value %s instead."
            message = message % (repr(user), name, repr(default))

    if message is not None:
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_fullheight(name, user, default):
    """ Validates a fullheight specificaiton, which can be either
     a dictionary or an array/list. """

    # Test whether parameter can be interpreted as a string
    if isinstance(user, basestring):
        user = ast.literal_eval(user)

    # If dictionary
    if isinstance(user, dict):

        # Make sure keys are ints and vals are length 1 strings
        keys = [int(k) for k in user.keys()]
        vals = [str(v) for v in user.values()]

        assert all([len(v) == 1 for v in vals]), \
         'Error: multiple characters passed to single position in fullheight'

        value = dict(zip(keys, vals))

    # If list
    elif isinstance(user, (list, np.array)):
        value = np.array(user).astype(int)

    else:
        value = default
        message = "Invalid value %s for parameter %s. " +\
                  "Using default %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_array(name, user, default, length=None, increasing=False):
    """ Validates an array of numbers. """

    try:
        # If string, convert to list of numbers
        if isinstance(user, basestring):
            user = ast.literal_eval(user)

        if length is not None:
            assert len(user) == length

        for i in range(len(user)):
            user[i] = float(user[i])

        if increasing:
            for i in range(1, len(user)):
                assert user[i - 1] < user[i]

        value = np.array(user).copy()

    except (AssertionError, ValueError):
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
        'color.expand_color_dict(user)'
    ]

    # Test lines of code
    is_valid = False
    for code_line in code_lines:
        try:
            eval(code_line)
            is_valid = True
        except:
            pass

    # For some reason, this needs to be tested separately.
    if user == 'random':
        is_valid = True

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


def _validate_FontProperties_parameter(name, user, default, passedas):
    """ Validates any parameter passed to the FontProperties constructor. """

    try:
        # Create a FontProperties object and try to use it for something
        prop = FontProperties(**{passedas:user})
        TextPath((0,0), 'A', size=1, prop=prop)

        value = user
    except ValueError:
        value = default
        message = ("Invalid string specification '%s' for parameter '%s'. "
                   + "Using default value %s instead.") \
                  % (user, name, default)
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_linestyle(name, user, default):
    """ Validates any parameter that specifies a linestyle. """

    try:
        # Create a FontProperties object and try to use it for something
        Line2D((0, 1), (0, 1), linestyle=user)
        value = user

    except (ValueError, TypeError):
        value = default
        message = ("Invalid string specification '%s' for parameter '%s'. "
                   + "Using default value %s instead.") \
                  % (user, name, default)
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def _validate_dict(name, user, default):
    """ Validates any parameter that specifies a dictionary. """

    if type(user) == dict:
        value = user
    else:
        message = "%s = %s is not a dictionary. Using %s instead." \
                  % (name, repr(user), repr(default))
        warnings.warn(message, UserWarning)
        value = default

    # Return valid value to user
    return value


def _validate_ticklabels(name, user, default):
    """ Validates parameters passed as tick labels. """

    message = None
    # Check that user can be read as list
    try:
        user = list(user)
    except TypeError:
        message = ("Cant interpret value '%s' for parameter '%s' as list. "
                   + "Using default value %s instead.") \
                  % (user, name, default)

    # Test that elements of user are strings or numbers
    tests = [isinstance(u, basestring) or isinstance(u, numbers.Number) \
                for u in user]
    if len(tests) > 0 and not all(tests):
        message = ("Cant interpret all elements of '%s', "
                   + "assigned to parameter '%s', as string or number. "
                   + "Using default value %s instead.") \
                  % (user, name, default)

    # If any errors were encountered, use default as value and display warning
    # Otherwise, use user input as value.
    if message is None:
        value = user
    else:
        value = default
        warnings.warn(message, UserWarning)

    # Return valid value to user
    return value


def validate_mat(matrix, allow_nan=True):
    '''
    Runs assert statements to verify that df is indeed a motif dataframe.
    Returns a cleaned-up version of df if possible
    '''

    # Copy and preserve logomaker_type
    matrix = matrix.copy()

    assert type(matrix) == pd.core.frame.DataFrame, 'Error: df is not a dataframe'

    if not allow_nan:
        # Make sure all entries are finite numbers
        assert np.isfinite(matrix.values).all(), \
            'Error: some matrix elements are not finite.' +\
            'Set allow_nan=True to allow.'

    # Make sure the matrix has a finite number of rows and columns
    assert matrix.shape[0] >= 1, 'Error: matrix has zero rows.'
    assert matrix.shape[1] >= 1, 'Error: matrix has zero columns.'

    # Remove columns whose names aren't strings exactly 1 character long.
    # Warn user when doing so
    cols = matrix.columns
    for col in cols:
        if not isinstance(col, basestring) or (len(col) != 1):
            del matrix[col]
            message = ('Matrix has invalid column name "%s". This column ' +
                       'has been removed.') % col
            warnings.warn(message, UserWarning)

    cols = matrix.columns
    for i, col_name in enumerate(cols):
        # Ok to have a 'pos' column
        if col_name == 'pos':
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
        matrix['pos'] = matrix['pos'].astype(int)
        matrix.set_index('pos', drop=True, inplace=True)

    # Remove name from index column
    matrix.index.names = [None]

    # Alphabetize character columns
    char_cols = list(matrix.columns)
    char_cols.sort()
    matrix = matrix[char_cols]

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

    return matrix


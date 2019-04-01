from __future__ import division
import numpy as np
import pandas as pd
import re
import os
import warnings
import pdb

from six import string_types
from matplotlib.colors import to_rgb, to_rgba
from logomaker.src.error_handling import check, handle_errors


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
        message = "Cannot interpret message %s for parameter '%s' as number. " +\
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Test inequalities
    if not value > greater_than:
        value = default
        message = "Value %s for parameter '%s' is not greater than %s. " + \
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(greater_than),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif not value >= greater_than_or_equal_to:
        value = default
        message = "Value %s for parameter '%s' is not greater or equal to %s." + \
                  " Using default message %s instead."
        message = message % (repr(user), name, repr(greater_than_or_equal_to),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif not value < less_than:
        value = default
        message = "Value %s for parameter '%s' is not less than %s. " + \
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(less_than),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif not value <= less_than_or_equal_to:
        value = default
        message = "Value %s for parameter '%s' is not less or equal to %s. " + \
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(less_than_or_equal_to),
                             repr(default))
        warnings.warn(message, UserWarning)

    elif (in_set is not None) and not (value in in_set):
        value = default
        message = "Value %s for parameter '%s' is not within the set. " + \
                  "of valid values %s. Using default message %s instead."
        message = message % (repr(user), name, repr(in_set),
                             repr(default))
        warnings.warn(message, UserWarning)

    return value


def _validate_bool(name, user, default):
    """ Validates a boolean parameter parameter. """

    # Convert to bool if string is passed
    #if isinstance(user, basestring):
    if isinstance(user, string_types):
        if user == 'True':
            user = True
        elif user == 'False':
            user = False
        else:
            user = default
            message = "Parameter '%s', if string, must be " + \
                      "'True' or 'False'. Using default message %s instead."
            message = message % (name, repr(default))
            warnings.warn(message, UserWarning)

    # Test whether parameter is already a boolean
    # (not just whether it can be interpreted as such)
    if isinstance(user, bool):
        value = user

    # If not, return default message and raise warning
    else:
        value = default
        message = "Parameter '%s' assigned a non-boolean message. " +\
                  "Using default message %s instead."
        message = message % (name, repr(default))
        warnings.warn(message, UserWarning)

    return value


def _validate_str(name, user, default):
    """ Validates a string parameter. """

    # Test whether parameter can be interpreted as a string
    try:
        value = str(user)

    # If user message is not valid, set to default and issue warning
    except ValueError:
        value = default
        message = "Cannot interpret message %s for parameter '%s' as string. " +\
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid message to user
    return value


def _validate_iupac(name, user, default):
    """ Validates an IUPAC string """

    message = None

    # Check that user input is a string
    if not isinstance(user, string_types):
        value = default
        message = "Value %s for parameter '%s' is not a string. " + \
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(default))

    # Make sure string has nonzero length
    elif len(user) == 0:
        value = default
        message = "String %s, set for parameter '%s', is empty. " + \
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(default))

    # Make sure string contains valid characters
    elif not set(list(user.upper())) <= set(iupac_dict.keys()):
        value = default
        message = "String %s, set for parameter '%s', contains " + \
                  "invalid characters. Using default message %s instead."
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
                      "Using default message %s instead."
            message = message % (repr(user), name, repr(default))

        elif open(user, 'r'):
            value = user
        else:
            value = default
            message = "File %s passed for parameter '%s' cannot be opened." + \
                      " Using default message %s instead."
            message = message % (repr(user), name, repr(default))

    except (ValueError,TypeError):
        value = default
        if message is None:
            message = "Value %s passed for parameter '%s' is invalid." + \
                      " Using default message %s instead."
            message = message % (repr(user), name, repr(default))

    if message is not None:
        warnings.warn(message, UserWarning)

    # Return valid message to user
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
        message = "Improper message %s for parameter '%s'. " + \
                  "Using default message %s instead."
        message = message % (repr(user), name, repr(default))
        warnings.warn(message, UserWarning)

    # Return valid message to user
    return value


@handle_errors
def validate_matrix(dataframe, allow_nan=False):
    """
    Runs checks to verify that df is indeed a motif dataframe.
    Returns a cleaned-up version of df if possible
    """

    check(isinstance(dataframe, pd.DataFrame),
          'dataframe needs to be a valid pandas dataframe, dataframe entered: ' + str(type(dataframe)))

    # Create copy of dataframe so that don't overwrite the user's data
    dataframe = dataframe.copy()

    check(isinstance(allow_nan,bool),'allow_nan = %s must be of type bool'%type(allow_nan))

    # Copy and preserve logomaker_type
    dataframe = dataframe.copy()

    if not allow_nan:
        # Make sure all entries are finite numbers
        check(np.isfinite(dataframe.values).all(),'some matrix elements are not finite.' + 'Set allow_nan=True to allow.')

    # Make sure the matrix has a finite number of rows and columns
    check(dataframe.shape[0] >= 1, 'matrix has zero rows.')
    check(dataframe.shape[1] >= 1, 'matrix has zero columns.')

    # Remove columns whose names aren't strings exactly 1 character long.
    # Warn user when doing so
    cols = dataframe.columns
    for col in cols:
        if not isinstance(col, string_types) or (len(col) != 1):
            del dataframe[col]
            message = ('Matrix has invalid column name "%s". This column ' +
                       'has been removed.') % col
            warnings.warn(message, UserWarning)

    cols = dataframe.columns
    for i, col_name in enumerate(cols):
        # Ok to have a 'pos' column
        if col_name == 'pos':
            continue

        # Convert column name to simple string if possible
        check(isinstance(col_name, string_types), 'column name %s is not a string' % col_name)
        new_col_name = str(col_name)

        # If column name is not a single chracter, try extracting single character
        # after an underscore
        if len(new_col_name) != 1:
            new_col_name = new_col_name.split('_')[-1]
            check((len(new_col_name)==1),'could not extract single character from colum name %s'%col_name)

        # Make sure that colun name is not a whitespace character
        check(re.match('\S',new_col_name),'column name "%s" is a whitespace charcter.'%repr(col_name))

        # Set revised column name
        dataframe.rename(columns={col_name:new_col_name}, inplace=True)

    # If there is a pos column, make that the index
    if 'pos' in cols:
        dataframe['pos'] = dataframe['pos'].astype(int)
        dataframe.set_index('pos', drop=True, inplace=True)

    # Remove name from index column
    dataframe.index.names = [None]

    # Alphabetize character columns
    char_cols = list(dataframe.columns)
    char_cols.sort()
    dataframe = dataframe[char_cols]

    # Return cleaned-up df
    return dataframe


@handle_errors
def validate_probability_mat(matrix):
    """
    Verifies that the df is indeed a probability matrix dataframe.
    Renormalizes df with Warning if it is not already normalized.
    Throws an error of df cannot be reliably normalized.
    """

    # Validate as motif
    matrix = validate_matrix(matrix)

    # Make sure all values are non-negative
    #assert (all(matrix.values.ravel() >= 0)), \
    #    'Error: not all values in df are >=0.'

    check(all(matrix.values.ravel() >= 0),
        'Error: not all values in df are >=0.')

    # Check to see if values sum to one
    sums = matrix.sum(axis=1).values

    # If any sums are close to zero, abort
    #assert not any(np.isclose(sums, 0.0)), \
    #    'Error: some columns in matrix sum to nearly zero.'
    check(not any(np.isclose(sums, 0.0)),
        'Error: some columns in matrix sum to nearly zero.')

    # If any sums are not close to one, renormalize all sums
    if not all(np.isclose(sums, 1.0)):
        print('Warning: Row sums in probability matrix are not close to 1. ' +
              'Reormalizing rows...')
        matrix.loc[:, :] = matrix.values / sums[:, np.newaxis]

    # Return data frame to user
    return matrix

@handle_errors
def validate_information_mat(matrix):
    """
    Verifies that the df is indeed an information matrix dataframe.
    Returns a cleaned-up version of df if possible
    """

    # Validate as motif
    matrix = validate_matrix(matrix)

    # Validate df values as info values
    #assert (all(matrix.values.ravel() >= 0)), \
    #    'Error: not all values in df are >=0.'
    check(all(matrix.values.ravel() >= 0),
            'Error: not all values in df are >=0.')

    # Return data frame to user
    return matrix
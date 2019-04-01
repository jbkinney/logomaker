from __future__ import division
import numpy as np
import pandas as pd
from logomaker.src.error_handling import check, handle_errors


@handle_errors
def validate_matrix(df, allow_nan=False):
    """
    Checks to make sure that the input dataframe, df, represents a valid
    matrix, i.e., an object that can be displayed as a logo.

    parameters
    ----------

    df: (dataframe)
        A pandas dataframe where each row represents an (integer) position
        and each column represents to a (single) character.

    allow_nan: (bool)
        Whether to allow NaN entries in the matrix.

    returns
    -------
    out_df: (dataframe)
        A cleaned-up version of df (if possible).
    """

    check(isinstance(df, pd.DataFrame),
          'out_df needs to be a valid pandas out_df, ' 
          'out_df entered: %s' % type(df))

    # Create copy of df so we don't overwrite the user's data
    out_df = df.copy()

    check(isinstance(allow_nan, bool),
          'allow_nan must be of type bool; is type %s.' % type(allow_nan))

    if not allow_nan:
        # Make sure all entries are finite numbers
        check(np.isfinite(out_df.values).all(),
              'some matrix elements are not finite. ' 
              'Set allow_nan=True to allow this.')

    # Make sure the matrix has a finite number of rows and columns
    check(out_df.shape[0] >= 1, 'df has zero rows. Needs multiple rows.')
    check(out_df.shape[1] >= 1, 'df has zero columns. Needs multiple columns.')

    # Check that all column names are strings and have length 1
    for i, col in enumerate(out_df.columns):
        check(isinstance(col, str),
              'column number %d is of type %s; must be a str' % (i, col))
        check(len(col) == 1,
              'column %d is %s and has length %d; ' % (i, repr(col), len(col))
              + 'must have length 1.')

    # Sort columns alphabetically
    char_cols = list(out_df.columns)
    char_cols.sort()
    out_df = out_df[char_cols]

    # Name out_df.index as 'pos'
    out_df.index.name = 'pos'

    # Try casting df.index as type int
    try:
        int_index = out_df.index.astype(int)
    except TypeError:
        check(False,
              'could not convert df.index to type int. Check that '
              'all positions have integer numerical values.')

    # Make sure that df.index values have not changed
    check(all(int_index == out_df.index),
          'could not convert df.index values to int without changing'
          'some values. Make sure that df.index values are integers.')

    # Check that all index values are unique
    check(len(set(out_df.index)) == len(out_df.index),
          'not all values of df.index are unique. Make sure all are unique.')

    # Return cleaned-up out_df
    return out_df


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
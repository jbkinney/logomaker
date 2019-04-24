from __future__ import division
import numpy as np
import pandas as pd
from logomaker.src.error_handling import check, handle_errors


@handle_errors
def validate_matrix(df, matrix_type=None, allow_nan=False):
    """
    Checks to make sure that the input dataframe, df, represents a valid
    matrix, i.e., an object that can be displayed as a logo.

    parameters
    ----------

    df: (dataframe)
        A pandas dataframe where each row represents an (integer) position
        and each column represents to a (single) character.

    matrix_type: (None or str)
        If 'probability', validates df as a probability matrix, i.e., all
        elements are in [0,1] and rows are normalized). If 'information',
        validates df as an information matrix, i.e., all elements >= 0.

    allow_nan: (bool)
        Whether to allow NaN entries in the matrix.

    returns
    -------
    out_df: (dataframe)
        A cleaned-up version of df (if possible).
    """

    # check that df is a dataframe
    check(isinstance(df, pd.DataFrame),
          'out_df needs to be a valid pandas out_df, ' 
          'out_df entered: %s' % type(df))

    # create copy of df so we don't overwrite the user's data
    out_df = df.copy()

    # check that type is valid
    check(matrix_type in {None, 'probability', 'information'},
          'matrix_type = %s; must be None, "probability", or "information"' %
          matrix_type)

    # check that allow_nan is boolean
    check(isinstance(allow_nan, bool),
          'allow_nan must be of type bool; is type %s.' % type(allow_nan))

    if not allow_nan:
        # make sure all entries are finite numbers
        check(np.isfinite(out_df.values).all(),
              'some matrix elements are not finite. ' 
              'Set allow_nan=True to allow this.')

    # make sure the matrix has a finite number of rows and columns
    check(out_df.shape[0] >= 1, 'df has zero rows. Needs multiple rows.')
    check(out_df.shape[1] >= 1, 'df has zero columns. Needs multiple columns.')

    # check that all column names are strings and have length 1
    for i, col in enumerate(out_df.columns):

        # convert from unicode to string for python 2
        col = str(col)
        check(isinstance(col, str),
              'column number %d is of type %s; must be a str' % (i, col))
        check(len(col) == 1,
              'column %d is %s and has length %d; ' % (i, repr(col), len(col))
              + 'must have length 1.')

    # sort columns alphabetically
    char_cols = list(out_df.columns)
    char_cols.sort()
    out_df = out_df[char_cols]

    # name out_df.index as 'pos'
    out_df.index.name = 'pos'

    # try casting df.index as type int
    try:
        int_index = out_df.index.astype(int)
    except TypeError:
        check(False,
              'could not convert df.index to type int. Check that '
              'all positions have integer numerical values.')

    # make sure that df.index values have not changed
    check(all(int_index == out_df.index),
          'could not convert df.index values to int without changing'
          'some values. Make sure that df.index values are integers.')

    # check that all index values are unique
    check(len(set(out_df.index)) == len(out_df.index),
          'not all values of df.index are unique. Make sure all are unique.')

    # if type is 'information', make sure elements are nonnegative
    if matrix_type is 'information':

        # make sure all elements are nonnegative
        check(all(df.values.ravel() >= 0), 'not all values in df are >=0.')

    # if type is 'probability', make sure elements are valid probabilities
    elif matrix_type is 'probability':

        # make sure all values are non-negative
        check(all(df.values.ravel() >= 0),
              'not all values in df are >=0.')

        # check to see if values sum to one
        sums = df.sum(axis=1).values

        # if any sums are close to zero, abort
        check(not any(np.isclose(sums, 0.0)),
              'some columns in df sum to nearly zero.')

        # if any sums are not close to one, renormalize all sums
        if not all(np.isclose(sums, 1.0)):
            print('in validate_matrix(): Row sums in df are not close to 1. '
                  'Reormalizing rows...')
            df.loc[:, :] = df.values / sums[:, np.newaxis]
            out_df = df.copy()

    # nothing more to check if type is None
    elif matrix_type is None:
        pass

    # return cleaned-up out_df
    return out_df


@handle_errors
def validate_probability_mat(df):
    """
    Verifies that the input dataframe df indeed represents a
    probability matrix. Renormalizes df with a text warning if it is not
    already normalized. Throws an error if df cannot be reliably normalized.

    parameters
    ----------

    df: (dataframe)
        A pandas dataframe where each row represents an (integer) position
        and each column represents to a (single) character.

    returns
    -------
    prob_df: (dataframe)
        A cleaned-up and normalized version of df (if possible).
    """

    # Validate as a matrix. Make sure this contains no NaN values
    prob_df = validate_matrix(df, allow_nan=False)

    # Make sure all values are non-negative
    check(all(prob_df.values.ravel() >= 0),
          'not all values in df are >=0.')

    # Check to see if values sum to one
    sums = prob_df.sum(axis=1).values

    # If any sums are close to zero, abort
    check(not any(np.isclose(sums, 0.0)),
          'some columns in prob_df sum to nearly zero.')

    # If any sums are not close to one, renormalize all sums
    if not all(np.isclose(sums, 1.0)):
        print('in validate_probability_mat(): '
              'Row sums in df are not close to 1. '
              'Reormalizing rows...')
        prob_df.loc[:, :] = prob_df.values / sums[:, np.newaxis]

    # Return validated probability matrix to user
    return prob_df

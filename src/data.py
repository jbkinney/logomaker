from __future__ import division
import numpy as np
import pandas as pd

# Logomaker imports
from logomaker import check, handle_errors
from logomaker.src.validate import validate_matrix, \
                                   validate_probability_mat, \
                                   validate_information_mat

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

# Set constants
SMALL = np.finfo(float).tiny
MATRIX_TYPES = {'counts', 'probability', 'weight', 'information'}


@handle_errors
def transform_matrix(df,
                     center_values=False,
                     normalize_values=False,
                     from_type=None,
                     to_type=None,
                     background=None,
                     pseudocount=1):
    """
    Performs transformations on a matrix. There are three types of
    transformations that can be performed:

    1. Center values:
        Subtracts the mean from each row in df. This is common for weight
        matrices or energy matrices. To do this, set center_values=True.

    2. Normalize values:
        Divides each row by the sum of the row. This is needed for probability
        matrices. To do this, set normalize_values=True.

    3. From/To transformations:
        Transforms from one type of matrix (e.g. 'counts') to another type
        of matrix (e.g. 'information'). To do this, set from_type and to_type
        arguments.

    Here are the mathematical formulas invoked by From/To transformations:

        from_type='counts' ->  to_type='probability':
            P_ic = (N_ic + l)/(N_i + C*l), N_i = sum_c(N_ic)

        from_type='probability' -> to_type='weight':
            W_ic = log_2(P_ic / Q_ic)

        from_type='weight' -> to_type='probability':
            P_ic = Q_ic * 2^(W_ic)

        from_type='probability' -> to_type='information':
            I_ic = P_ic * sum_d(P_id * log2(P_id / W_id))

        from_type='information' -> to_type='probability':
            P_ic = I_ic / sum_d(I_id)

        notation:
            i = position
            c, d = character
            l = pseudocount
            C = number of characters
            N_ic = counts matrix element
            P_ic = probability matrix element
            Q_ic = background probability matrix element
            W_ic = weight matrix element
            I_ic = information matrix element

    Using these five 1-step transformations, 2-step transformations
    are also enabled, e.g., from_type='counts' -> to_type='information'.

    parameters
    ----------

    df: (dataframe)
        The matrix to be transformed.

    center_values: (bool)
        Whether to center matrix values, i.e., subtract the mean from each
        row.

    normalize_values: (bool)
        Whether to normalize each row, i.e., divide each row by
        the sum of that row.

    from_type: (str)
        Type of input matrix. Must be one of 'counts', 'probability',
        'weight', or 'information'.

    to_type: (str)
        Type of output matrix. Must be one of 'probability', 'weight', or
        'information'. Can be 'counts' ONLY if from_type is 'counts' too.

    background: (array, or df)
        Specification of background probabilities. If array, should be the
        same length as df.columns and correspond to the probability of each
        column's character. If df, should be a probability matrix the same
        shape as df.

    pseudocount: (number >= 0)
        Pseudocount to use when transforming from a counts matrix to a
        probability matrix.

    returns
    -------
    out_df: (dataframe)
        Transformed matrix
    """

    # validate matrix dataframe
    df = validate_matrix(df)

    # validate center_values
    check(isinstance(center_values, bool),
          'type(center_values) = %s must be of type bool' %
          type(center_values))

    # validate normalize_values
    check(isinstance(normalize_values, bool),
          'type(normalize_values) = %s must be of type bool' %
          type(normalize_values))

    # validate from_type
    check((from_type in MATRIX_TYPES) or (from_type is None),
          'from_type = %s must be None or in %s' %
          (from_type, MATRIX_TYPES))

    # validate to_type
    check((to_type in MATRIX_TYPES) or (to_type is None),
          'to_type = %s must be None or in %s' %
          (to_type, MATRIX_TYPES))

    # validate background
    check(isinstance(background, (type([]), np.ndarray)) or
          (background is None),
          'type(background) = %s must be None or of type list or array' %
          type(background))

    # validate pseudocount
    check(isinstance(pseudocount, (int, float)),
          'type(pseudocount) = %s must be a number' % type(pseudocount))
    check(pseudocount >= 0,
          'pseudocount=%s must be >= 0' % pseudocount)

    # If centering values, do that
    if center_values is True:
        check((from_type is None) and (to_type is None),
              "If center_values is True, both from_type and to_type" 
              "must be None. Here, from_type=%s, to_type=%s" %
              (from_type, to_type))

        # Do centering
        out_df = _center_matrix(df)

    # Otherwise, if normalizing values, do that
    elif normalize_values is True:
        check((from_type is None) and (to_type is None),
              "If normalize_values is True, both from_type and to_type" 
              "must be None. Here, from_type=%s, to_type=%s" %
              (from_type, to_type))

        # Do centering
        out_df = _normalize_matrix(df)

    # otherwise, if to_type == from_type, just return matrix
    # Note, this is the only way that to_type='counts' is valid
    elif from_type == to_type:
        out_df = df.copy()

    # Otherwise, we're converting from one type of matrix to another. Do this.
    else:
        # Check that from_type and to_type are not None
        check((from_type is not None) and (to_type is not None),
              'Unless center_values is True or normalize_values is True,'
              'Neither from_type (=%s) nor to_type (=%s) can be None.' %
              (from_type, to_type))

        # Check that to_type != 'counts'
        check(to_type != 'counts', "Can only have to_type='counts' if "
                                   "from_type='counts'. Here, however, "
                                   "from_type='%s'" % from_type)

        # If converting from a probability matrix
        if from_type == 'probability':

            # ... to a weight matrix
            if to_type == 'weight':
                out_df = _probability_mat_to_weight_mat(df, background)

            # ... to an information matrix
            elif to_type == 'information':
                out_df = _probability_mat_to_information_mat(df, background)

            # This should never execute
            else:
                assert False, 'THIS SHOULD NEVER EXECUTE'

        # Otherwise, convert to probability matrix, then call function again
        else:

            # If converting from a counts matrix,
            # convert to probability matrix first
            if from_type == 'counts':
                prob_df = _counts_mat_to_probability_mat(df, pseudocount)

            # If converting from a weight matrix,
            # convert to probability matrix first
            elif from_type == 'weight':
                prob_df = _weight_mat_to_probability_mat(df, background)

            # If converting from an information matrix,
            # convert to probability matrix first
            elif from_type == 'information':
                prob_df = _information_mat_to_probability_mat(df, background)

            # This should never execute
            else:
                assert False, 'THIS SHOULD NEVER EXECUTE'

            # Now that we have the probability matrix,
            # onvert to user-specified to_type
            out_df = transform_matrix(prob_df,
                                      from_type='probability',
                                      to_type=to_type,
                                      background=background)

    # Validate and return
    out_df = validate_matrix(out_df)
    return out_df


################### END OF CODE REVIEW ON 19.03.26 #############################

def _counts_mat_to_probability_mat(df, pseudocount=1.0):
    """
    Converts a counts matrix to a probability matrix
    """

    assert pseudocount >= 0, 'Error: Pseudocount must be >= 0. '

    # Validate matrix before use
    df = validate_matrix(df)

    # Compute out_df
    out_df = df.copy()
    vals = df.values + pseudocount
    out_df.loc[:, :] = vals / vals.sum(axis=1)[:, np.newaxis]
    out_df = _normalize_matrix(out_df)

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df


def _probability_mat_to_weight_mat(df, background=None):
    """
    Converts a probability matrix to a weight matrix
    """

    # Validate matrix before use
    df = validate_probability_mat(df)

    # Get background matrix
    bg_df = _get_background_mat(df, background)

    # Compute out_df
    out_df = df.copy()
    out_df.loc[:, :] = np.log2(df+SMALL) - np.log2(bg_df+SMALL)

    # Validate and return
    out_df = validate_matrix(out_df)
    return out_df


def _weight_mat_to_probability_mat(df, background=None):
    """
    Converts a probability matrix to a weight matrix
    """

    # Validate matrix before use
    df = validate_matrix(df)

    # Get background matrix
    bg_df = _get_background_mat(df, background)

    # Compute out_df
    out_df = df.copy()
    out_df.loc[:, :] = bg_df.values * np.power(2, df.values)

    # Normalize matrix. Needed if matrix is centered.
    out_df = _normalize_matrix(out_df)

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df


def _probability_mat_to_information_mat(df, background=None):
    """
    Converts a probability matrix to an information matrix
    """

    # Validate matrix before use
    df = validate_probability_mat(df)

    # Get background matrix
    bg_df = _get_background_mat(df, background)

    # Compute out_df
    out_df = df.copy()
    fg_vals = df.values
    bg_vals = bg_df.values
    tmp_vals = fg_vals*(np.log2(fg_vals+SMALL)-np.log2(bg_vals+SMALL))
    info_vec = tmp_vals.sum(axis=1)
    out_df.loc[:, :] = fg_vals * info_vec[:, np.newaxis]

    # Validate and return
    out_df = validate_information_mat(out_df)
    return out_df


def _information_mat_to_probability_mat(df, background=None):
    """
    Converts a probability matrix to an information matrix
    """

    # Validate matrix before use
    df = validate_matrix(df)

    # Just need to normalize matrix
    out_df = _normalize_matrix(df)

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df


def _normalize_matrix(df):
    """
    Normalizes a matrix df to a probability matrix out_df
    """

    # Validate matrix
    df = validate_matrix(df)

    # Make sure all df values are zero
    #assert all(df.values.ravel() >= 0), \
    #    'Error: Some data frame entries are negative.'
    check(all(df.values.ravel() >= 0),
          'Error: Some data frame entries are negative.')

    # Check to see if values sum to one
    sums = df.sum(axis=1).values

    # If any sums are close to zero, abort
    #assert not any(np.isclose(sums, 0.0)), \
    #    'Error: some columns in df sum to nearly zero.'
    check(not any(np.isclose(sums, 0.0)),
          'Error: some columns in df sum to nearly zero.')

    out_df = df.copy()
    out_df.loc[:, :] = df.values / sums[:, np.newaxis]

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df


def _center_matrix(df):
    """
    Centers each row of a matrix about zero by subtracting out the mean.
    """

    # Validate matrix
    df = validate_matrix(df)

    # Check to see if values sum to one
    means = df.mean(axis=1).values

    # If any sums are close to zero, abort
    out_df = df.copy()
    out_df.loc[:, :] = df.values - means[:, np.newaxis]

    # Validate and return
    out_df = validate_matrix(out_df)
    return out_df


def _get_background_mat(df, background):
    """
    Creates a background matrix given a background specification. There
    are three possiblities:

    1. background is None => out_df represents a uniform background
    2. background is a vector => this vector is normalized then used as
        the entries of the rows of out_df
    3. background is a dataframe => it is then normalized and use as out_df
    """

    # Get dimensions of df
    num_pos, num_cols = df.shape

    # Create background using df as template
    out_df = df.copy()

    # If background is not specified, use uniform background
    if background is None:
        out_df.loc[:, :] = 1 / num_cols

    # If background is array-like
    elif isinstance(background, (np.ndarray, list, tuple)):
        background = np.array(background)
        assert len(background) == num_cols, \
            'Error: df and background have mismatched dimensions.'
        out_df.loc[:, :] = background[np.newaxis, :]
        out_df = _normalize_matrix(out_df)

    # If background is a dataframe
    elif isinstance(background, pd.core.frame.DataFrame):
        background = validate_matrix(background)
        assert all(df.index == background.index), \
            'Error: df and bg_mat have different indexes.'
        assert all(df.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        out_df = background.copy()
        out_df = _normalize_matrix(out_df)

    # validate as probability matrix
    out_df = validate_probability_mat(out_df)
    return out_df

@handle_errors
def iupac_to_matrix(iupac_seq, to_type='probability', **kwargs):
    """
    Generates a matrix corresponding to a (DNA) IUPAC string.

    parameters
    ----------
    iupac_seq: (str)
        An IUPAC sequence.

    to_type: (str)
        The type of matrix to convert to. Must be 'probability', 'weight',
        or 'information'

    **kwargs:
        Additional arguments to send to transform_matrix, e.g. background
        or center_values

    returns
    -------
    out_df: (dataframe)
        A matrix of the requested type.
    """

    # validate inputs
    check(isinstance(iupac_seq, str), 'type(iupac_seq) = %s must be of type str' % type(iupac_seq))
    check(isinstance(to_type, str), 'type(to_type) = %s must be of type str' % type(to_type))
    check(to_type in MATRIX_TYPES, 'invalid to_type=%s' % to_type)

    # Create counts matrix based on IUPAC string
    L = len(iupac_seq)
    cols = list('ACGT')
    index = list(range(L))
    counts_mat = pd.DataFrame(data=0.0, columns=cols, index=index)
    for i, c in enumerate(list(iupac_seq)):
        bs = iupac_dict[c]
        for b in bs:
            counts_mat.loc[i, b] = 1

    # Convert to requested type
    out_df = transform_matrix(counts_mat,
                              pseudocount=0,
                              from_type='counts',
                              to_type=to_type)
    return out_df

@handle_errors
def alignment_to_matrix(sequences,
                        to_type='counts',
                        characters_to_ignore='.-',
                        **kwargs):
    """
    Generates matrix from a sequence alignment

    parameters
    ----------
    sequences: (list of str)
        An list of sequences, all of which must be the same length

    to_type: (str)
        The type of matrix to output. Must be 'counts', 'probability',
        'weight', or 'information'

    **kwargs:
        Other arguments to pass to logomaker.transform_matrix(), e.g.
        pseudocount

    returns
    -------
    out_df: (dataframe)
        A matrix of the requested type.
    """

    # validate inputs

    # the first check is not very efficient.
    check(all(isinstance(str_index, str) for str_index in sequences),'sequences must be of type string')
    check(isinstance(to_type, str), 'type(from_type) = %s must be of type str' % type(to_type))
    TO_TYPES = {'counts','probability', 'weight', 'information'}
    check(to_type in TO_TYPES, 'invalid to_type=%s' % to_type)

    # Create array of characters at each position
    char_array = np.array([np.array(list(seq)) for seq in sequences])
    L = char_array.shape[1]

    # Get list of unique characters
    unique_characters = np.unique(char_array.ravel())
    unique_characters.sort()

    # Remove characters to ignore
    columns = [c for c in unique_characters if not c in characters_to_ignore]
    index = list(range(L))
    counts_df = pd.DataFrame(data=0, columns=columns, index=index)

    # Sum of the number of occurrances of each character at each position
    for c in columns:
        counts_df.loc[:, c] = \
            (char_array == c).astype(float).sum(axis=0).ravel()

    # Convert counts matrix to matrix of requested type
    out_df = transform_matrix(counts_df,
                              from_type='counts',
                              to_type=to_type,
                              **kwargs)
    return out_df


@handle_errors
def saliency_to_matrix(sequence, saliency):

    """
    saliency_to_matrix takes a sequence string and a saliency \n
    array and outputs a saliency dataframe. The saliency \n
    dataframe is a C by L matrix (C is characters, L is sequence \n
    length) where the elements of the matrix are hot-encoded \n
    according to the saliency list. E.g. the element saliency_{c,l} \n
    will be non-zero if character c occurs at position l, the message \n
    of the element is equal to the message of the saliency list at that \n
    position. All other elements are zero.

    example usage:

    saliency_mat = logomaker.saliency_to_matrix(sequence,saliency_values)
    logomaker.Logo(saliency_mat)

    parameters
    ----------

    sequence: (str)
        sequence for which saliency logo will be drawn

    saliency: (list)
        array of saliency message. len(saliency) == sequence

    returns
    -------
    saliency_matrix: (dataframe)
        dataframe that contains saliency values \n
        can be used directly with the Logo constructor

    """

    # validate inputs

    # validate sequence
    check(isinstance(sequence,str),'type(sequence) = %s must be of type str' % type(sequence))

    # validate background: check that it is a list or array
    check(isinstance(saliency,(type([]),np.ndarray)),
          'type(saliency) = %s must be of type list' % type(saliency))

    # check length of sequence and saliency are equal
    check(len(sequence)==len(saliency),'length of sequence and saliency list must be equal.')

    # in case the user provides an np.array, this method should still work
    saliency = list(saliency)

    # turn sequence into binary one-hot encoded matrix.
    ohe_sequence = pd.get_dummies(pd.Series(list(sequence)))

    # multiply saliency list with one-hot encoded sequence to get
    # saliency matrix or dataframe
    saliency_matrix = saliency * (ohe_sequence.T)

    # the transpose here puts positions on the x-axis and characters
    # on the y-axis, thus making it easy to use with the constructor.
    return saliency_matrix.T

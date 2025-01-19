from __future__ import division
import numpy as np
import pandas as pd

# Logomaker imports
from logomaker.src.error_handling import check, handle_errors
from logomaker.src.validate import validate_matrix

# Specifies built-in character alphabets
ALPHABET_DICT = {
    'dna': 'ACGT',
    'rna': 'ACGU',
    'protein': 'ACDEFGHIKLMNPQRSTVWY'
}

# Specifies IUPAC string transformations
IUPAC_DICT = {
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
    check(isinstance(background, (type([]), np.ndarray, pd.DataFrame)) or
          (background is None),
          'type(background) = %s must be None or array-like or a dataframe.' %
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


def _counts_mat_to_probability_mat(counts_df, pseudocount=1.0):
    """
    Converts a counts matrix to a probability matrix
    """

    # Validate matrix before use
    counts_df = validate_matrix(counts_df)

    # Check pseudocount value
    check(pseudocount >= 0, "pseudocount must be >= 0.")

    # Compute prob_df
    prob_df = counts_df.copy().astype(float)  # 25.01.19 Issue #41 - Need to specify float to avoid FutureWarning. 
    vals = counts_df.values + pseudocount
    prob_df.loc[:, :] = vals / vals.sum(axis=1)[:, np.newaxis]
    prob_df = _normalize_matrix(prob_df)

    # Validate and return
    prob_df = validate_matrix(prob_df, matrix_type='probability')
    return prob_df


def _probability_mat_to_weight_mat(prob_df, background=None):
    """
    Converts a probability matrix to a weight matrix
    """

    # Validate matrix before use
    prob_df = validate_matrix(prob_df, matrix_type='probability')

    # Get background matrix
    bg_df = _get_background_mat(prob_df, background)

    # Compute weight_df; SMALL is a regularization factor to make sure
    # np.log2 doesn't throw an error.
    weight_df = prob_df.copy()
    weight_df.loc[:, :] = np.log2(prob_df + SMALL) - np.log2(bg_df + SMALL)

    # Validate and return
    weight_df = validate_matrix(weight_df)
    return weight_df


def _weight_mat_to_probability_mat(weight_df, background=None):
    """
    Converts a weight matrix to a probability matrix
    """

    # Validate matrix before use
    weight_df = validate_matrix(weight_df)

    # Get background matrix
    bg_df = _get_background_mat(weight_df, background)

    # Compute prob_df
    prob_df = weight_df.copy()
    prob_df.loc[:, :] = bg_df.values * np.power(2, weight_df.values)

    # Normalize matrix. Needed if matrix is centered.
    prob_df = _normalize_matrix(prob_df)

    # Validate and return
    prob_df = validate_matrix(prob_df, matrix_type='probability')
    return prob_df


def _probability_mat_to_information_mat(prob_df, background=None):
    """
    Converts a probability matrix to an information matrix
    """

    # Validate matrix before use
    prob_df = validate_matrix(prob_df, matrix_type='probability')

    # Get background matrix
    bg_df = _get_background_mat(prob_df, background)

    # Compute info_df
    info_df = prob_df.copy()
    fg_vals = prob_df.values
    bg_vals = bg_df.values
    tmp_vals = fg_vals * (np.log2(fg_vals + SMALL) - np.log2(bg_vals + SMALL))
    info_vec = tmp_vals.sum(axis=1)
    info_df.loc[:, :] = fg_vals * info_vec[:, np.newaxis]

    # Validate and return
    info_df = validate_matrix(info_df, matrix_type='information')
    return info_df


def _information_mat_to_probability_mat(info_df, background=None):
    """
    Converts an information matrix to an probability matrix
    """

    # Validate matrix before use
    info_df = validate_matrix(info_df, matrix_type='information')

    # Get background matrix
    bg_df = _get_background_mat(info_df, background)

    # This is a little subtle. If any rows of info_df are zero,
    # _normalize_matrix() cannot be relied on. But in this case,
    # we know that the corresponding row of prob_df should just
    # reflect background. So we replace rows that are zero with the
    # corresponding bg_df row.
    zero_indices = np.isclose(info_df.sum(axis=1), 0.0)
    info_df.loc[zero_indices, :] = bg_df.loc[zero_indices, :]

    # Just need to normalize matrix
    prob_df = _normalize_matrix(info_df)

    # Validate and return
    prob_df = validate_matrix(prob_df, matrix_type='probability')
    return prob_df


def _normalize_matrix(df):
    """
    Normalizes a matrix df to a probability matrix prob_df
    """

    # Validate matrix
    df = validate_matrix(df)

    # Make sure all df values are greater than or equal to zero
    check(all(df.values.ravel() >= 0),
          'Some data frame entries are negative.')

    # Check to see if values sum to one
    sums = df.sum(axis=1).values

    # If any sums are close to zero, abort
    check(not any(np.isclose(sums, 0.0)),
          'Some columns in df sum to nearly zero.')

    # Create normalized version of input matrix
    prob_df = df.copy()
    prob_df.loc[:, :] = df.values / sums[:, np.newaxis]

    # Validate and return probability matrix
    prob_df = validate_matrix(prob_df, matrix_type='probability')
    return prob_df


def _center_matrix(df):
    """
    Centers each row of a matrix about zero by subtracting out the mean.
    """

    # Validate matrix
    df = validate_matrix(df)

    # Compute mean value of each row
    means = df.mean(axis=1).values

    # Subtract out means
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
        the entries of the rows of out_df. Vector must be the same length
        as the number of columns in df
    3. background is a dataframe => it is then normalized and use as out_df.
        In this case, background must have the same rows and cols as df
    """

    # Get dimensions of df
    num_pos, num_cols = df.shape

    # Create background using df as template
    bg_df = df.copy()

    # If background is not specified, use uniform background
    if background is None:
        bg_df.loc[:, :] = 1 / num_cols

    # If background is array-like
    elif isinstance(background, (np.ndarray, list, tuple)):
        background = np.array(background)
        check(len(background) == num_cols,
              'df and background have mismatched dimensions.')
        bg_df.loc[:, :] = background[np.newaxis, :]
        bg_df = _normalize_matrix(bg_df)

    # If background is a dataframe
    elif isinstance(background, pd.core.frame.DataFrame):
        bg_df = validate_matrix(background)
        check(all(df.index == bg_df.index),
              'Error: df and bg_mat have different indexes.')
        check(all(df.columns == bg_df.columns),
              'Error: df and bg_mat have different columns.')
        bg_df = _normalize_matrix(bg_df)

    # validate as probability matrix
    bg_df = validate_matrix(bg_df, matrix_type='probability')
    return bg_df


@handle_errors
def alignment_to_matrix(sequences,
                        counts=None,
                        to_type='counts',
                        background=None,
                        characters_to_ignore='.-',
                        center_weights=False,
                        pseudocount=1.0):
    """
    Generates matrix from a sequence alignment

    parameters
    ----------
    sequences: (list of strings)
        A list of sequences, all of which must be the same length

    counts: (None or list of numbers)
        If not None, must be a list of numbers the same length os sequences,
        containing the (nonnegative) number of times that each sequence was
        observed. If None, defaults to 1.

    to_type: (str)
        The type of matrix to output. Must be 'counts', 'probability',
        'weight', or 'information'

    background: (array, or df)
        Specification of background probabilities. If array, should be the
        same length as df.columns and correspond to the probability of each
        column's character. If df, should be a probability matrix the same
        shape as df.

    characters_to_ignore: (str)
        Characters to ignore within sequences. This is often needed when
        creating matrices from gapped alignments.

    center_weights: (bool)
        Whether to subtract the mean of each row, but only if to_type=='weight'.

    pseudocount: (number >= 0.0)
        Pseudocount to use when converting from counts to probabilities.

    returns
    -------
    out_df: (dataframe)
        A matrix of the requested type.
    """

    # validate inputs

    # Make sure sequences is list-like
    check(isinstance(sequences, (list, tuple, np.ndarray, pd.Series)),
          'sequences must be a list, tuple, np.ndarray, or pd.Series.')
    sequences = list(sequences)

    # Make sure sequences has at least 1 element
    check(len(sequences) > 0, 'sequences must have length > 0.')

    # Make sure all elements are sequences
    check(all(isinstance(seq, str) for seq in sequences),
          'sequences must all be of type string')

    # validate characters_to_ignore
    check(isinstance(characters_to_ignore, str),
          'type(seq) = %s must be of type str' % type(characters_to_ignore))

    # validate center_weights
    check(isinstance(center_weights, bool),
          'type(center_weights) = %s; must be bool.' % type(center_weights))

    # Get sequence length
    L = len(sequences[0])

    # Make sure all sequences are the same length
    check(all([len(s) == L for s in sequences]),
          'all elements of sequences must have the same length.')

    # validate counts as list-like
    check(isinstance(counts, (list, tuple, np.ndarray, pd.Series)) or
          (counts is None),
          'counts must be None or a list, tuple, np.ndarray, or pd.Series.')

    # make sure counts has the same length as sequences
    if counts is None:
        counts = np.ones(len(sequences))
    else:
        check(len(counts) == len(sequences),
              'counts must be the same length as sequences;'
              'len(counts) = %d; len(sequences) = %d' %
              (len(counts), len(sequences)))
        
    # 2025.01.19 Fix for Issue #25 - Cast counts to numpy array
    counts = np.array(counts, dtype=int)

    # validate background
    check(isinstance(background, (type([]), np.ndarray, pd.DataFrame)) or
          (background is None),
          'type(background) = %s must be None or array-like or a dataframe.' %
          type(background))

    # Define valid types
    valid_types = MATRIX_TYPES.copy()

    # Check that to_type is valid
    check(to_type in valid_types,
          'to_type=%s; must be in %s' % (to_type, valid_types))

    # Create a 2D array of characters
    char_array = np.array([np.array(list(seq)) for seq in sequences])

    # Get list of unique characters
    unique_characters = np.unique(char_array.ravel())
    unique_characters.sort()

    # Remove characters to ignore
    columns = [c for c in unique_characters if not c in characters_to_ignore]
    index = list(range(L))
    counts_df = pd.DataFrame(data=0, columns=columns, index=index)

    # Sum of the number of occurrences of each character at each position
    for c in columns:
        tmp_mat = (char_array == c).astype(float) * counts[:, np.newaxis]
        counts_df.loc[:, c] = tmp_mat.sum(axis=0).T

    # Convert counts matrix to matrix of requested type
    out_df = transform_matrix(counts_df,
                              from_type='counts',
                              to_type=to_type,
                              pseudocount=pseudocount,
                              background=background)

    # Center values only if center_weights is True and to_type is 'weight'
    if center_weights and to_type == 'weight':
        out_df = transform_matrix(out_df, center_values=True)

    return out_df


@handle_errors
def sequence_to_matrix(seq,
                       cols=None,
                       alphabet=None,
                       is_iupac=False,
                       to_type='probability',
                       center_weights=False):
    """
    Generates a matrix from a sequence. With default keyword arguments,
    this is a one-hot-encoded version of the sequence provided. Alternatively,
    is_iupac=True allows users to get matrix models based in IUPAC motifs.

    parameters
    ----------

    seq: (str)
        Sequence from which to construct matrix.

    cols: (str or array-like or None)
        The characters to use for the matrix columns. If None, cols is
        constructed from the unqiue characters in seq. Overriden by alphabet
        and is_iupac.

    alphabet: (str or None)
        The alphabet used to determine the columns of the matrix.
        Options are: 'dna', 'rna', 'protein'. Ignored if None. Overrides cols.

    is_iupac: (bool)
        If True, it is assumed that the sequence represents an IUPAC DNA
        string. In this case, cols is overridden, and alphabet must be None.

    to_type: (str)
        The type of matrix to output. Must be 'probability', 'weight',
        or 'information'

    center_weights: (bool)
        Whether to subtract the mean of each row, but only if to_type='weight'.

    returns
    -------
    seq_df: (dataframe)
        the matrix returned to the user.
    """

    # Define valid types
    valid_types = MATRIX_TYPES.copy()
    valid_types.remove('counts')

    # validate seq
    check(isinstance(seq, str),
          'type(seq) = %s must be of type str' % type(seq))

    # validate center_weights
    check(isinstance(center_weights, bool),
          'type(center_weights) = %s; must be bool.' % type(center_weights))

    # If cols is None, set to list of unique characters in sequence
    if cols is None:
        cols = list(set(seq))
        cols.sort()

    # Otherwise, validate cols
    else:
        cols_types = (str, list, set, np.ndarray)
        check(isinstance(cols, cols_types),
              'cols = %s must be None or a string, set, list, or np.ndarray')

    # If alphabet is specified, override cols
    if alphabet is not None:

        # Validate alphabet
        valid_alphabets = list(ALPHABET_DICT.keys())
        check(alphabet in valid_alphabets,
              'alphabet = %s; must be in %s.' % (alphabet, valid_alphabets))

        # Set cols
        cols = list(ALPHABET_DICT[alphabet])

    # validate to_type
    check(to_type in valid_types,
          'invalid to_type=%s; to_type must be in %s' % (to_type, valid_types))

    # validate is_iupac
    check(isinstance(is_iupac, bool),
          'type(is_iupac) = %s; must be bool.' % type(is_iupac))

    # If is_iupac, override alphabet and cols
    if is_iupac:

        # Check that alphabet has not been specified
        check(alphabet is None, 'must have alphabet=None if is_iupac=True')
        cols = list(ALPHABET_DICT['dna'])

    # Initialize counts dataframe
    L = len(seq)
    index = list(range(L))
    counts_df = pd.DataFrame(data=0.0, columns=cols, index=index)

    # If is_iupac, fill counts_df:
    if is_iupac:

        # Get list of valid IUPAC characters
        iupac_characters = list(IUPAC_DICT.keys())

        # Iterate over sequence positions
        for i, c in enumerate(seq):

            # Check that c is in the set of valid IUPAC characters
            check(c in iupac_characters,
                  'character %s at position %d is not a valid IUPAC character;'
                  'must be one of %s' %
                  (c, i, iupac_characters))

            # Fill in a count for each possible base
            bs = IUPAC_DICT[c]
            for b in bs:
                counts_df.loc[i, b] = 1.0

    # Otherwise, fill counts the normal way
    else:

        # Iterate over sequence positions
        for i, c in enumerate(seq):

            # Check that c is in columns
            check(c in cols,
                  'character %s at position %d is not in cols=%s' %
                  (c, i, cols))

            # Increment counts_df
            counts_df.loc[i, c] = 1.0

    # Convert to requested type
    out_df = transform_matrix(counts_df,
                              pseudocount=0,
                              from_type='counts',
                              to_type=to_type)

    # Center values only if center_weights is True and to_type is 'weight'
    if center_weights and to_type == 'weight':
        out_df = transform_matrix(out_df, center_values=True)

    return out_df


@handle_errors
def saliency_to_matrix(seq, values, cols=None, alphabet=None):
    """
    Takes a sequence string and an array of values values and outputs a
    values dataframe. The returned dataframe is a L by C matrix where C is
    the number ofcharacters and L is sequence length.  If matrix is denoted as
    S, i indexes positions and c indexes characters, then S_ic will be non-zero
    (equal to the value in the values array at position p) only if character c
    occurs at position p in sequence. All other elements of S are zero.

    example usage:

    saliency_mat = logomaker.saliency_to_matrix(sequence,values)
    logomaker.Logo(saliency_mat)

    parameters
    ----------

    seq: (str or array-like list of single characters)
        sequence for which values matrix is constructed

    values: (array-like list of numbers)
        array of values values for each character in sequence

    cols: (str or array-like or None)
        The characters to use for the matrix columns. If None, cols is
        constructed from the unqiue characters in seq. Overridden by alphabet
        and is_iupac.

    alphabet: (str or None)
        The alphabet used to determine the columns of the matrix.
        Options are: 'dna', 'rna', 'protein'. Ignored if None. Overrides cols.

    returns
    -------
    saliency_df: (dataframe)
        values matrix in the form of a dataframe

    """

    # try to convert seq to str; throw exception if fail
    if isinstance(seq, (list, np.ndarray, pd.Series)):
        try:
            seq = ''.join([str(x) for x in seq])
        except:
            check(False, 'could not convert %s to type str' % repr(str))
    else:
        try:
            seq = str(seq)
        except:
            check(False, 'could not convert %s to type str' % repr(str))

    # validate seq
    check(isinstance(seq, str),
          'type(seq) = %s must be of type str' % type(seq))

    # validate values: check that it is a list or array
    check(isinstance(values, (type([]), np.ndarray, pd.Series)),
          'type(values) = %s must be of type list' % type(values))

    # cast values as a list just to be sure what we're working with
    values = list(values)

    # check length of seq and values are equal
    check(len(seq) == len(values),
          'length of seq and values list must be equal.')

    # If cols is None, set to list of unique characters in sequence
    if cols is None:
        cols = list(set(seq))
        cols.sort()

    # Otherwise, validate cols
    else:
        cols_types = (str, list, set, np.ndarray)
        check(isinstance(cols, cols_types),
              'cols = %s must be None or a string, set, list, or np.ndarray')

        # perform additional checks to validate cols
        check(len(set(cols))==len(set(seq)),
              'length of set of unique characters must be equal for "cols " and "seq"')
        check(set(cols) == set(seq),
              'unique characters for "cols" and "seq" must be equal.')

    # If alphabet is specified, override cols
    if alphabet is not None:

        # Validate alphabet
        valid_alphabets = list(ALPHABET_DICT.keys())
        check(alphabet in valid_alphabets,
              'alphabet = %s; must be in %s.' % (alphabet, valid_alphabets))

        # Set cols
        cols = list(ALPHABET_DICT[alphabet])

    # turn seq into binary one-hot encoded matrix.
    ohe_sequence = sequence_to_matrix(seq, cols=cols)

    # multiply values list with one-hot encoded seq to get
    # values matrix or dataframe
    saliency_df = ohe_sequence.copy()
    saliency_df.loc[:, :] = ohe_sequence.values * \
                            np.array(values)[:, np.newaxis]

    return saliency_df


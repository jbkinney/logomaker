from __future__ import division
import numpy as np
import pandas as pd
import pdb

# from validate import validate_matrix, validate_probability_mat, iupac_dict
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

def transform_matrix(df, from_type, to_type,
                     background=None,
                     pseudocount=1,
                     center=False):
    """
    Transforms a matrix of one type into a matrix of another type.

    i = position
    c, d = character

    l = pseudocount
    C = number of characters

    N_ic = counts matrix element
    P_ic = probability matrix element
    Q_ic = background probability matrix element
    W_ic = weight matrix element
    I_ic = information matrix element

    counts -> probability:
        P_ic = (N_ic + l)/(N_i + C*l), N_i = sum_c(N_ic)

    probability -> weight:
        W_ic = log_2(P_ic / Q_ic)

    weight -> probability:
        P_ic = Q_ic * 2^(W_ic)

    probability -> information:
        I_ic = P_ic * sum_d(P_id * log2(P_id / W_id))

    information -> probability:
        P_ic = I_ic / sum_d(I_id)


    parameters
    ----------

    df: (dataframe)
        The matrix to be transformed.

    from_type: (str)
        Type of input matrix. Must be one of 'counts', 'probability',
        'weight', or 'information'.

    to_type: (str)
        Type of output matrix. Must be one of 'probability', 'weight', or
        'information'. Can NOT be 'counts'.

    background: (array, or df)
        Specification of background probabilities. If array, should be the
        same length as df.columns and correspond to the probability of each
        column's character. If df, should be a probability matrix the same
        shape as df.

    pseudocount: (number >= 0)
        Pseudocount to use when transforming from a count matrix to a
        probability matrix.

    center: (bool)
        Whether to center_values the output matrix. Note: this only works when
        to_type = 'weight', as centering a matrix doesn't make sense otherwise.

    returns
    -------
    out_df: (dataframe)
        Transformed matrix
    """

    FROM_TYPES = {'counts', 'probability', 'weight', 'information'}
    TO_TYPES = {'probability', 'weight', 'information'}

    # Check that matrix is valid
    df = validate_matrix(df)

    # If to_type == from_type, just return matrix
    if from_type == to_type:
        out_df = df.copy()

    else:
        assert from_type in FROM_TYPES, \
            'Error: invalid from_type=%s' % from_type

        assert to_type in TO_TYPES, \
            'Error: invalid to_type="%s"' % from_type

        # If converting from a probability matrix
        if from_type == 'probability':

            if to_type == 'weight':
                out_df = _probability_mat_to_weight_mat(df, background)

            elif to_type == 'information':
                out_df = _probability_mat_to_information_mat(df, background)

        # Otherwise, convert to probability matrix, then call function again
        else:

            # If converting from a counts matrix
            if from_type == 'counts':
                prob_df = _counts_mat_to_probability_mat(df, pseudocount)

            # If converting from a weight matrix
            elif from_type == 'weight':
                prob_df = _weight_mat_to_probability_mat(df, background)

            elif from_type == 'information':
                prob_df = _information_mat_to_probability_mat(df, background)

            else:
                assert False, 'THIS SHOULD NEVER HAPPEN'

            out_df = transform_matrix(prob_df,
                                      from_type='probability',
                                      to_type=to_type,
                                      background=background)

    # Check if user wishes to center_values the matrix
    if center:
        assert to_type == 'weight', \
            'Error: the option center_values=True is only compatible with ' + \
            'to_type == "weight"'
        out_df = center_matrix(out_df)


    # Validate and return
    out_df = validate_matrix(out_df)
    return out_df


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
    out_df = normalize_matrix(out_df)

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
    out_df = normalize_matrix(out_df)

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df


# Needed only for display purposes
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


# Needed only for display purposes
def _information_mat_to_probability_mat(df, background=None):
    """
    Converts a probability matrix to an information matrix
    """

    # Validate matrix before use
    df = validate_matrix(df)

    # Just need to normalize matrix
    out_df = normalize_matrix(df)

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df


# Normalize a data frame of probabilities
def normalize_matrix(df):
    """
    Normalizes a matrix df to a probability matrix out_df
    """

    # Validate matrix
    df = validate_matrix(df)

    # Make sure all df values are zero
    assert all(df.values.ravel() >= 0), \
        'Error: Some data frame entries are negative.'

    # Check to see if values sum to one
    sums = df.sum(axis=1).values

    # If any sums are close to zero, abort
    assert not any(np.isclose(sums, 0.0)), \
        'Error: some columns in df sum to nearly zero.'
    out_df = df.copy()
    out_df.loc[:, :] = df.values / sums[:, np.newaxis]

    # Validate and return
    out_df = validate_probability_mat(out_df)
    return out_df



# Normalize a data frame of probabilities
def center_matrix(df):
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
        out_df = normalize_matrix(out_df)

    # If background is a dataframe
    elif isinstance(background, pd.core.frame.DataFrame):
        background = validate_matrix(background)
        assert all(df.index == background.index), \
            'Error: df and bg_mat have different indexes.'
        assert all(df.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        out_df = background.copy()
        out_df = normalize_matrix(out_df)

    # validate as probability matrix
    out_df = validate_probability_mat(out_df)
    return out_df


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


# from Bio import SeqIO
# def load_alignment(fasta_file=None,
#                    csv_file=None,
#                    seq_col=None,
#                    ct_col=None,
#                    csv_kwargs={},
#                    sequences=None,
#                    sequence_counts=None,
#                    sequence_type=None,
#                    characters=None,
#                    positions=None,
#                    ignore_characters='.-',
#                    occurance_threshold=0):
#     # If loading file name
#     if fasta_file is not None:
#
#         # Load sequences using SeqIO
#         sequences = [str(record.seq) for record in \
#                      SeqIO.parse(fasta_file, "fasta")]
#
#     # If loading from a CSV file
#     elif csv_file is not None:
#
#         # Make sure that seq_col is specified
#         assert seq_col is not None, \
#             'Error: seq_col is None. If csv_file is specified, seq_col must' \
#             + ' also be specified'
#
#         # Load csv file as a dataframe
#         df = pd.read_csv(csv_file, **csv_kwargs)
#         # df = df.fillna(csv_fillna)
#         df = df.fillna(csv_file)
#
#         # Make sure that seq_col is in df
#         assert seq_col in df.columns, \
#             ('Error: seq_col %s is not in the columns %s read from '
#              + 'csv_file %s') % (seq_col, df.columns, csv_file)
#
#         # Get sequences
#         sequences = df[seq_col].values
#
#         # Optionally set sequence_counts
#         if ct_col is not None:
#             # Make sure that seq_col is in df
#             assert seq_col in df.columns, \
#                 ('Error: ct_col %s is not None, but neither is it in the '
#                  + 'columns %s loaded from csv_file'
#                  + ' file %s') % (ct_col, df.columns, csv_file)
#
#             # Load sequences counts
#             sequence_counts = df[ct_col].values
#
#     # Make sure that, whatever was passed, sequences is set
#     assert sequences is not None, \
#         'Error: either fasta_file or sequences must not be None.'
#
#     # Get seq length
#     L = len(sequences[0])
#     # print('debugging:')
#     # print(type(sequences))
#     # print(L)
#     # print(np.shape(sequences))
#     assert all([len(seq) == L for seq in sequences]), \
#         'Error: not all sequences have length %d.' % L
#
#     # Get counts list
#     if sequence_counts is None:
#         assert len(sequences) > 0, 'Error: sequences is empty'
#         counts_array = np.ones(len(sequences))
#     else:
#         assert len(sequence_counts) == len(sequences), \
#             'Error: sequence_counts is not the same length as sequences'
#         counts_array = np.array(sequence_counts)
#
#     # If positions is not specified by user, make it
#     if positions is not None:
#         assert len(positions) == L, 'Error: positions, if passed, must be ' + \
#                                     'same length as sequences.'
#     else:
#         positions = range(L)
#
#     # Create counts matrix
#     counts_mat = pd.DataFrame(index=positions).fillna(0)
#
#     # Create array of characters at each position
#     char_array = np.array([np.array(list(seq)) for seq in sequences])
#
#     # Get list of unique characters
#     unique_characters = np.unique(char_array.ravel())
#     unique_characters.sort()
#
#     # Sum of the number of occurances of each character at each position
#     for c in unique_characters:
#         v = (char_array == c).astype(float)
#         v *= counts_array[:, np.newaxis]
#         counts_mat.loc[:, c] = v.sum(axis=0).ravel()
#
#     # Filter columns
#     counts_mat = filter_columns(counts_mat,
#                                 sequence_type=sequence_type,
#                                 characters=characters,
#                                 ignore_characters=ignore_characters)
#
#     # Remove rows with too few counts
#     position_counts = counts_mat.values.sum(axis=1)
#     max_counts = position_counts.max()
#     positions_to_keep = position_counts >= occurance_threshold * max_counts
#     counts_mat = counts_mat.loc[positions_to_keep, :]
#
#     # Name index
#     counts_mat.index.name = 'pos'
#
#     return counts_mat


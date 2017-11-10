from __future__ import division
import numpy as np
import pandas as pd
import re
import pdb
from Bio import SeqIO
import warnings

# Set constants
SMALL = 1E-6

# Character lists
DNA = list('ACGT')
RNA = list('ACGU')
dna = [c.lower() for c in DNA]
rna = [c.lower() for c in DNA]
PROTEIN = list('RKDENQSGHTAPYVMCLFIW')
protein = [c.lower() for c in PROTEIN]
PROTEIN_STOP = PROTEIN + ['*']
protein_stop = protein + ['*']

# Character transformations dictionaries
to_DNA = {'a': 'A', 'c': 'C', 'g': 'G', 't': 'T', 'U': 'T', 'u': 'T'}
to_dna = {'A': 'a', 'C': 'c', 'G': 'g', 'T': 't', 'U': 't', 'u': 't'}
to_RNA = {'a': 'A', 'c': 'C', 'g': 'G', 't': 'U', 'T': 'U', 'u': 'U'}
to_rna = {'A': 'a', 'C': 'c', 'G': 'g', 'T': 'u', 't': 'u', 'U': 'u'}
to_PROTEIN = dict(zip(protein, PROTEIN))
to_protein = dict(zip(PROTEIN, protein))

from validate import validate_mat, validate_probability_mat, iupac_dict

def transform_mat(matrix, to_type, from_type=None, background=None,
                  pseudocount=1, enrichment_logbase=2,
                  center_columns=True, information_units='bits'):
    '''
    transform_mat(): transforms a matrix of one type into another.
    :param matrix: input matrix, in data frame format
    :param from_type: type of matrix to transform from
    :param to_type: type of matrix to transform to
    :param background: background base frequencies
    :param gamma: parameter for converting energies to frequencies
    :param pseudocount: psuedocount used to convert count_mat to freq_mat
    :return: out_mat, a matrix in data frame format of the specified output type
    '''

    # Check that matrix is valid
    if (from_type is None) and (to_type is None):

        # Center values if requested
        matrix = validate_mat(matrix, allow_nan=True)
        if center_columns:
            means = matrix.mean(axis=1, skipna=True).values
            means[np.isnan(means)] = 0.0
            out_mat = matrix.copy()
            out_mat.iloc[:, :] = matrix.values - means[:, np.newaxis]
        else:
            out_mat = matrix.copy()

        return out_mat

    else:
        matrix = validate_mat(matrix, allow_nan=False)

    # If not changing types, just return matrix
    if from_type == to_type:
        return matrix

    # Create background matrix
    bg_mat = set_bg_mat(background=background, matrix=matrix)

    # Compute freq_mat from from_type
    if from_type == 'probability':
        probability_mat = validate_probability_mat(matrix)

    elif from_type == 'counts':
        probability_mat = \
            counts_mat_to_probability_mat(matrix, pseudocount=pseudocount)
    else:
        assert False, 'Error! from_type %s is invalid.'%from_type

    # Compute out_mat from freq_mat
    if to_type == 'counts':
        if from_type == 'counts':
            out_mat = matrix
        else:
            assert False, 'Cannot convert from %s to count_mat' % \
                          from_type

    elif to_type == 'probability':
        out_mat = probability_mat

    elif to_type == 'enrichment':
        out_mat = \
            probability_mat_to_enrichment_mat(probability_mat, bg_mat,
                                              base=enrichment_logbase,
                                              centering=center_columns)

    elif to_type == 'information':
        out_mat = probability_mat_to_information_mat(probability_mat, bg_mat,
                                                     units=information_units)

    else:
        assert False, 'Error! to_type %s is invalid.'%to_type

    # Return out_mat
    return out_mat

def counts_mat_to_probability_mat(count_mat, pseudocount=1):
    '''
    Converts a count_mat to a freq_mat
    '''
    # Validate mat before use
    count_mat = validate_mat(count_mat)

    # Compute freq_mat
    freq_mat = count_mat.copy()
    vals = count_mat.values + pseudocount
    freq_mat.loc[:,:] = vals / vals.sum(axis=1)[:,np.newaxis]
    freq_mat = normalize_probability_matrix(freq_mat)

    # Validate and return
    freq_mat = validate_probability_mat(freq_mat)
    return freq_mat


def probability_mat_to_enrichment_mat(freq_mat, bg_mat, base=2,
                                      centering=True):
    '''
    Converts a probability matrix to an enrichment matrix
    '''
    # Validate mat before use
    freq_mat = validate_probability_mat(freq_mat + SMALL)

    # Make sure base is a float

    # Compute weight_mat
    weight_mat = freq_mat.copy()
    weight_mat.loc[:, :] = np.log2(freq_mat / bg_mat)/np.log2(base)

    # Center if requested
    if centering:
        weight_mat.loc[:, :] = \
            weight_mat.values - weight_mat.values.mean(axis=1)[:, np.newaxis]

    # Validate and return
    weight_mat = validate_mat(weight_mat)
    return weight_mat

# Needed only for display purposes
def probability_mat_to_information_mat(freq_mat, bg_mat, units='bits'):
    '''
    Converts a prob df to an information df
    '''
    # Set units
    if units=='bits':
        multiplier = 1
    elif units=='nats':
        multiplier = 1./np.log2(np.e)
    else:
        assert False, 'Error: invalid selection for units = %s' % units

    # Validate mat before use
    freq_mat = validate_probability_mat(freq_mat + SMALL)
    info_mat = freq_mat.copy()

    info_list = (freq_mat.values * multiplier *
                 np.log2(freq_mat.values/bg_mat.values)).sum(axis=1)

    info_mat.loc[:, :] = freq_mat.values*info_list[:,np.newaxis]

    # Validate and return
    info_mat = validate_mat(info_mat)
    return info_mat


# Normalize a data frame of probabilities
def normalize_probability_matrix(freq_mat, regularize=True):
    mat = freq_mat.copy()
    assert all(np.ravel(mat.values) >= 0), \
        'Error: Some data frame entries are negative.'
    mat.loc[:, :] = mat.values / mat.values.sum(axis=1)[:, np.newaxis]
    if regularize:
        mat.loc[:, :] += SMALL
    return mat


def set_bg_mat(background, matrix):
    '''
    Creates a background matrix given a background specification and matrix
    with the right rows and columns
     '''
    num_pos, num_cols = matrix.shape

    # Create background from scratch
    if background is None:
        new_bg_mat = matrix.copy()
        new_bg_mat.loc[:, :] = 1 / num_cols

    # Expand rows of list or numpy array background
    elif type(background) == list or type(background) == np.ndarray:
        assert len(background) == matrix.shape[1], \
            'Error: df and background have mismatched dimensions.'
        new_bg_mat = matrix.copy()
        background = np.array(background).ravel()
        new_bg_mat.loc[:, :] = background

    elif type(background) == dict:
        assert set(background.keys()) == set(matrix.columns), \
            'Error: df and background have different columns.'
        new_bg_mat = matrix.copy()
        for i in new_bg_mat.index:
            new_bg_mat.loc[i, :] = background

    # Expand single-row background data frame
    elif type(background) == pd.core.frame.DataFrame and \
                    background.shape == (1, num_cols):
        assert all(matrix.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        new_bg_mat = matrix.copy()
        new_bg_mat.loc[:, :] = background.values.ravel()

    # Use full background dataframe
    elif type(background) == pd.core.frame.DataFrame and \
                    len(background.index) == len(matrix.index):
        assert all(matrix.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        new_bg_mat = background.copy()

    else:
        assert False, 'Error: bg_mat and df are incompatible'

    # Match indices of new_bg_mat to matrix
    new_bg_mat['pos'] = matrix.index.astype(int)
    new_bg_mat.set_index('pos', inplace=True, drop=True)

    # Normalize new_bg_mat as probability matrix
    new_bg_mat = normalize_probability_matrix(new_bg_mat)
    return new_bg_mat


def load_matrix(csv_file, csv_kwargs={}):
    """ Loads a matrix from a csv file """

    # Make sure that a file name is specified
    assert csv_file is not None, 'Error: csv_file is not specified.'
    matrix = pd.read_csv(csv_file, **csv_kwargs)
    matrix = validate_mat(matrix)
    return matrix


def load_alignment(fasta_file=None,
                   csv_file=None,
                   seq_col=None,
                   ct_col=None,
                   csv_kwargs={},
                   sequences=None,
                   sequence_counts=None,
                   sequence_type=None,
                   characters=None,
                   positions=None,
                   ignore_characters='.-',
                   occurance_threshold=0):

    # If loading file name
    if fasta_file is not None:

        # Load sequences using SeqIO
        sequences = [str(record.seq) for record in \
                     SeqIO.parse(fasta_file, "fasta")]

    # If loading from a CSV file
    elif csv_file is not None:

        # Make sure that seq_col is specified
        assert seq_col is not None, \
            'Error: seq_col is None. If csv_file is specified, seq_col must' \
            + ' also be specified'

        # Load csv file as a dataframe
        df = pd.read_csv(csv_file, **csv_kwargs)
        df = df.fillna(csv_fillna)

        # Make sure that seq_col is in df
        assert seq_col in df.columns, \
            ('Error: seq_col %s is not in the columns %s read from '
            + 'csv_file %s') % (seq_col, df.columns, csv_file)

        # Get sequences
        sequences = df[seq_col].values

        # Optionally set sequence_counts
        if ct_col is not None:

            # Make sure that seq_col is in df
            assert seq_col in df.columns, \
                ('Error: ct_col %s is not None, but neither is it in the '
                 + 'columns %s loaded from csv_file'
                 + ' file %s') % (ct_col, df.columns, csv_file)

            # Load sequences counts
            sequence_counts = df[ct_col].values

    # Make sure that, whatever was passed, sequences is set
    assert sequences is not None, \
        'Error: either fasta_file or sequences must not be None.'

    # Get seq length
    L = len(sequences[0])
    assert all([len(seq) == L for seq in sequences]), \
        'Error: not all sequences have length %d.' % L

    # Get counts list
    if sequence_counts is None:
        assert len(sequences) > 0, 'Error: sequences is empty'
        counts_array = np.ones(len(sequences))
    else:
        assert len(sequence_counts) == len(sequences), \
            'Error: sequence_counts is not the same length as sequences'
        counts_array = np.array(sequence_counts)

    # If positions is not specified by user, make it
    if positions is not None:
        assert len(positions) == L, 'Error: positions, if passed, must be '+\
                                    'same length as sequences.'
    else:
        positions = range(L)

    # Create counts matrix
    counts_mat = pd.DataFrame(index=positions).fillna(0)

    # Create array of characters at each position
    char_array = np.array([np.array(list(seq)) for seq in sequences])

    # Get list of unique characters
    unique_characters = np.unique(char_array.ravel())
    unique_characters.sort()

    # Sum of the number of occurances of each character at each position
    for c in unique_characters:
        v = (char_array == c).astype(float)
        v *= counts_array[:, np.newaxis]
        counts_mat.loc[:, c] = v.sum(axis=0).ravel()

    # Filter columns
    counts_mat = filter_columns(counts_mat,
                                sequence_type=sequence_type,
                                characters=characters,
                                ignore_characters=ignore_characters)

    # Remove rows with too few counts
    position_counts = counts_mat.values.sum(axis=1)
    max_counts = position_counts.max()
    positions_to_keep = position_counts >= occurance_threshold * max_counts
    counts_mat = counts_mat.loc[positions_to_keep, :]

    # Name index
    counts_mat.index.name = 'pos'

    return counts_mat


def filter_columns(matrix,
                   sequence_type=None,
                   characters=None,
                   ignore_characters='.-'):

    # Rename characters if appropriate
    if sequence_type is None:
        translation_dict = {}
    elif sequence_type == 'dna':
        translation_dict = to_dna
    elif sequence_type == 'DNA':
        translation_dict = to_DNA
    elif sequence_type == 'rna':
        translation_dict = to_rna
    elif sequence_type == 'RNA':
        translation_dict = to_RNA
    elif sequence_type == 'protein':
        translation_dict = to_protein
    elif sequence_type == 'PROTEIN':
        translation_dict = to_PROTEIN
    else:
        message = \
            "Could not interpret sequence_type = %s. Columns not filtered." %\
            repr(sequence_type)
        warnings.warn(message, UserWarning)
        translation_dict = {}

    # Copy matrix as a precaution
    # The resulting copy will undergo a number of modifications
    matrix = matrix.copy()

    # Get current list of columns
    cols = list(matrix.columns)

    # Remove any columns to ignore
    for char in ignore_characters:
        if char in cols:
            del matrix[char]

    # If manually restricting to specific characters, do it:
    if characters is not None:
        new_columns = [c for c in characters if c in matrix.columns]
        matrix = matrix.loc[:, new_columns]

    # Otherwise performing translation, do it
    elif len(translation_dict) > 0:

        # Union of keys and values.
        allowed_chars = set(translation_dict.values()) | \
                        set(translation_dict.keys())
        invalid_chars = list(set(matrix.columns) - allowed_chars)
        invalid_chars.sort()

        # assert len(invalid_chars)==0, \
        #     'Matrix contains invalid characters %s ' % \
        #     repr(list(invalid_chars)) + \
        #     ' for sequence_type %s ' % sequence_type

        # Remove invalid characters, giving a warning while doing so
        matrix.drop(invalid_chars, axis=1, inplace=True)
        message = ("Invalid matrix columns %s for sequence_type %s." +
                   " These columns have been removed.") %\
                  (repr(invalid_chars), sequence_type)
        warnings.warn(message, UserWarning)

        # Rename columns
        matrix = matrix.rename(columns=translation_dict)

        # Collapse columns with same name
        matrix = matrix.groupby(matrix.columns, axis=1).sum()

        # Order columns alphabetically
        new_columns = list(matrix.columns)
        new_columns.sort()
        matrix = matrix.loc[:, new_columns]


    # Validate new matrix
    matrix = validate_mat(matrix)

    return matrix


def iupac_to_probability_mat(iupac):
    """Returns a probability matrix correspondign to a specified iupac string
    """

    # Create counts matrix based on IUPAC string
    L = len(iupac)
    rows = range(L)
    cols = list('ACGT')
    counts_mat = pd.DataFrame(index=rows, columns=cols).fillna(0)
    for i, c in enumerate(list(iupac)):
        bs = iupac_dict[c]
        for b in bs:
            counts_mat.loc[i, b] = 1

    # Convert counts matrix to probability matrix
    probability_mat = counts_mat_to_probability_mat(counts_mat,
                                                    pseudocount=SMALL)

    # Return probability matrix to user
    return probability_mat

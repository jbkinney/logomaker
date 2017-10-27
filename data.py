from __future__ import division
import numpy as np
import pandas as pd
import re
import pdb
from Bio import SeqIO

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

from validate import validate_mat, validate_probability_mat

def transform_mat(matrix, to_type, from_type=None, background=None,
                  pseudocount=1, enrichment_logbase=2,
                  enrichment_centering=True, information_units='bits'):
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
    matrix = validate_mat(matrix)

    # Create background matrix
    bg_mat = set_bg_mat(background=background, matrix=matrix)

    # Compute freq_mat from from_type
    if from_type == 'probability':
        probability_mat = validate_probability_mat(matrix)

    elif from_type == 'counts':
        probability_mat = \
            counts_mat_to_probability_mat(matrix, pseudocount=pseudocount)

    elif from_type == 'enrichment':
        probability_mat = \
            enrichment_mat_to_probability_mat(matrix, bg_mat,
                                              base=enrichment_logbase)

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
                                              centering=enrichment_centering)

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


def enrichment_mat_to_probability_mat(weight_mat, bg_mat, base=2):
    '''
    Converts a weight_mat to a freq_mat
    '''
    # Validate mat before use
    weight_mat = validate_mat(weight_mat)

    # Compute freq_mat
    vals = weight_mat.values
    vals -= vals.mean(axis=1)[:, np.newaxis]
    weights = np.pow(base, vals) * bg_mat.values
    freq_mat.loc[:, :] = weights / weights.sum(axis=1)[:, np.newaxis]
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
    freq_mat = validate_probability_mat(freq_mat)

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
    freq_mat = validate_probability_mat(freq_mat)
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
                    all(background.index == matrix.index):
        assert all(matrix.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        new_bg_mat = background.copy()

    else:
        assert False, 'Error: bg_mat and df are incompatible'
    new_bg_mat = normalize_probability_matrix(new_bg_mat)
    return new_bg_mat


def load_alignment(fasta_file=None,
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

        # Assign each sequence a count of 1
        sequence_counts = np.ones(len(sequences))

    assert sequences is not None, \
        'Error: either fasta_file or sequences must not be None.'

    # Get seq length
    L = len(sequences[0])
    assert all([len(seq) == L for seq in sequences]), 'Error: not all sequences have length %d.' % L

    # Get counts list
    if sequence_counts is None:
        assert len(sequences) > 0, 'Error: sequences is empty'
        counts_array = np.ones(len(sequences))
    else:
        assert len(sequence_counts) == len(sequences), 'Error: sequence_counts is not the same length as sequences'
        counts_array = np.array(sequence_counts)

    # If positions is not specified by user, make it
    if positions is not None:
        assert len(positions) == L, 'Error: positions, if passed, must be same length as sequences.'
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

    # If manually restricting to specific characters, do it:
    if characters is not None:
        new_columns = [c for c in characters if c in matrix.columns]
        new_matrix = matrix.loc[:,new_columns]

    # Otherwise performing translation, do it
    elif len(translation_dict) > 0:
        # Rename columns
        new_matrix = matrix.rename(columns=translation_dict)

        # Collapse columns with same name
        new_matrix = new_matrix.groupby(new_matrix.columns, axis=1).sum()

        # Order columns alphabetically
        new_columns = list(set(translation_dict.values()))
        new_columns.sort()
        new_matrix = new_matrix.loc[:, new_columns]

    # Otherwise, just copy matrix
    else:
        new_matrix = matrix.copy()

    # Remove any characters to ignore
    for char in ignore_characters:
        if char in new_matrix.columns:
            del new_matrix[char]

    return new_matrix
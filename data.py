from __future__ import division
import numpy as np
import pandas as pd
import re
import pdb

# Set constants
SMALL = 1E-6

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


def transform_mat(matrix, to_type, from_type=None, background=None,
                  energy_gamma=1, pseudocount=1, enrichment_logbase=2,
                  information_units='bits'):
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

    # Get from_type if not specified
    if from_type is None:
        try:
            from_type = matrix.logomaker_type
        except:
            assert False, 'Cant determine from_type'

    # Create background matrix
    bg_mat = set_bg_mat(background=background, matrix=matrix)

    # Compute freq_mat from from_type
    if from_type == 'probability':
        probability_mat = validate_probability_mat(matrix)

    elif from_type == 'counts':
        probability_mat = counts_mat_to_probability_mat(matrix,
                                                        pseudocount=pseudocount)

    elif from_type == 'energy':
        probability_mat = energy_mat_to_probability_mat(matrix, bg_mat,
                                                        gamma=energy_gamma)

    elif from_type == 'enrichment':
        probability_mat = enrichment_mat_to_probability_mat(matrix, bg_mat,
                                                            base=enrichment_logbase)

    else:
        assert False, 'Error! from_type %s is invalid.'%from_type

    # Compute out_mat from freq_mat
    if to_type == 'counts':
        if from_type == 'counts':
            out_mat = matrix
        else:
            assert False, 'Cannot convert from %s to count_mat' % \
                          matrix.logomaker_type

    elif to_type == 'probability':
        out_mat = probability_mat

    elif to_type == 'energy':
        out_mat = probability_mat_to_energy_mat(probability_mat, bg_mat,
                                                gamma=energy_gamma)

    elif to_type == 'enrichment':
        out_mat = probability_mat_to_enrichment_mat(probability_mat, bg_mat,
                                                    base=enrichment_logbase)

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
    freq_mat = normalize_freq_mat(freq_mat)

    # Validate and return
    freq_mat = validate_probability_mat(freq_mat)
    freq_mat.logomaker_type = 'probability'
    return freq_mat

def energy_mat_to_probability_mat(energy_mat, bg_mat, gamma=1):
    '''
    Converts an energy_mat to a freq_mat
    '''
    # Validate mat before use
    energy_mat = validate_mat(energy_mat)

    # Compute freq_mat
    freq_mat = energy_mat.copy()
    vals = energy_mat.values
    vals -= vals.mean(axis=1)[:, np.newaxis]
    weights = np.exp(-gamma * vals) * bg_mat.values
    freq_mat.loc[:, :] = weights / weights.sum(axis=1)[:, np.newaxis]
    freq_mat = normalize_freq_mat(freq_mat)

    # Validate and return
    freq_mat = validate_probability_mat(freq_mat)
    freq_mat.logomaker_type = 'probability'
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
    freq_mat = normalize_freq_mat(freq_mat)

    # Validate and return
    freq_mat = validate_probability_mat(freq_mat)
    freq_mat.logomaker_type = 'probability'
    return freq_mat

def probability_mat_to_energy_mat(freq_mat, bg_mat, gamma):
    '''
    Converts a freq_mat to an energy_mat
    '''
    # Validate mat before use
    freq_mat = validate_probability_mat(freq_mat)

    # Compute energy_mat
    energy_mat = freq_mat.copy()
    vals = freq_mat.values
    energy_mat.loc[:, :] = (1 / gamma) * np.log(vals / bg_mat.values)
    energy_mat = normalize_energy_mat(energy_mat)

    # Validate and return
    energy_mat = validate_mat(energy_mat)
    energy_mat.logomaker_type = 'energy'
    return energy_mat

def probability_mat_to_enrichment_mat(freq_mat, bg_mat, base=2):
    '''
    Converts a freq_mat to an energy_mat
    '''
    # Validate mat before use
    freq_mat = validate_probability_mat(freq_mat)

    # Compute weight_mat
    weight_mat = freq_mat.copy()
    weight_mat.loc[:, :] = np.log2(freq_mat / bg_mat)/np.log2(base)

    # Validate and return
    weight_mat = validate_mat(weight_mat)
    weight_mat.logomaker_type = 'enrichment'
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
    info_mat.logomaker_type = 'information'
    return info_mat

# Normalize a data frame of energies
def normalize_energy_mat(energy_mat):
    mat = energy_mat.copy()
    mat.loc[:, :] = mat.values - mat.values.mean(axis=1)[:, np.newaxis]
    mat.logomaker_type = 'energy'
    return mat

# Normalize a data frame of probabilities
def normalize_freq_mat(freq_mat, regularize=True):
    mat = freq_mat.copy()
    assert all(np.ravel(mat.values) >= 0), \
        'Error: Some data frame entries are negative.'
    mat.loc[:, :] = mat.values / mat.values.sum(axis=1)[:, np.newaxis]
    if regularize:
        mat.loc[:, :] += SMALL
    mat.logomaker_type = 'probability'
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
    new_bg_mat = normalize_freq_mat(new_bg_mat)
    new_bg_mat.logomaker_type='probability'
    return new_bg_mat



def load_alignment(fasta_file=None, sequences=None, sequence_counts=None,
                   characters=None, positions=None, ignore_characters='.-'):

    # If loading file name
    if fasta_file is not None:

        # Load lines
        with open(fasta_file, 'r') as f:
            lines = f.readlines()

        # Remove whitespace
        pattern = re.compile(r'\s+')
        lines = [re.sub(pattern, '', line) for line in lines]

        # Remove comment lines
        pattern = re.compile(r'^[>#]')
        lines = [line for line in lines if not re.match(pattern, line)]

        # Remove empty lines
        lines = [line for line in lines if (len(line) > 0)]

        # Store sequences and counts
        sequences = lines
        sequence_counts = np.ones(len(lines))

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

    # If characters are not specified by user, get list of unique characters from sequence
    if characters is None:
        seq_concat = ''.join(sequences)
        characters = list(set(seq_concat))
        characters.sort()
    elif isinstance(characters, basestring):
        characters = list(characters)

    # If positions is not specified by user, make it
    if positions is not None:
        assert len(positions) == L, 'Error: positions, if passed, must be same length as sequences.'
    else:
        positions = range(L)

    # Create counts matrix
    counts_mat = pd.DataFrame(index=positions, columns=characters).fillna(0)

    # Create array of characters at each position
    char_array = np.array([np.array(list(seq)) for seq in sequences])

    # Sum of the number of occurances of each character at each position
    for c in characters:
        v = (char_array == c).astype(float)
        v *= counts_array[:, np.newaxis]
        counts_mat.loc[:, c] = v.sum(axis=0).ravel()

    # Remove columns corresponding to unwanted characters
    columns = counts_mat.columns.copy()
    for char in ignore_characters:
        if char in columns:
            del counts_mat[char]

    # Name index
    counts_mat.index.name = 'pos'
    counts_mat.logomaker_type = 'counts'

    return counts_mat
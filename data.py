from __future__ import division
import numpy as np
import pandas as pd
import re
import pdb

from utils import SMALL

def validate_mat(mat):
    '''
    Runs assert statements to verify that df is indeed a motif dataframe.
    Returns a cleaned-up version of df if possible
    '''
    mat = mat.copy()
    assert type(mat) == pd.core.frame.DataFrame, 'Error: df is not a dataframe'
    cols = mat.columns

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
        if not len(new_col_name)==1:
            new_col_name = new_col_name.split('_')[-1]
            assert (len(new_col_name)==1), \
                'Error: could not extract single character from colum name %s'%col_name

        # Make sure that colun name is not a whitespace character
        assert re.match('\S',new_col_name), \
            'Error: column name "%s" is a whitespace charcter.'%repr(col_name)

        # Set revised column name
        mat.rename(columns={col_name:new_col_name}, inplace=True)

    # If there is a pos column, make that the index
    if 'pos' in cols:
        mat.set_index('pos', drop=True, inplace=True)

    # Remove name from index column
    mat.index.names = [None]

    # Alphabetize character columns
    char_cols = list(mat.columns)
    char_cols.sort()
    mat = mat[char_cols]

    # Return cleaned-up df
    return mat

def validate_freq_mat(mat):
    '''
    Verifies that the df is indeed a probability motif dataframe.
    Returns a normalized and cleaned-up version of df if possible
    '''

    # Validate as motif
    mat = validate_mat(mat)

    # Validate df values as info values
    assert (all(mat.values.ravel() >= 0)), \
        'Error: not all values in df are >=0.'

    # Normalize across columns
    mat.loc[:, :] = mat.values / mat.values.sum(axis=1)[:, np.newaxis]

    return mat

def validate_info_mat(mat):
    '''
     Verifies that the df is indeed an info motif dataframe.
     Returns a cleaned-up version of df if possible
     '''

    # Validate as motif
    mat = validate_mat(mat)

    # Validate df values as info values
    assert (all(mat.values.ravel() >= 0)), \
        'Error: not all values in df are >=0.'

    return mat

def transform_mat(mat, from_type, to_type, background=None, beta=1, pseudocount=1):
    '''
    transform_mat(): transforms a matrix of one type into another.
    :param mat: input matrix, in data frame format
    :param from_type: type of matrix to transform from
    :param to_type: type of matrix to transform to
    :param background: background base frequencies
    :param beta: parameter for converting energies to frequencies
    :param pseudocount: psuedocount used to convert count_mat to freq_mat
    :return: out_mat, a matrix in data frame format of the specified output type
    '''

    # Check that mat is valid
    mat = validate_mat(mat)

    # Create background mat
    bg_mat = set_bg_mat(background=None, mat=mat)

    # Compute freq_mat from from_type
    if from_type == 'freq_mat':
        freq_mat = validate_freq_mat(mat)

    elif from_type == 'count_mat':
        freq_mat = count_mat_to_freq_mat(mat, pseudocount=pseudocount)

    elif from_type == 'energy_mat':
        freq_mat = energy_mat_to_freq_mat(mat, bg_mat, beta)

    elif from_type == 'weight_mat':
        freq_mat = weight_mat_to_freq_mat(mat, bg_mat)

    else:
        assert False, 'Error! from_type %s is invalid.'%from_type

    # Compute out_mat from freq_mat
    if to_type == 'freq_mat':
        out_mat = freq_mat

    elif to_type == 'energy_mat':
        out_mat = freq_mat_to_energy_mat(freq_mat, bg_mat, beta)

    elif to_type == 'weight_mat':
        out_mat = freq_mat_to_weight_mat(freq_mat, bg_mat)

    elif to_type == 'info_mat':
        out_mat = freq_mat_to_info_mat(freq_mat, bg_mat)

    else:
        assert False, 'Error! to_type %s is invalid.'%to_type

    # Return out_mat
    return out_mat

def count_mat_to_freq_mat(count_mat, pseudocount=1):
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
    freq_mat = validate_freq_mat(freq_mat)
    return freq_mat

def energy_mat_to_freq_mat(energy_mat, bg_mat, beta=1):
    '''
    Converts an energy_mat to a freq_mat
    '''
    # Validate mat before use
    energy_mat = validate_mat(energy_mat)

    # Compute freq_mat
    freq_mat = energy_mat.copy()
    vals = energy_mat.values
    vals -= vals.mean(axis=1)[:, np.newaxis]
    weights = np.exp(-beta * vals) * bg_mat.values
    freq_mat.loc[:, :] = weights / weights.sum(axis=1)[:, np.newaxis]
    freq_mat = normalize_freq_mat(freq_mat)

    # Validate and return
    freq_mat = validate_freq_mat(freq_mat)
    return freq_mat

def weight_mat_to_freq_mat(weight_mat, bg_mat, base=2):
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
    freq_mat = validate_freq_mat(freq_mat)
    return freq_mat

def freq_mat_to_energy_mat(freq_mat, bg_mat, beta):
    '''
    Converts a freq_mat to an energy_mat
    '''
    # Validate mat before use
    freq_mat = validate_freq_mat(freq_mat)

    # Compute energy_mat
    energy_mat = freq_mat.copy()
    vals = freq_mat.values
    energy_mat.loc[:,:] = (1/beta)*np.log(vals/bg_mat.values)
    energy_mat = normalize_energy_mat(energy_mat)

    # Validate and return
    energy_mat = validate_mat(energy_mat)
    return energy_mat

def freq_mat_to_weight_mat(freq_mat, bg_mat):
    '''
    Converts a freq_mat to an energy_mat
    '''
    # Validate mat before use
    freq_mat = validate_freq_mat(freq_mat)

    # Compute weight_mat
    weight_mat = freq_mat.copy()
    weight_mat.loc[:,:]  = freq_mat * np.log2(freq_mat / bg_mat)

    # Validate and return
    weight_mat = validate_mat(weight_mat)
    return weight_mat

# Needed only for display purposes
def freq_mat_to_info_mat(freq_mat, bg_mat):
    '''
    Converts a prob df to an information df
    '''
    # Validate mat before use
    freq_mat = validate_freq_mat(freq_mat)

    info_mat = freq_mat * np.log2(freq_mat/bg_mat)

    # Validate and return
    info_mat = validate_mat(info_mat)
    return info_mat


# def get_beta_for_effect_mat(effect_mat, bg_mat, target_info, \
#                            min_beta=.001, max_beta=100, num_betas=1000):
#     betas = np.exp(np.linspace(np.log(min_beta), np.log(max_beta), num_betas))
#     infos = np.zeros(len(betas))
#     for i, beta in enumerate(betas):
#         prob_mat = effect_mat_to_prob_mat(effect_mat, bg_mat, beta)
#         infos[i] = get_prob_mat_info(prob_mat, bg_mat)
#     i = np.argmin(np.abs(infos - target_info))
#     beta = betas[i]
#     return beta

# Normalize a data frame of energies
def normalize_energy_mat(energy_mat):
    mat = energy_mat.copy()
    mat.loc[:, :] = mat.values - mat.values.mean(axis=1)[:, np.newaxis]
    return mat

# Normalize a data frame of probabilities
def normalize_freq_mat(freq_mat, regularize=True):
    mat = freq_mat.copy()
    assert all(np.ravel(mat.values) >= 0), \
        'Error: Some data frame entries are negative.'
    mat.loc[:, :] = mat.values / mat.values.sum(axis=1)[:, np.newaxis]
    if regularize:
        mat.loc[:, :] += SMALL
    return mat


def set_bg_mat(background, mat):
    '''
    Creates a background matrix given a background specification and matrix
    with the right rows and columns
     '''
    num_pos, num_cols = mat.shape

    # Create background from scratch
    if background is None:
        new_bg_mat = mat.copy()
        new_bg_mat.loc[:, :] = 1 / num_cols

    # Expand rows of list or numpy array background
    elif type(background) == list or type(background) == np.ndarray:
        assert len(background) == mat.shape[1], \
            'Error: df and background have mismatched dimensions.'
        new_bg_mat = mat.copy()
        background = np.array(background).ravel()
        new_bg_mat.loc[:, :] = background

    elif type(background) == dict:
        assert set(background.keys()) == set(mat.columns), \
            'Error: df and background have different columns.'
        new_bg_mat = mat.copy()
        for i in new_bg_mat.index:
            new_bg_mat.loc[i, :] = background

    # Expand single-row background data frame
    elif type(background) == pd.core.frame.DataFrame and \
                    background.shape == (1, num_cols):
        assert all(mat.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        new_bg_mat = mat.copy()
        new_bg_mat.loc[:, :] = background.values.ravel()

    # Use full background dataframe
    elif type(background) == pd.core.frame.DataFrame and \
                    background.index == mat.index:
        assert all(mat.columns == background.columns), \
            'Error: df and bg_mat have different columns.'
        new_bg_mat = background.copy()

    else:
        assert False, 'Error: bg_mat and df are incompatible'
    new_bg_mat = normalize_freq_mat(new_bg_mat)
    return new_bg_mat
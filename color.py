from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import restrict_dict

# Create color scheme dict
three_zeros = np.zeros(3)
three_ones = np.ones(3)
color_scheme_dict = {

    'classic': {
        'G': [1, .65, 0],
        'TU': [1, 0, 0],
        'C': [0, 0, 1],
        'A': [0, .5, 0]
    },

    'black': {
        'ACGT': three_zeros
    },

    'gray': {
        'A': .2 * three_ones,
        'C': .4 * three_ones,
        'G': .6 * three_ones,
        'TU': .8 * three_ones
    },

    'base pairing': {
        'TAU': [1, .55, 0],
        'GC': [0, 0, 1]
    },

    'hydrophobicity': {
        'RKDENQ': [0, 0, 1],
        'SGHTAP': [0, .5, 0],
        'YVMCLFIW': [0, 0, 0]
    },

    'chemistry': {
        'GSTYC': [0, .5, 0],
        'QN': [.5, 0, .5],
        'KRH': [0, 0, 1],
        'DE': [1, 0, 0],
        'AVLIPWFM': [0, 0, 0]
    },

    'charge': {
        'KRH': [0, 0, 1],
        'DE': [1, 0, 0],
        'GSTYCQNAVLIPWFM': [.5, .5, .5]
    }
}

def cmap_to_color_scheme(chars, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    num_char = len(chars)
    vals = np.linspace(0, 1, 2 * num_char + 1)[1::2]
    color_scheme = {}
    for n, char in enumerate(chars):
        color = cmap(vals[n])[:3]
        color_scheme[char] = color
    return color_scheme

# Expand strings in color dict into individual characters
def expand_color_dict(color_dict):
    new_dict = {}
    for key in color_dict.keys():
        value = color_dict[key]
        for char in key:
            new_dict[char.upper()] = value
            new_dict[char.lower()] = value
    return new_dict

def get_color_dict(color_scheme,chars,shuffle_colors=False):
    ''' get color_dict: each key is 1 char, each value is a 4-vector of rgba values '''
    # Set color scheme
    if type(color_scheme) == dict:
        color_dict = expand_color_dict(color_scheme)
        for char in chars:
            assert char in color_dict
    elif type(color_scheme) == str:
        if color_scheme in color_scheme_dict:
            color_dict = color_scheme_dict[color_scheme]
            color_dict = expand_color_dict(color_dict)
        elif color_scheme == 'random':
            color_dict = {}
            for char in chars:
                color_dict[char] = np.random.rand(3)
        else:
            cmap_name = color_scheme
            color_dict = cmap_to_color_scheme(chars, cmap_name)
            # assert False, 'invalid color_scheme %s'%color_scheme;
    else:
        assert False, 'color_scheme has invalid type.'

    # Restrict color_dict to only characters in columns
    assert set(chars) <= set(color_dict.keys()), \
        'Error: column characters not in color_dict'
    color_dict = restrict_dict(color_dict, chars)

    # Shuffle colors if requested
    if shuffle_colors:
        chars = color_dict.keys()
        values = color_dict.values()
        np.random.shuffle(chars)
        color_dict = dict(zip(chars, values))

    return color_dict

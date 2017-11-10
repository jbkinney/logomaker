from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import warnings
import pdb

# Create facecolor scheme dict
three_zeros = np.zeros(3)
three_ones = np.ones(3)
color_scheme_dict = {
    'classic': {
        'G': [1, .65, 0],
        'TU': [1, 0, 0],
        'C': [0, 0, 1],
        'A': [0, .5, 0]
    },

    'grays': {
        'A': .2 * three_ones,
        'C': .4 * three_ones,
        'G': .6 * three_ones,
        'TU': .8 * three_ones
    },

    'base_pairing': {
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
    },

    'NajafabadiEtAl2017': {
        'DEC': [.42, .16, .42],
        'PG': [.47, .47, 0.0],
        'MIWALFV': [.13, .35, .61],
        'NTSQ': [.25, .73, .28],
        'RK': [.74, .18, .12],
        'HY': [.09, .47, .46],
    },
}

def restrict_dict(in_dict, keys_to_keep):
    return dict([(k, v) for k, v in in_dict.iteritems() if k in keys_to_keep])

def cmap_to_color_scheme(chars, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    num_char = len(chars)
    vals = np.linspace(0, 1, 2 * num_char + 1)[1::2]
    color_scheme = {}
    for n, char in enumerate(chars):
        color = cmap(vals[n])[:3]
        color_scheme[char] = color
    return color_scheme

# Expand strings in facecolor dict into individual characters
def expand_color_dict(color_dict):
    new_dict = {}
    for key in color_dict.keys():
        value = color_dict[key]
        for char in key:
            new_dict[char.upper()] = value
            new_dict[char.lower()] = value
    return new_dict


def get_color_dict(color_scheme, chars, alpha, shuffle_colors=False):
    '''
    get color_dict: each key is 1 char, each value is a 4-vector of rgba values
    This is the main function that Logo interfaces with
    '''

    # First check if color_scheme can be interpreted as a simple facecolor
    is_color = None
    try:
        color = to_rgb(color_scheme)
        is_color = True
    except:
        pass;

    # Initialize color_dict
    color_dict = {}

    # If a single facecolor
    if is_color:
        for char in chars:
            color_dict[char] = color

    # If a predefined facecolor scheme
    elif type(color_scheme) == dict:
        color_dict = expand_color_dict(color_scheme)

    # If a string
    elif type(color_scheme) == str:
        # Check if color scheme is 'none'
        if color_scheme == 'none':
            for char in chars:
                color_dict[char] = [0, 0, 0, 0]
            alpha = 0

        # Otherwise, check if random
        elif color_scheme == 'random':
            for char in chars:
                color_dict[char] = np.random.rand(3)

        # Check if there is a pre-defined facecolor scheme
        elif color_scheme in color_scheme_dict:
            color_dict = color_scheme_dict[color_scheme]
            color_dict = expand_color_dict(color_dict)

        else:
            cmap_name = color_scheme
            color_dict = cmap_to_color_scheme(chars, cmap_name)
            # assert False, 'invalid color_scheme %s'%color_scheme;
    else:
        assert False, 'color_scheme has invalid type.'

    # Restrict color_dict to only characters in columns
    # assert set(chars) <= set(color_dict.keys()), \
    #     'Error: column characters not in color_dict'
    # color_dict = restrict_dict(color_dict, chars)

    if not (set(chars) <= set(color_dict.keys())):
        for c in chars:
            if not c in color_dict:
                message = "Character '%s' is not in color_dict. Using black."\
                    % c
                warnings.warn(message, UserWarning)
                color_dict[c] = to_rgb('black')

    # Shuffle colors if requested
    if shuffle_colors:
        chars = color_dict.keys()
        values = color_dict.values()
        np.random.shuffle(chars)
        color_dict = dict(zip(chars, values))

    # Set alpha=1 if None is provided
    if alpha is None:
        alpha = 1.0

    # Add an alpha to each color
    for key in color_dict:
        rgb = color_dict[key]
        rgba = np.array(list(rgb)[:3] + [alpha])
        color_dict[key] = rgba

    return color_dict

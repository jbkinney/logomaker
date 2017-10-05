from __future__ import division
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import pdb
import string
import os

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties, findSystemFonts
from matplotlib.colors import to_rgba
from matplotlib import font_manager

# Set constants
SMALL = 1E-6
DEFAULT_FONT = 'Arial Rounded Bold'

# Build font_file_dict
font_file_list = \
    findSystemFonts(fontpaths=None, fontext='ttf')
FONT_FILE_DICT = {}
for font_file in font_file_list:
    base_name = os.path.basename(font_file)
    font_name = os.path.splitext(base_name)[0]
    FONT_FILE_DICT[str(font_name)] = font_file
FONTS = FONT_FILE_DICT.keys()
FONTS.sort()

class Box:
    def __init__(self, xlb, xub, ylb, yub):
        self.xlb = xlb
        self.xub = xub
        self.ylb = ylb
        self.yub = yub
        self.x = xlb
        self.y = ylb
        self.w = xub-xlb
        self.h = yub-ylb
        self.bounds = (xlb, xub, ylb, yub)

class Character:
    def __init__(self, c, x, y, w, h, color, 
        font_name=DEFAULT_FONT, 
        flip=False, 
        shade=1, 
        alpha=1):
        assert w > 0
        assert h > 0

        self.c = c
        self.box = Box(x,x+w,y,y+h)
        self.font_name = font_name
        self.flip = flip

        # Set color
        try:
            self.color = np.array(to_rgba(color))*\
                np.array([shade,shade,shade,1])
        except:
            assert False, 'Error! Unable to interpret color %s'%repr(color)
        
        # Set tranparency
        self.color[3] = alpha

    def draw(self,ax):

        # Define character bounding box
        bbox = list(self.box.bounds)
        if self.flip:
            bbox[2:] = bbox[2:][::-1]
                
        # Draw character
        put_char_in_box(ax, self.c, bbox, \
            facecolor=self.color, 
            font_name=self.font_name)

# Logo base class
class Logo:
    def __init__(self,logo_set=False):
        self.logo_set = logo_set

    def draw(self,ax):
        assert self.logo_set, 'Error: cant plot because logo is not set yet.'

        # Draw floor
        plt.axhline(0,linewidth=self.floor_line_width,color='k',zorder=-1)

        # Draw characters
        for char in self.char_list:
            char.draw(ax)

        # Logo-specific formatting
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xticks(self.xticks)
        ax.set_yticks(self.yticks)
        ax.set_yticklabels(self.yticklabels)
        ax.set_xticklabels(self.xticklabels,rotation=90)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        # Standard formatting
        ax.xaxis.set_tick_params(width=0,length=0)
        plt.box('off')


# Information logo clas
class InformationLogo(Logo):
    def __init__(self, prob_df, bg_df, color_dict, 
        font_name=DEFAULT_FONT, floor_line_width=.5):
        
        df = prob_df.copy()
        info_vec = prob_df.values*np.log2(prob_df.values/bg_df.values)
        df.loc[:,:] = prob_df.values*info_vec.sum(axis=1)[:,np.newaxis]
        assert all(np.ravel(df.values)>=0)

        char_list, box = compute_logo_characters(
                                df=df, 
                                stack_order='big_on_top', 
                                color_dict=color_dict,
                                font_name=font_name,
                                use_transparency=False)

        assert np.isclose(box.ylb,0),\
            'Error: box.ylb=%f is not zero.'%box.ylb
        
        self.signed_heights_df = df
        self.floor_line_width = floor_line_width
        self.font_name = font_name
        self.prob_df = prob_df.copy()
        self.bg_df = bg_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1]))+1
            )
        self.xticklabels = ['%d'%x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [-.01, max(2,box.yub)+.01]
        self.yticks = range(
            int(np.ceil(self.ylim[0])),
            int(np.floor(self.ylim[1]))+1
            )
        self.yticklabels = ['%d'%y for y in self.yticks]
        self.ylabel = 'information\n(bits)'


        # Register that logo has been set
        Logo.__init__(self,logo_set=True)

# Probability logo clas
class ProbabilityLogo(Logo):
    def __init__(self, prob_df, color_dict, 
        font_name=DEFAULT_FONT, floor_line_width=.5):
        
        df = prob_df.copy()
        assert all(np.ravel(df.values)>=0)

        char_list, box = compute_logo_characters(
                                df=df, 
                                stack_order='small_on_top', 
                                color_dict=color_dict,
                                font_name=font_name,
                                use_transparency=True)

        assert np.isclose(box.ylb,0),\
            'Error: box.ylb=%f is not zero.'%box.ylb
        assert np.isclose(box.yub,1),\
            'Error: box.yub=%f is not one.'%box.yub
        
        self.signed_heights_df = df
        self.floor_line_width = floor_line_width
        self.font_name = font_name
        self.prob_df = prob_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1]))+1
            )
        self.xticklabels = ['%d'%x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [-.01, 1.01]
        self.yticks = [0, .5, 1]
        self.yticklabels = ['%.1f'%y for y in self.yticks]
        self.ylabel = 'probability'

        # Register that logo has been set
        Logo.__init__(self,logo_set=True)

# Energy logo clas
class EnergyLogo(Logo):
    def __init__(self, energy_df, color_dict, 
        font_name=DEFAULT_FONT, floor_line_width=.5):
        
        df = -energy_df.copy()
        char_list, box = compute_logo_characters(
                                df=df, 
                                stack_order='big_on_top', 
                                color_dict=color_dict,
                                font_name=font_name,
                                use_transparency=False,
                                neg_shade=.5, 
                                neg_flip=True
                                )
        
        self.signed_heights_df = df
        self.floor_line_width = floor_line_width
        self.font_name = font_name
        self.energy_df = energy_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1]))+1
            )
        self.xticklabels = ['%d'%x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [box.ylb, box.yub]
        self.yticks = range(
            int(np.ceil(self.ylim[0])),
            int(np.floor(self.ylim[1]))+1
            )
        self.yticklabels = ['%d'%y for y in self.yticks]
        self.ylabel = '-energy\n($k_B T$)'

        # Register that logo has been set
        Logo.__init__(self,logo_set=True)

# Enrichment logo clas
class EnrichmentLogo(Logo):
    def __init__(self, prob_df, bg_df, color_dict, 
        font_name=DEFAULT_FONT, floor_line_width=.5):
        
        df = prob_df.copy()
        df.loc[:,:] = np.log2(prob_df.values/bg_df.values)
        char_list, box = compute_logo_characters(
                                df=df, 
                                stack_order='big_on_top', 
                                color_dict=color_dict,
                                font_name=font_name,
                                use_transparency=False,
                                neg_shade=.5, 
                                neg_flip=True
                                )
        
        self.signed_heights_df = df
        self.floor_line_width = floor_line_width
        self.font_name = font_name
        self.prob_df = prob_df.copy()
        self.bg_df = bg_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1]))+1
            )
        self.xticklabels = ['%d'%x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [box.ylb, box.yub]
        self.yticks = range(
            int(np.ceil(self.ylim[0])),
            int(np.floor(self.ylim[1]))+1
            )
        while len(self.yticks) >= 6:
            self.yticks = self.yticks[::2]
        self.yticklabels = ['%d'%y for y in self.yticks]
        self.ylabel = '$\log_2$ enrichment'

        # Register that logo has been set
        Logo.__init__(self,logo_set=True)


def compute_logo_characters(df, \
    stack_order, color_dict, \
    font_name=DEFAULT_FONT,\
    neg_shade=1, neg_flip=False, \
    use_transparency=False):

    poss = df.index.copy()
    chars = df.columns.copy()

    char_list = []
    for i, pos in enumerate(poss):

        vals = df.loc[pos,:].values
        ymin = (vals*(vals < 0)).sum()
        ymax = (vals*(vals > 0)).sum()
        
        # Reorder columns
        if stack_order=='big_on_top':
            indices = np.argsort(vals)
        elif stack_order=='small_on_top':
            indices = np.argsort(vals)[::-1]
        ordered_chars = chars[indices]

        # This is the same for every character
        x = pos-.5
        w = 1.0

        # Initialize y
        y = ymin

        for n, char in enumerate(ordered_chars):

            # Get value
            val = df.loc[pos,char]

            # Get height
            h = abs(val)
            if h < SMALL:
                continue

            # Get color
            color = color_dict[char]

            # Get flip, alpha, and shade
            if val >= 0.0:
                alpha = h if use_transparency else 1.0
                assert alpha <= 1.0, \
                    'Error: alpha=%f must be in [0,1]'%alpha
                flip = False
                shade = 1.0
            else:
                alpha = neg_shade
                flip = neg_flip
                shade = neg_shade

            # Create and store character
            char = Character(
                c=char, x=x, y=y, w=w, h=h, 
                alpha=alpha, color=color, flip=flip, 
                shade=shade, font_name=font_name)
            char_list.append(char)

            # Increment y
            y += h

    # Get box
    xlb = min([c.box.xlb for c in char_list])
    xub = max([c.box.xub for c in char_list])
    ylb = min([c.box.ylb for c in char_list])
    yub = max([c.box.yub for c in char_list])
    box = Box(xlb, xub, ylb, yub)

    return char_list, box


def get_prob_df_info(prob_df,bg_df):
    return np.sum(prob_df.values*np.log2(prob_df.values/bg_df.values))


def energy_df_to_prob_df(energy_df,bg_df,beta):
    prob_df = energy_df.copy()
    vals = energy_df.values
    vals -= vals.mean(axis=1)[:,np.newaxis]
    weights = np.exp(-beta*vals)*bg_df.values
    prob_df.loc[:,:] = weights/weights.sum(axis=1)[:,np.newaxis]
    return prob_df


def get_beta_for_energy_df(energy_df,bg_df,target_info,\
    min_beta=.001,max_beta=100,num_betas=1000):
    betas = np.exp(np.linspace(np.log(min_beta),np.log(max_beta),num_betas))
    infos = np.zeros(len(betas))
    for i, beta in enumerate(betas):
        prob_df = energy_df_to_prob_df(energy_df,bg_df,beta)
        infos[i] = get_prob_df_info(prob_df,bg_df)
    i = np.argmin(np.abs(infos-target_info))
    beta = betas[i]
    return beta


def restrict_dict(in_dict,keys_to_keep):
    return dict([(k,v) for k,v in in_dict.iteritems() if k in keys_to_keep])

def get_fonts():
    return FONTS

def put_char_in_box(ax, char, bbox, facecolor='k', \
    edgecolor='none', font_name=None, zorder=0):
    
    # Get default font properties if none specified
    if font_name is None:
        font_properties = FontProperties(family='sans-serif', weight='bold')
    elif font_name in FONT_FILE_DICT:
        font_file = FONT_FILE_DICT[font_name]
        font_properties = FontProperties(fname=font_file)
    else:
        assert False, 'Error: unable to interpret font name %s'%font_name

    # Create raw path
    path = TextPath((0,0), char, size=1, prop=font_properties)
    
    # Get params from bounding box
    try:
        set_xlb, set_xub, set_ylb, set_yub = tuple(bbox)
    except:
        pdb.set_trace()
    set_w = set_xub - set_xlb
    set_h = set_yub - set_ylb
    
    # Rescale path vertices to fit in bounding box
    num_vertices = len(path._vertices)
    raw_xs = [x[0] for x in path._vertices]
    raw_ys = [x[1] for x in path._vertices]
    raw_xlb = min(raw_xs)
    raw_ylb = min(raw_ys)
    raw_w = max(raw_xs)-raw_xlb
    raw_h = max(raw_ys)-raw_ylb
    set_xs = set_xlb + (raw_xs-raw_xlb)*(set_w/raw_w)
    set_ys = set_ylb + (raw_ys-raw_ylb)*(set_h/raw_h)
    
    # Reset vertices of path
    path._vertices = [100*np.array([set_xs[i],set_ys[i]]) for i in range(num_vertices)]
    
    # Make and display patch
    patch = PathPatch(path,facecolor=facecolor, edgecolor=edgecolor,
        zorder=zorder)
    ax.add_patch(patch)


def cmap_to_color_scheme(chars,cmap_name):
    cmap=plt.get_cmap(cmap_name)
    num_char = len(chars)
    vals = np.linspace(0,1,2*num_char+1)[1::2]
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


# Normalize a data frame of probabilities
def normalize_prob_df(prob_df,regularize=True):
    df = prob_df.copy()
    assert all(np.ravel(df.values) >= 0), \
        'Error: Some data frame entries are negative.'
    df.loc[:,:] = df.values/df.values.sum(axis=1)[:,np.newaxis]
    if regularize:
        df.loc[:,:] += SMALL
    return df


# Normalize a data frame of energies
def normalize_energy_df(energies_df):
    df = energies_df.copy()
    df.loc[:,:] = df.values - df.values.mean(axis=1)[:,np.newaxis]
    return df


def set_bg_df(background,df):
    num_pos, num_cols = df.shape

    # Create background from scratch
    if background is None:
        new_bg_df = df.copy()
        new_bg_df.loc[:,:] = 1/num_cols

    # Expand rows of list or numpy array background
    elif type(background)==list or type(background)==np.ndarray:
        assert len(background)==df.shape[1], \
            'Error: df and background have mismatched dimensions.'
        new_bg_df = df.copy()
        background = np.array(background).ravel()
        new_bg_df.loc[:,:] = background

    elif type(background)==dict:
        assert set(background.keys())==set(df.columns),\
            'Error: df and background have different columns.'
        new_bg_df = df.copy()
        for i in new_bg_df.index:
            new_bg_df.loc[i,:] = background

    # Expand single-row background data frame
    elif type(background)==pd.core.frame.DataFrame and \
            background.shape == (1,num_cols):
        assert all(df.columns == background.columns),\
            'Error: df and bg_df have different columns.'
        new_bg_df = df.copy()
        new_bg_df.loc[:,:] = background.values.ravel()

    # Use full background dataframe
    elif type(background)==pd.core.frame.DataFrame and \
            background.index==df.index:
        assert all(df.columns == background.columns),\
            'Error: df and bg_df have different columns.'
        new_bg_df = background.copy()

    else:
        assert False, 'Error: bg_df and df are incompatible'

    new_bg_df = normalize_prob_df(new_bg_df)
    return new_bg_df


# Create color scheme dict
three_zeros = np.zeros(3)
three_ones = np.ones(3)
color_scheme_dict = {

    'classic':{
        'G':[1,.65,0],
        'TU':[1,0,0],
        'C':[0,0,1],
        'A':[0,.5,0]
    },

    'black':{
        'ACGT':three_zeros
    },

    'gray':{
        'A':.2*three_ones,
        'C':.4*three_ones,
        'G':.6*three_ones,
        'TU':.8*three_ones
    },

    'base pairing':{
        'TAU':[1,.55,0],
        'GC':[0,0,1]
    },

    'hydrophobicity':{
        'RKDENQ':[0,0,1],
        'SGHTAP':[0,.5,0],
        'YVMCLFIW':[0,0,0]
    },

    'chemistry':{
        'GSTYC':[0,.5,0],
        'QN':[.5,0,.5],
        'KRH':[0,0,1],
        'DE':[1,0,0],
        'AVLIPWFM':[0,0,0]
    },

    'charge':{
        'KRH':[0,0,1],
        'DE':[1,0,0],
        'GSTYCQNAVLIPWFM':[.5,.5,.5]
    }
}

def draw(ax,
    prob_df=None,
    energy_df=None,
    find_beta=False,
    info_per_pos=1.0,
    use_transparency=True,
    background=None,
    logo_type='probability',
    color_scheme='classic',
    shuffle_colors=False,
    font_name=DEFAULT_FONT, 
    floor_line_width=.5):

    # Convert energy_df to prob_df
    if (prob_df is None) and not (energy_df is None):
        bg_df = set_bg_df(background,energy_df)
        target_info = info_per_pos*len(energy_df)
        energy_df = normalize_energy_df(energy_df)
        if find_beta:
            beta = get_beta_for_energy_df(energy_df,bg_df,target_info)
        else:
            beta = 1
        prob_df = energy_df_to_prob_df(energy_df,bg_df,beta)

    # Convert prob_df to energy_df (does use bg_Df)
    elif (energy_df is None) and not (prob_df is None):
        bg_df = set_bg_df(background,prob_df)
        prob_df = normalize_prob_df(prob_df)
        energy_df = prob_df.copy()
        values = np.log(prob_df.values/bg_df.values)
        energy_df.loc[:,:] = values - values.mean(axis=1)[:,np.newaxis]
    else:
        assert False,\
            'Error: exactly one of energy_df or prob_df must not be None.'
        
    # Get dimensions
    poss = prob_df.index
    cols = prob_df.columns
    num_pos, num_cols = prob_df.shape

    # Set color scheme
    if type(color_scheme)==dict:
        color_dict = expand_color_dict(color_scheme)
        for char in cols:
            assert char in color_dict
    elif type(color_scheme)==str:
        if color_scheme in color_scheme_dict:
            color_dict = color_scheme_dict[color_scheme]
            color_dict = expand_color_dict(color_dict)
        elif color_scheme=='random':
            color_dict = {}
            for char in cols:
                color_dict[char] = np.random.rand(3)
        else:
            cmap_name = color_scheme
            color_dict = cmap_to_color_scheme(cols,cmap_name)
            #assert False, 'invalid color_scheme %s'%color_scheme;
    else: 
        assert False, 'color_scheme has invalid type.'

    # Restrict color_dict to only characters in columns
    assert set(cols) <= set(color_dict.keys()),\
        'Error: column characters not in color_dict'
    color_dict = restrict_dict(color_dict,cols)

    # Shuffle colors if requested
    if shuffle_colors:
        chars = color_dict.keys()
        values = color_dict.values()
        np.random.shuffle(chars)
        color_dict = dict(zip(chars,values))

    # Create logos based on logo_type
    if logo_type=='information':
        logo = InformationLogo(prob_df, bg_df, color_dict, 
            font_name=font_name, 
            floor_line_width=floor_line_width)

    elif logo_type=='probability':
        logo = ProbabilityLogo(prob_df, color_dict, 
            font_name=font_name, 
            floor_line_width=floor_line_width)

    elif logo_type=='energy':
        logo = EnergyLogo(energy_df, color_dict, 
            font_name=font_name, 
            floor_line_width=floor_line_width)

    elif logo_type=='enrichment':
        logo = EnrichmentLogo(prob_df, bg_df, color_dict, 
            font_name=font_name, 
            floor_line_width=floor_line_width)

    else:
        assert False, 'Error! Unrecognized logo_type %s'%logo_type

    # Draw logo
    logo.draw(ax)

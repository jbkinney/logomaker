from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgba
import pdb

# From logomaker package
from data import SMALL
import character 
import color
import data

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

from data import load_alignment

def make_logo(mat,
              mat_type=None,
              logo_type=None,
              background=None,
              ylabel=None,
              ylim=None,
              **kwargs):
    '''
    '''

    # Get mat_type if not specified by user but
    # is specified in matrix
    if mat_type is None:
        try:
            mat_type = mat.logomaker_type
        except:
            mat_type = None
            logo_type = None

    # Validate df
    mat = data.validate_mat(mat)

    # Get background mat
    bg_mat = data.set_bg_mat(background, mat)

    if logo_type == 'counts':
        # Transform input mat to freq_mat
        mat = data.transform_mat(mat, from_type=mat_type, to_type='counts', background=bg_mat)
        ymax = mat.values.sum(axis=1).max()

        # Change default plot settings
        if ylim is None:
            ylim = [0, ymax]
        if ylabel is None:
            ylabel = 'counts'

    elif logo_type == 'probability':
        # Transform input mat to freq_mat
        mat = data.transform_mat(mat, from_type=mat_type, to_type='probability', background=bg_mat)

        # Change default plot settings
        if ylim is None:
            ylim = [0, 1]
        if ylabel is None:
            ylabel = 'probability'

    elif logo_type == 'information':
        # Transform input mat to info_mat
        mat = data.transform_mat(mat, from_type=mat_type, to_type='information', background=bg_mat)

        # Change default plot settings
        if ylim is None and (background is None):
            ylim = [0, np.log2(mat.shape[1])]
        if ylabel is None:
            ylabel = 'information\n(bits)'

    elif logo_type == 'enrichment':
        # Transform input mat to weight_mat
        mat = data.transform_mat(mat, from_type=mat_type, to_type='enrichment', background=bg_mat)

        # Change default plot settings
        if ylabel is None:
            ylabel = '$\log_2$\nenrichment'

    elif logo_type == 'energy':
        # Transform input mat to weight_mat
        mat = data.transform_mat(mat, from_type=mat_type, to_type='energy', background=background)

        # Change default plot settings
        if ylabel is None:
            ylabel = '- energy\n($k_B T$)'

    elif logo_type is None:
        mat = data.validate_mat(mat)
        ylabel = ''

    else:
        assert False, 'Error! logo_type %s is invalid' % logo_type

    # Create and return logo
    logo = Logo(matrix=mat, ylim=ylim, ylabel=ylabel, **kwargs)
    logo.logo_type = logo_type if logo_type is not None else 'generic'
    return logo


# Logo base class
class Logo:
    def __init__(self, matrix,
                 colors='classic',
                 characters=None,
                 alpha=1,
                 edgecolors='none',
                 edgewidth=0,
                 boxcolors='white',
                 boxalpha=0,
                 highlight_sequence=None,
                 highlight_colors=None,
                 highlight_alpha=None,
                 highlight_edgecolors=None,
                 highlight_edgewidth=None,
                 highlight_boxcolors=None,
                 highlight_boxalpha=None,
                 hpad = 0,
                 vpad = 0,
                 axes_style='classic',
                 font_family=None,
                 font_weight=None,
                 font_file=None,
                 font_style=None,
                 font_properties=None,
                 stack_order='big_on_top',
                 use_transparency=False,
                 max_alpha_val=None,
                 below_shade=1,
                 below_alpha=1,
                 below_flip=True,
                 baseline_width=.5,
                 xlabel='position',
                 ylabel=None,
                 xlim=None,
                 ylim=None,
                 xticks=None,
                 xticklabels=None):


        # Record user font input
        self.in_font_file = font_file
        self.in_font_style = font_style
        self.in_font_weight = font_weight
        self.in_font_family = font_family 
        self.in_font_properties = font_properties

        # If user supplies a FontProperties object, validate it
        if font_properties is not None:
            assert type(font_properties) == FontProperties
            self.font_properties = font_properties.copy()

        # Otherwise, create a FontProperties object based on user's input
        else:
            self.font_properties = FontProperties(family=self.in_font_family,
                                                  weight=self.in_font_weight,
                                                  fname=self.in_font_file,
                                                  style=self.in_font_style)
        # Set data
        self.in_df = matrix.copy()

        # Characters:
        # Restrict to provided characters if string or list of characters
        # Transform to provided characters if dictionary
        if characters is None:
            self.df = self.in_df.copy()
        elif isinstance(characters, dict):
            self.df = self.in_df.rename(columns=characters)
        elif isinstance(characters, (str, unicode, list, np.array)):
            characters = list(characters)
            self.df = self.in_df[characters]
        else:
            assert False, 'Error: cant interpret characters.'

        self.poss = self.df.index.copy()
        self.chars = np.array([str(c) for c in self.df.columns])
        self.L = len(self.poss)

        # Set normal character format
        self.facecolors = colors
        self.edgecolors = edgecolors
        self.edgewidth = edgewidth
        self.alpha = float(alpha)
        self.boxcolors = boxcolors
        self.boxalpha = float(boxalpha)
        self.hpad = hpad
        self.vpad = vpad

        # Set normal character color dicts
        self.facecolors_dict = color.get_color_dict(color_scheme=self.facecolors,
                                                    chars=self.chars,
                                                    alpha=self.alpha)
        self.edgecolors_dict = color.get_color_dict(color_scheme=self.edgecolors,
                                                    chars=self.chars,
                                                    alpha=self.alpha)
        self.boxcolors_dict = color.get_color_dict(color_scheme=self.boxcolors,
                                                   chars=self.chars,
                                                   alpha=self.boxalpha)

        # Set higlighted character format
        self.highlight_facecolors = highlight_colors if highlight_colors is not None else colors
        self.highlight_edgecolors = highlight_edgecolors if highlight_edgecolors is not None else edgecolors
        self.highlight_edgewidth = highlight_edgewidth if highlight_edgewidth is not None else edgewidth
        self.highlight_alpha = float(highlight_alpha) if highlight_alpha is not None else alpha
        self.highlight_boxcolors = highlight_boxcolors if highlight_boxcolors is not None else boxcolors
        self.highlight_boxalpha = float(highlight_boxalpha) if highlight_boxalpha is not None else boxalpha

        # Set highlight character color dicts
        self.highlight_facecolors_dict = color.get_color_dict(color_scheme=self.highlight_facecolors,
                                                              chars=self.chars,
                                                              alpha=self.highlight_alpha)
        self.highlight_edgecolors_dict = color.get_color_dict(color_scheme=self.highlight_edgecolors,
                                                              chars=self.chars,
                                                              alpha=self.highlight_alpha)
        self.highlight_boxcolors_dict = color.get_color_dict(color_scheme=self.highlight_boxcolors,
                                                             chars=self.chars,
                                                             alpha=self.highlight_alpha)

        # Set wild type sequence
        self.highlight_sequence = highlight_sequence
        if highlight_sequence is not None:
            assert isinstance(highlight_sequence, basestring)
            assert len(highlight_sequence) == len(self.poss)
            assert set(list(str(highlight_sequence))) == set(self.chars), \
                'Error: highlight_sequence %s contains invalid characters' % highlight_sequence
            self.use_highlight = True
        else:
            self.use_highlight = False

        # Set other character styling
        self.logo_style = axes_style
        self.stack_order = stack_order
        self.use_transparency = use_transparency
        self.neg_shade = float(below_shade)
        self.neg_alpha = float(below_alpha)
        self.neg_flip = below_flip
        self.max_alpha_val = max_alpha_val
        self.use_transparency = use_transparency

        # Compute characters and box
        self.compute_characters()

        # Set x axis params
        self.xlim = [self.bbox.xmin, self.bbox.xmax]\
            if xlim is None else xlim
        self.xticks = range(len(self.poss)) \
            if xticks is None else xticks
        self.xticklabels = self.poss\
            if xticklabels is None else xticklabels
        self.xlabel = xlabel

        # Set y axis params
        self.ylim = [self.bbox.ymin, self.bbox.ymax]\
            if ylim is None else ylim
        self.ylabel = ylabel

        # Set other formatting parameters
        self.floor_line_width=baseline_width


    def compute_characters(self):

        # Get largest value for computing transparency
        if self.max_alpha_val is None:
            max_alpha_val = abs(self.df.values).max()
        else:
            max_alpha_val = self.max_alpha_val

        char_list = []
        for i, pos in enumerate(self.poss):

            vals = self.df.iloc[i, :].values
            ymin = (vals * (vals < 0)).sum()

            # Reorder columns
            if self.stack_order == 'big_on_top':
                indices = np.argsort(vals)
            elif self.stack_order == 'small_on_top':
                indices = np.argsort(vals)[::-1]
            else:
                indices = range(len(vals))
            ordered_chars = self.chars[indices]

            # This is the same for every character
            x = float(i) - .5
            w = 1.0

            # Initialize y
            y = ymin

            for n, char in enumerate(ordered_chars):

                # Get value
                val = self.df.loc[pos, char]

                # Get height
                h = abs(val)
                if h < SMALL:
                    continue

                # Get facecolor, edgecolor, and edgewidth
                if self.use_highlight and (char == self.highlight_sequence[i]):
                    facecolor = self.highlight_facecolors_dict[char].copy()
                    edgecolor = self.highlight_edgecolors_dict[char].copy()
                    boxcolor = self.highlight_boxcolors_dict[char].copy()
                    boxalpha = self.highlight_boxalpha
                    edgewidth = self.highlight_edgewidth
                else:
                    facecolor = self.facecolors_dict[char].copy()
                    edgecolor = self.edgecolors_dict[char].copy()
                    boxcolor = self.boxcolors_dict[char].copy()
                    boxalpha = self.boxalpha
                    edgewidth = self.edgewidth

                # Get flip and shade character accordingly
                if val <= 0.0:
                    flip = self.neg_flip
                    shade = self.neg_shade
                    alpha = self.neg_alpha
                    facecolor = facecolor * np.array([shade, shade, shade, alpha])
                    edgecolor = edgecolor * np.array([shade, shade, shade, alpha])
                else:
                    flip = False

                # Set alpha
                if self.use_transparency:
                    alpha = h / max_alpha_val
                    if alpha > 1:
                        alpha = 1.0
                    elif alpha <= 0:
                        alpha = 0.0
                    facecolor[3] *= alpha
                    edgecolor[3] *= alpha


                # Create and store character
                char = character.Character(
                    c=char, xmin=x, ymin=y, width=w, height=h,
                    facecolor=facecolor, flip=flip,
                    font_properties = self.font_properties,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    boxcolor=boxcolor,
                    boxalpha=boxalpha,
                    hpad = self.hpad,
                    vpad = self.vpad)
                char_list.append(char)

                # Increment y
                y += h

        # Get box
        xmin = min([c.bbox.xmin for c in char_list])
        xmax = max([c.bbox.xmax for c in char_list])
        ymin = min([c.bbox.ymin for c in char_list])
        ymax = max([c.bbox.ymax for c in char_list])
        bbox = Bbox([[xmin, ymin], [xmax, ymax]])


        # Set char_list and box
        self.char_list = char_list
        self.bbox = bbox

    def draw(self, ax=None):
        if ax is None:
            ax = plt.gca()

        # Draw characters
        for char in self.char_list:
            char.draw(ax)

        # Draw floor line
        ax.axhline(0,color='k',linewidth=self.floor_line_width)

        # Logo-specific formatting
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if self.logo_style == 'classic':

            # x axis
            ax.set_xticks(self.xticks)
            ax.xaxis.set_tick_params(length=0)
            ax.set_xticklabels(self.xticklabels, rotation=90)

            # y axis
            ax.set_ylabel(self.ylabel)

            # box
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        elif self.logo_style == 'naked':

            # Turn everything off
            ax.axis('off')

        elif self.logo_style == 'rails':
            ax.set_xticks([])
            ax.set_yticks(self.ylim)
            ax.set_ylabel(self.ylabel)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)


        elif self.logo_style == 'light_rails':
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            [i.set_linewidth(0.5) for i in ax.spines.itervalues()]


        elif self.logo_style == 'everything':

            # x axis
            ax.set_xticks(self.xticks)
            ax.set_xticklabels(self.xticklabels)
            ax.set_xlabel(self.xlabel)

            # y axis
            ax.set_ylabel(self.ylabel)

            # box
            ax.axis('on')

        else:
            assert False, 'Error! Undefined logo_style=%s' % self.logo_style

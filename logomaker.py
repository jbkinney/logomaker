from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import pdb

# From logomaker package
from data import SMALL
import character 
import color
import data

from data import load_alignment

def make_logo(mat,
              mat_type=None,
              logo_type=None,
              background=None,
              ylabel=None,
              ylim=None,
              **kwargs):
    '''
    Primary function used to create logos
    :param mat:
    :param mat_type:
    :param logo_type:
    :param background:
    :param ylabel:
    :param ylim:
    :param kwargs:
    :return:
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

    else:
        assert False, 'Error! logo_type %s is invalid' % logo_type

    # Create and return logo
    logo = Logo(mat=mat, ylim=ylim, ylabel=ylabel, **kwargs)
    return logo


# Logo base class
class Logo:
    def __init__(self, mat,
                 wtseq=None,
                 color_scheme='classic',
                 logo_style='classic',
                 font_family=None,
                 font_weight=None,
                 font_file=None,
                 font_style=None,
                 font_properties=None,
                 stack_order='big_on_top',
                 use_transparency=False,
                 max_alpha_val=None,
                 neg_shade=.5,
                 neg_flip=True,
                 floor_line_width=.5,
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

        # Decide on final set of font properties based on this input
        self.set_font_properties()

        # Set data
        self.df = mat.copy()
        self.poss = mat.index.copy()
        self.chars = mat.columns.copy()
        self.wtseq = wtseq

        # Set colors
        self.color_scheme = color_scheme
        self.color_dict = color.get_color_dict(color_scheme=self.color_scheme, chars=self.chars) 

        # Set character style
        self.logo_style = logo_style
        self.stack_order = stack_order
        self.use_transparency = use_transparency
        self.neg_shade = neg_shade
        self.neg_flip = neg_flip
        self.max_alpha_val = max_alpha_val
        self.use_transparency = use_transparency

        # Compute characters and box
        self.compute_characters()

        # Set x axis params
        self.xlim = [self.bbox.xmin, self.bbox.xmax]\
            if xlim is None else xlim
        self.xticks = self.poss \
            if xticks is None else xticks
        self.xticklabels = ['%d' % x for x in self.xticks] \
            if xticklabels is None else xticklabels
        self.xlabel = xlabel

        # Set y axis params
        self.ylim = [self.bbox.ymin, self.bbox.ymax]\
            if ylim is None else ylim
        self.ylabel = ylabel

        # Set other formatting parameters
        self.floor_line_width=floor_line_width

    def set_font_properties(self):
        ''' Sets properties of font to use in logo '''

        # If user supplies a FontProperties object, validate it
        if not (self.in_font_properties is None):
            assert type(self.in_font_properties) == FontProperties
            self.font_properties = self.in_font_properties.copy()

        # Otherwise, create a FontProperties object based on user's input
        else:
            self.font_properties = FontProperties(family=self.in_font_family,
                                                  weight=self.in_font_weight, 
                                                  fname=self.in_font_file,
                                                  style=self.in_font_style)

    def compute_characters(self):

        # Get largest value for computing transparency
        if self.max_alpha_val is None:
            self.max_alpha_val = abs(self.df.values).max()

        char_list = []
        for i, pos in enumerate(self.poss):

            vals = self.df.loc[pos, :].values
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
            x = pos - .5
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

                # Get color
                color = self.color_dict[char]

                # Get flip, alpha, and shade
                if val >= 0.0:
                    alpha = 1.0
                    flip = False
                    shade = 1.0
                else:
                    alpha = self.neg_shade
                    flip = self.neg_flip
                    shade = self.neg_shade

                if self.use_transparency:
                    alpha *= h / self.max_alpha_val
                    if alpha > 1:
                        alpha = 1.0

                assert alpha <= 1.0, \
                    'Error: alpha=%f must be in [0,1]' % alpha

                # Create and store character
                char = character.Character(
                    c=char, xmin=x, ymin=y, width=w, height=h,
                    alpha=alpha, color=color, flip=flip,
                    shade=shade, 
                    font_properties = self.font_properties,
                    edgecolor='none')
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

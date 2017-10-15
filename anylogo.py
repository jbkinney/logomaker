from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import Box, SMALL
import color
from character import Character, get_fonts, get_default_font, validate_font
import data
import pdb

# Logo base class
class Logo:
    def __init__(self,
                 mat,
                 mat_type=None,
                 logo_type=None,
                 wtseq=None,
                 color_scheme='classic',
                 logo_style='classic',
                 font_name=None,
                 stack_order='big_on_top',
                 use_transparency=False,
                 max_alpha_val=None,
                 neg_shade=.5,
                 neg_flip=False,
                 floor_line_width=.5,
                 background=None,
                 xlabel='position',
                 ylabel=None,
                 xlim=None,
                 ylim=None,
                 xticks=None,
                 yticks=None,
                 xticklabels=None,
                 yticklabels=None):

        # Validate df
        mat = data.validate_mat(mat)

        # Get background mat
        bg_mat = data.set_bg_mat(background, mat)

        if logo_type == 'freq_logo':
            # Transform input mat to freq_mat
            mat = data.transform_mat(mat, from_type=mat_type, to_type='freq_mat', background=bg_mat)

            # Change default plot settings
            if ylim is None:
                ylim = [0, 1]
            if ylabel is None:
                ylabel = 'probability'

        elif logo_type == 'info_logo':
            # Transform input mat to info_mat
            mat = data.transform_mat(mat, from_type=mat_type, to_type='info_mat', background=bg_mat)

            # Get max info value
            self.max_info = max(mat.values.sum(axis=1))

            # Change default plot settings
            if ylim  is None and (background is None):
                ylim = [0, np.log2(mat.shape[1])]
            if ylabel is None:
                ylabel = 'information\n(bits)'

        elif logo_type == 'weight_logo':
            # Transform input mat to weight_mat
            mat = data.transform_mat(mat, from_type=mat_type, to_type='weight_mat', background=bg_mat)

            # Change default plot settings
            if ylabel is None:
                ylabel = 'score'

        elif logo_type == 'energy_logo':
            # Transform input mat to weight_mat
            mat = data.transform_mat(mat, from_type=mat_type, to_type='energy_mat', background=background)

            # Change default plot settings
            if ylabel is None:
                ylabel = '- energy\n($k_B T$)'

        elif logo_type is None:
            mat = data.validate_mat(mat)

        else:
            assert False, 'Error! logo_type %s is invalid'%logo_type

        # Set data
        self.df = mat.copy()
        self.poss = mat.index.copy()
        self.chars = mat.columns.copy()
        self.wtseq = wtseq

        # Set colors
        self.color_scheme = color_scheme
        self.color_dict = color.get_color_dict(color_scheme=self.color_scheme, chars=self.chars) 

        # Set character style
        self.font_name = validate_font(font_name)
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
        self.xlim = [self.box.xlb, self.box.xub]\
            if xlim is None else xlim
        self.xticks = self.poss \
            if xticks is None else xticks
        self.xticklabels = ['%d' % x for x in self.xticks] \
            if xticklabels is None else xticklabels
        self.xlabel = xlabel

        # Set y axis params
        self.ylim = [self.box.ylb, self.box.yub]\
            if ylim is None else ylim
        self.yticks = range(
            int(np.ceil(self.ylim[0])),
            int(np.floor(self.ylim[1])) + 1) \
            if yticks is None else yticks
        self.yticklabels = ['%d' % y for y in self.yticks]\
            if yticklabels is None else yticklabels
        self.ylabel = ylabel

        # Set other formatting parameters
        self.floor_line_width=floor_line_width

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
            ax.set_yticks(self.yticks)
            ax.set_yticklabels(self.yticklabels)
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
            ax.set_yticks(self.yticks)
            ax.set_yticklabels(self.yticklabels)
            ax.set_ylabel(self.ylabel)

            # box
            ax.axis('on')

        else:
            assert False, 'Error! Undefined logo_style=%s' % self.logo_style

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
                char = Character(
                    c=char, x=x, y=y, w=w, h=h,
                    alpha=alpha, color=color, flip=flip,
                    shade=shade, font_name=self.font_name, edgecolor='none')
                char_list.append(char)

                # Increment y
                y += h

        # Get box
        xlb = min([c.box.xlb for c in char_list])
        xub = max([c.box.xub for c in char_list])
        ylb = min([c.box.ylb for c in char_list])
        yub = max([c.box.yub for c in char_list])
        box = Box(xlb, xub, ylb, yub)

        # Set char_list and box
        self.char_list = char_list
        self.box = box

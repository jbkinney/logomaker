from __future__ import division
import numpy as np
import pandas as pd
import ast
import inspect
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, FontManager
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgba
import matplotlib as mpl
import pdb

# From logomaker package
from data import SMALL
import character 
import color
import data

from validate import validate_parameter, validate_mat

# Create global font manager instance. This takes a second or two
font_manager = FontManager()

def get_fontnames():
    font_names = [f.name for f in font_manager.ttflist] + \
                 [f.name for f in font_manager.afmlist]
    font_names = list(set(font_names))
    font_names.sort()
    return font_names

from data import load_alignment

def remove_none_from_dict(d):
    """ Removes None values from dictionary """
    assert isinstance(d,dict), 'Error: d is not a dictionary.'

    # Create new dictionary, this time without any Nones
    new_d = {}
    for key, value in d.items():
        if value is not None:
            new_d[key] = value
    return new_d

def make_logo(matrix=None,

              # Matrix transformation (make_logo only)
              matrix_type='counts',
              logo_type=None,
              background=None,
              pseudocount=1.0,
              enrichment_logbase=2,
              enrichment_centering=True,
              information_units='bits',
              counts_threshold=0,

              # Immediate drawing (make_logo only)
              figsize=None,
              draw_now=False,
              save_to_file=None,

              # Position choice
              position_range=None,
              shift_first_position_to=1,

              # Character choice
              sequence_type=None,
              characters=None,
              ignore_characters='.-',

              # Character formatting
              colors='gray',
              alpha=1,
              edgecolors='k',
              edgealpha=1,
              edgewidth=0,
              boxcolors='white',
              boxedgecolors='black',
              boxedgewidth=0,
              boxalpha=0,
              boxedgealpha=0,

              # Highlighted character formatting
              highlight_sequence=None,
              highlight_colors=None,
              highlight_alpha=None,
              highlight_edgecolors=None,
              highlight_edgewidth=None,
              highlight_edgealpha=None,
              highlight_boxcolors=None,
              highlight_boxalpha=None,
              highlight_boxedgecolors=None,
              highlight_boxedgewidth=None,
              highlight_boxedgealpha=None,

              # Character font
              font_properties=None,
              font_file=None,
              font_family=None,
              font_weight=None,
              font_style=None,
              font_name=None,

              # Character placement
              stack_order='big_on_top',
              use_transparency=False,
              max_alpha_val=None,
              below_shade=1.,
              below_alpha=1.,
              below_flip=True,
              hpad=0.,
              vpad=0.,
              width=1,
              uniform_stretch=False,
              max_stretched_character=None,

              # Special axes formatting
              axes_type='classic',
              style_sheet=None,
              baseline_width=.5,

              # Grid line formatting
              show_gridlines=False,
              gridline_axis=None,
              gridline_width=None,
              gridline_color=None,
              gridline_alpha=None,
              gridline_style=None,

              # Set other standard axes formatting
              show_binary_yaxis=False,
              xlim=None,
              xticks=None,
              xtick_spacing=None,
              xtick_anchor=0,
              xticklabels=None,
              xtick_rotation=None,
              xtick_length=None,
              xtick_format=None,
              xlabel=None,
              ylim=None,
              yticks=None,
              yticklabels=None,
              ytick_rotation=None,
              ytick_length=None,
              ytick_format=None,
              ylabel=None,
              title=None,
              left_spine=None,
              right_spine=None,
              top_spine=None,
              bottom_spine=None,
              use_tightlayout=False,

              # Default axes font
              axes_fontfile=None,
              axes_fontfamily='sans',
              axes_fontweight=None,
              axes_fontstyle=None,
              axes_fontsize=10,
              axes_fontname=None,

              # tick font
              tick_fontfile=None,
              tick_fontfamily=None,
              tick_fontweight=None,
              tick_fontstyle=None,
              tick_fontsize=None,
              tick_fontname=None,

              # label font
              label_fontfile=None,
              label_fontfamily=None,
              label_fontweight=None,
              label_fontstyle=None,
              label_fontsize=None,
              label_fontname=None,

              # title font
              title_fontfile=None,
              title_fontfamily=None,
              title_fontweight=None,
              title_fontstyle=None,
              title_fontsize=None,
              title_fontname=None,

              # Any other axes formatting
              rcparams={}):
    """
    """

    ######################################################################
    # Validate all parameters

    names, vargs, kwargs, default_values = inspect.getargspec(make_logo)
    user_values = [eval(name) for name in names]

    assert len(names)==len(default_values), \
        'len(names)==%d does not match len(default_values)==%d' %\
        (len(names), len(default_values))

    for name, user_value, default_value in \
            zip(names, user_values, default_values):

        # Validate parameter value
        valid_value = validate_parameter(name, user_value, default_value)

        # Set parameter value equal to the valid value
        exec("%s = valid_value" % name)

    ######################################################################
    # matrix.columns

    # Filter matrix columns based on sequence and character specifications
    matrix = data.filter_columns(matrix=matrix,
                                 sequence_type=sequence_type,
                                 characters=characters,
                                 ignore_characters=ignore_characters)
    characters = matrix.columns

    ######################################################################
    # matrix.index

    # If matrix_type is counts, remove positions with too few counts
    if matrix_type == 'counts':
        position_counts = matrix.values.sum(axis=1)
        matrix = matrix.loc[position_counts >= counts_threshold, :]

    # Restrict to specific position range if requested
    if position_range is not None:
        min = position_range[0]
        max = position_range[1]
        matrix = matrix.loc[(matrix.index >= min) & (matrix.index < max), :]

    # Matrix length is now set. Record it.
    L = len(matrix)

    # Shift positions to requested start
    positions = range(shift_first_position_to, shift_first_position_to + L)
    matrix['pos'] = positions
    matrix.set_index('pos', inplace=True, drop=True)
    matrix = validate_mat(matrix)
    positions = matrix.index

    ######################################################################
    # matrix.values

    # Set logo_type equal to matrix_type if is currently None
    if logo_type is None:
        logo_type = matrix_type
    logo_type = validate_parameter('logo_type', logo_type, None)

    # Get background matrix
    bg_mat = data.set_bg_mat(background, matrix)

    # Transform matrix:
    matrix = data.transform_mat(matrix=matrix,
                                from_type=matrix_type,
                                to_type=logo_type,
                                pseudocount=pseudocount,
                                background=bg_mat,
                                enrichment_logbase=enrichment_logbase,
                                enrichment_centering=enrichment_centering,
                                information_units=information_units)

    ######################################################################
    # font_properties

    # If font_properties is set directly by user, validate it
    if font_properties is not None:
        assert isinstance(font_properties, FontProperties), \
            'Error: font_properties is not an instance of FontProperties.'
    # Otherwise, create font_properties from other font information
    else:

        # Look up font file if name provided instead
        if font_file is None and font_name is not None:
            font_file = font_manager.findfont(font_name)

        # Create properties
        font_properties = FontProperties(family=font_family,
                                         weight=font_weight,
                                         fname=font_file,
                                         style=font_style)

    ######################################################################
    # character_style

    character_style = {
        'facecolors': color.get_color_dict(color_scheme=colors,
                                           chars=characters,
                                           alpha=alpha),
        'edgecolors': color.get_color_dict(color_scheme=edgecolors,
                                           chars=characters,
                                           alpha=edgealpha),
        'boxcolors': color.get_color_dict(color_scheme=boxcolors,
                                          chars=characters,
                                          alpha=boxalpha),
        'boxedgecolors': color.get_color_dict(color_scheme=boxedgecolors,
                                              chars=characters,
                                              alpha=boxedgealpha),
        'edgewidth': edgewidth,
        'boxedgewidth': boxedgewidth,
    }

    ######################################################################
    # highlight_style

    # Set higlighted character format
    highlight_colors = highlight_colors \
        if highlight_colors is not None \
        else colors
    highlight_alpha = float(highlight_alpha) \
        if highlight_alpha is not None \
        else alpha
    highlight_edgecolors = highlight_edgecolors \
        if highlight_edgecolors is not None \
        else edgecolors
    highlight_edgewidth = highlight_edgewidth \
        if highlight_edgewidth is not None \
        else edgewidth
    highlight_edgealpha = float(highlight_edgealpha) \
        if highlight_edgealpha is not None \
        else edgealpha
    highlight_boxcolors = highlight_boxcolors \
        if highlight_boxcolors is not None \
        else boxcolors
    highlight_boxalpha = float(highlight_boxalpha) \
        if highlight_boxalpha is not None \
        else boxalpha
    highlight_boxedgecolors = highlight_boxedgecolors \
        if highlight_boxedgecolors is not None \
        else boxedgecolors
    highlight_boxedgewidth = highlight_boxedgewidth \
        if highlight_boxedgewidth is not None \
        else boxedgewidth
    highlight_boxedgealpha = highlight_boxedgealpha \
        if highlight_boxedgealpha is not None \
        else boxedgealpha

    highlight_style = {
        'facecolors': color.get_color_dict(color_scheme=highlight_colors,
                                           chars=characters,
                                           alpha=highlight_alpha),
        'edgecolors': color.get_color_dict(color_scheme=highlight_edgecolors,
                                           chars=characters,
                                           alpha=highlight_edgealpha),
        'boxcolors': color.get_color_dict(color_scheme=highlight_boxcolors,
                                          chars=characters,
                                          alpha=highlight_boxalpha),
        'boxedgecolors': color.get_color_dict(
                                          color_scheme=highlight_boxedgecolors,
                                          chars=characters,
                                          alpha=highlight_boxedgealpha),
        'edgewidth': highlight_edgewidth,
        'boxedgewidth': highlight_boxedgewidth,
    }

    ######################################################################
    # placement_style

    placement_style = {
        'stack_order': stack_order,
        'use_transparency': use_transparency,
        'max_alpha_val': max_alpha_val,
        'below_shade': below_shade,
        'below_alpha': below_alpha,
        'below_flip': below_flip,
        'hpad': hpad,
        'vpad': vpad,
        'width': width,
        'uniform_stretch': uniform_stretch,
        'max_stretched_character': max_stretched_character
    }

    ######################################################################
    # axes_style

    # Modify ylim and ylabel according to logo_type
    if logo_type == 'counts':
        ymax = matrix.values.sum(axis=1).max()
        if ylim is None:
            ylim = [0, ymax]
        if ylabel is None and axes_type != 'naked':
            ylabel = 'counts'
    elif logo_type == 'probability':
        if ylim is None:
            ylim = [0, 1]
        if ylabel is None and axes_type != 'naked':
            ylabel = 'probability'
    elif logo_type == 'information':
        if ylim is None and (background is None):
            ylim = [0, np.log2(matrix.shape[1])]
        if ylabel is None and axes_type != 'naked':
            ylabel = 'information\n(%s)' % information_units
    elif logo_type == 'enrichment':
        if ylabel is None and axes_type != 'naked':
            ylabel = 'enrichment\n'
            if enrichment_logbase == 2:
                ylabel += '($\log_2$)'
            elif enrichment_logbase == 10:
                ylabel += '($\log_{10})$'
            elif enrichment_logbase == np.e:
                ylabel += '$(\ln)$ enrichment'
            else:
                assert False, 'Error: invalid choice of enrichment_logbase=%f'\
                              % enrichment_logbase
    else:
        if ylabel is None:
            ylabel = ''

    # If showing binary yaxis, symmetrize ylim and set yticks to +/-
    if show_binary_yaxis:
        if ylim is None:
            y = np.max(abs(ylim[0]), abs(ylim[1]))
            ylim = [-y, y]
        if yticks is None:
            yticks = ylim
        if yticklabels is None:
            yticklabels = ['$-$', '$+$']
        if ytick_length is None:
            ytick_length = 0

    # Set ylim (will not be None)
    if ylim is None:
        ymax = (matrix.values * (matrix.values > 0)).sum(axis=1).max()
        ymin = (matrix.values * (matrix.values < 0)).sum(axis=1).min()
        ylim = [ymin, ymax]

    # Set xlim (will not be None)
    if xlim is None:
        xmin = matrix.index.min() - .5
        xmax = matrix.index.max() + .5
        xlim = [xmin, xmax]

    # Set xticks
    if not (axes_type in ['rails', 'naked']):
        xtick_spacing = 1
    if xticks is None and xtick_spacing is not None:
        xticks = [pos for pos in positions if
                  (pos - xtick_anchor) % xtick_spacing == 0.0]

    # If axes_type is specified, make additional modifications
    if axes_type == 'classic':
        if xtick_length is None:
            xtick_length = 0
        if xtick_rotation is None:
            xtick_rotation = 90
        if xlabel is None:
            xlabel = 'position'
        if left_spine is None:
            left_spine = True
        if right_spine is None:
            right_spine = False
        if top_spine is None:
            top_spine = False
        if bottom_spine is None:
            bottom_spine = True

    elif axes_type == 'naked':
        if xticks is None:
            xticks = []
        if xlabel is None:
            xlabel = ''
        if yticks is None:
            yticks = []
        if ylabel is None:
            ylabel = ''
        if left_spine is None:
            left_spine = False
        if right_spine is None:
            right_spine = False
        if top_spine is None:
            top_spine = False
        if bottom_spine is None:
            bottom_spine = False

    elif axes_type == 'rails':
        if xticks is None:
            xticks = []
        if xlabel is None:
            xlabel = ''
        if yticks is None and ylim is not None:
            yticks = ylim
        if left_spine is None:
            left_spine = False
        if right_spine is None:
            right_spine = False
        if top_spine is None:
            top_spine = True
        if bottom_spine is None:
            bottom_spine = True

    elif axes_type == 'everything':
        if xlabel is None:
            xlabel = 'position'
        if left_spine is None:
            left_spine = True
        if right_spine is None:
            right_spine = True
        if top_spine is None:
            top_spine = True
        if bottom_spine is None:
            bottom_spine = True

    elif axes_type == 'vlines':
        if xtick_length is None:
            xtick_length = 0
        if xlabel is None:
            xlabel = 'position'
        if left_spine is None:
            left_spine = False
        if right_spine is None:
            right_spine = False
        if top_spine is None:
            top_spine = False
        if bottom_spine is None:
            bottom_spine = False

    # Set label rotation
    if xtick_rotation is None:
        xtick_rotation = 0
    if ytick_rotation is None:
        ytick_rotation = 0

    if title is None:
        title = ''

    # Translate font names into font files
    if axes_fontfile is None and axes_fontname is not None:
        axes_fontfile = font_manager.findfont(axes_fontname)
    if tick_fontfile is None and tick_fontname is not None:
        tick_fontfile = font_manager.findfont(tick_fontname)
    if label_fontfile is None and label_fontname is not None:
        label_fontfile = font_manager.findfont(label_fontname)
    if title_fontfile is None and title_fontname is not None:
        title_fontfile = font_manager.findfont(title_fontname)


    # Default font for all axes elements
    axes_fontdict = {
        'fname': axes_fontfile,
        'family': axes_fontfamily,
        'weight': axes_fontweight,
        'style': axes_fontstyle,
        'size': axes_fontsize,
    }
    axes_fontdict = remove_none_from_dict(axes_fontdict)

    # Font for x and y axis tickmarks
    tick_fontdict = {
        'fname': tick_fontfile,
        'family': tick_fontfamily,
        'weight': tick_fontweight,
        'style': tick_fontstyle,
        'size': tick_fontsize,
    }
    tick_fontdict = remove_none_from_dict(tick_fontdict)
    tick_fontdict = dict(axes_fontdict, **tick_fontdict)
    tick_fontproperties = FontProperties(**tick_fontdict)

    # Font for x and y axis labels
    label_fontdict = {
        'fname': label_fontfile,
        'family': label_fontfamily,
        'weight': label_fontweight,
        'style': label_fontstyle,
        'size': label_fontsize,
    }
    label_fontdict = remove_none_from_dict(label_fontdict)
    label_fontdict = dict(axes_fontdict, **label_fontdict)
    label_fontproperties = FontProperties(**label_fontdict)

    # Font for title
    title_fontdict = {
        'fname': title_fontfile,
        'family': title_fontfamily,
        'weight': title_fontweight,
        'style': title_fontstyle,
        'size': title_fontsize,
    }
    title_fontdict = remove_none_from_dict(title_fontdict)
    title_fontdict = dict(axes_fontdict, **title_fontdict)
    title_fontproperties = FontProperties(**title_fontdict)

    # Gridline styling
    gridline_dict = {
        'axis': gridline_axis,
        'alpha': gridline_alpha,
        'color': gridline_color,
        'linewidth': gridline_width,
        'linestyle': gridline_style,
    }
    gridline_dict = remove_none_from_dict(gridline_dict)

    # Set axes_style dictionary
    axes_style = {
        'baseline_width': baseline_width,
        'title': title,
        'ylim': ylim,
        'yticks': yticks,
        'yticklabels': yticklabels,
        'ylabel': ylabel,
        'xlim': xlim,
        'xticks': xticks,
        'xticklabels': xticklabels,
        'xlabel': xlabel,
        'left_spine': left_spine,
        'right_spine': right_spine,
        'top_spine': top_spine,
        'bottom_spine': bottom_spine,
        'xtick_length': xtick_length,
        'xtick_rotation': xtick_rotation,
        'xtick_format': xtick_format,
        'ytick_length': ytick_length,
        'ytick_rotation': ytick_rotation,
        'ytick_format': ytick_format,
        'font_dict': axes_fontdict.copy(),
        'tick_fontproperties': tick_fontproperties,
        'label_fontproperties': label_fontproperties,
        'title_fontproperties': title_fontproperties,
        'show_gridlines': show_gridlines,
        'gridline_dict': gridline_dict,
        'use_tightlayout': use_tightlayout,
    }

    ######################################################################
    # Create Logo instance
    logo = Logo(matrix=matrix,
                highlight_sequence=highlight_sequence,
                font_properties=font_properties,
                character_style=character_style,
                highlight_style=highlight_style,
                placement_style=placement_style,
                axes_style=axes_style)

    # Decorate logo
    logo.logo_type = logo_type
    logo.background = background
    logo.bg_mat = bg_mat

    ######################################################################
    # Optionally draw logo

    # Set style sheet:
    if style_sheet is not None:
        plt.style.use(style_sheet)

    # Set RC parameters
    for key, value in rcparams.items():
        mpl.rcParams[key] = value

    # If user specifies a figure size, make figure and axis,
    # draw logo, then return all three
    if figsize is not None:

        fig, ax = plt.subplots(figsize=figsize)
        logo.draw(ax)

        if use_tightlayout:
            plt.tight_layout()
            plt.draw()

        if save_to_file:
            fig.savefig(save_to_file)
        plt.draw()

        return logo, ax, fig

    # If draw_now, get current axes, draw logo, and return both
    elif draw_now:
        ax = plt.gca()
        logo.draw(ax)

        if use_tightlayout:
            plt.tight_layout()
            plt.draw()

        return logo, ax

    # Otherwise, just return logo to user
    else:
        return logo


# Logo base class
class Logo:
    def __init__(self,
                 matrix,
                 font_properties,
                 character_style,
                 highlight_sequence,
                 highlight_style,
                 placement_style,
                 axes_style):

        # Set font properties
        self.font_properties = font_properties

        # Set matrix
        self.df = matrix.copy()

        # Get list of characters
        self.chars = np.array([str(c) for c in self.df.columns])

        # Get list of positions
        self.poss = self.df.index.copy()

        # Set positions
        self.L = len(self.df)

        # Set character styles
        self.character_style = character_style.copy()
        self.highlight_style = highlight_style.copy()
        self.placement_style = placement_style.copy()

        # Set axes style
        self.axes_style = axes_style.copy()

        # Set wild type sequence
        self.highlight_sequence = highlight_sequence
        if self.highlight_sequence is not None:
            assert isinstance(self.highlight_sequence, basestring), \
                'Error: highlight_sequence is not a string.'
            assert len(self.highlight_sequence ) == len(self.poss), \
                'Error: highlight_sequence has a different length than matrix.'
            assert set(list(str(self.highlight_sequence))) == set(self.chars),\
                'Error: highlight_sequence %s contains invalid characters'\
                %self.highlight_sequence
            self.use_highlight = True
        else:
            self.use_highlight = False

        # Compute characters and box
        self.compute_characters()


    def compute_characters(self):

        # Get largest value for computing transparency
        max_alpha_val = self.placement_style['max_alpha_val']
        if max_alpha_val is None:
            max_alpha_val = abs(self.df.values).max()

        # Compute hstretch values for all characters
        width = self.placement_style['width']
        hpad = self.placement_style['hpad']
        vpad = self.placement_style['vpad']
        hstretch_dict, vstretch_dict = \
            character.get_stretch_vals(self.chars,
                                    width=width,
                                    height=1,
                                    font_properties=self.font_properties,
                                    hpad=hpad,
                                    vpad=vpad)
        # Set max_hstretch
        uniform_stretch = self.placement_style['uniform_stretch']
        max_stretched_character = \
            self.placement_style['max_stretched_character']
        if uniform_stretch:
            max_hstretch = min(hstretch_dict.values())
        elif max_stretched_character:
            max_hstretch = hstretch_dict[max_stretched_character]
        else:
            max_hstretch = np.Inf

        char_list = []
        for i, pos in enumerate(self.poss):

            vals = self.df.iloc[i, :].values
            ymin = vals[vals < 0].sum()

            # Reorder columns
            stack_order = self.placement_style['stack_order']
            if stack_order == 'big_on_top':
                indices = np.argsort(vals)
            elif stack_order == 'small_on_top':
                tmp_indices = np.argsort(vals)
                pos_tmp_indices = tmp_indices[vals[tmp_indices] >= 0]
                neg_tmp_indices = tmp_indices[vals[tmp_indices] < 0]
                indices = np.array(list(neg_tmp_indices[::-1]) +
                                   list(pos_tmp_indices[::-1]))
            elif stack_order == 'fixed':
                indices = range(len(vals))
            else:
                assert False, 'Error: unrecognized stack_order value %s.'%\
                              stack_order
            ordered_chars = self.chars[indices]

            # This is the same for every character
            x = pos - width/2.0
            w = width

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
                    facecolor = self.highlight_style['facecolors'][char].copy()
                    edgecolor = self.highlight_style['edgecolors'][char].copy()
                    boxcolor = self.highlight_style['boxcolors'][char].copy()
                    boxedgecolor = self.highlight_style['boxedgecolors'][char]\
                        .copy()
                    edgewidth = self.highlight_style['edgewidth']
                    boxedgewidth = self.highlight_style['boxedgewidth']
                else:
                    facecolor = self.character_style['facecolors'][char].copy()
                    edgecolor = self.character_style['edgecolors'][char].copy()
                    boxcolor = self.character_style['boxcolors'][char].copy()
                    boxedgecolor = self.character_style['boxedgecolors'][char]\
                        .copy()
                    edgewidth = self.character_style['edgewidth']
                    boxedgewidth = self.character_style['boxedgewidth']

                # Get flip and shade character accordingly
                if val <= 0.0:
                    flip = self.placement_style['below_flip']
                    shade = self.placement_style['below_shade']
                    alpha = self.placement_style['below_alpha']
                    facecolor *= np.array([shade, shade, shade, alpha])
                    edgecolor *= np.array([shade, shade, shade, alpha])
                else:
                    flip = False

                # Set alpha
                if self.placement_style['use_transparency']:
                    alpha = h / max_alpha_val
                    if alpha > 1:
                        alpha = 1.0
                    elif alpha <= 0:
                        alpha = 0.0
                    facecolor[3] *= alpha
                    edgecolor[3] *= alpha

                # Create Character object
                char = character.Character(
                    c=char, xmin=x, ymin=y, width=w, height=h, flip=flip,
                    font_properties=self.font_properties,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    boxcolor=boxcolor,
                    boxedgecolor=boxedgecolor,
                    boxedgewidth=boxedgewidth,
                    hpad=hpad,
                    vpad=vpad,
                    max_hstretch=max_hstretch)

                # Store Character object
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

        # If no axes is provided, default to current axes
        if ax is None:
            ax = plt.gca()

        # Draw characters
        for char in self.char_list:
            char.draw(ax)

        # Draw floor line
        ax.axhline(0, color='k', linewidth=self.axes_style['baseline_width'])

        # Draw gridlines
        if self.axes_style['show_gridlines']:
            ax.grid(**self.axes_style['gridline_dict'])

        # Set limits
        ax.set_xlim(self.axes_style['xlim'])
        ax.set_ylim(self.axes_style['ylim'])

        # Draw x-axis annotation
        if self.axes_style['xticks'] is not None:
            ax.set_xticks(self.axes_style['xticks'])

        if self.axes_style['yticks'] is not None:
            ax.set_yticks(self.axes_style['yticks'])

        if self.axes_style['xlabel'] is not None:
            ax.set_xlabel(self.axes_style['xlabel'],
                    font_properties=self.axes_style['label_fontproperties'])

        if self.axes_style['ylabel'] is not None:
            ax.set_ylabel(self.axes_style['ylabel'],
                    font_properties=self.axes_style['label_fontproperties'])

        if self.axes_style['title'] is not None:
            ax.set_title(self.axes_style['title'],
                    font_properties=self.axes_style['title_fontproperties'])

        if self.axes_style['xtick_length'] is not None:
            ax.xaxis.set_tick_params(length=self.axes_style['xtick_length'])

        if self.axes_style['ytick_length'] is not None:
            ax.yaxis.set_tick_params(length=self.axes_style['ytick_length'])

        if self.axes_style['left_spine'] is not None:
            ax.spines['left'].set_visible(self.axes_style['left_spine'])

        if self.axes_style['right_spine'] is not None:
            ax.spines['right'].set_visible(self.axes_style['right_spine'])

        if self.axes_style['top_spine'] is not None:
            ax.spines['top'].set_visible(self.axes_style['top_spine'])

        if self.axes_style['bottom_spine'] is not None:
            ax.spines['bottom'].set_visible(self.axes_style['bottom_spine'])

        # Initial rendering needed to get axes tick labels
        if self.axes_style['use_tightlayout']:
            plt.tight_layout()
        plt.draw()

        if self.axes_style['xticklabels'] is not None:
            xticklabels = self.axes_style['xticklabels']
        elif self.axes_style['xtick_format'] is not None:
            xticklabels = [self.axes_style['xtick_format'] % x
                           for x in ax.get_xticks()]
        else:
            xticklabels = ax.get_xticklabels()

        ax.set_xticklabels(xticklabels,
                        rotation=self.axes_style['xtick_rotation'],
                        font_properties=self.axes_style['tick_fontproperties'])

        if self.axes_style['yticklabels'] is not None:
            yticklabels = self.axes_style['yticklabels']
        elif self.axes_style['ytick_format'] is not None:
            yticklabels = [self.axes_style['ytick_format'] % y
                           for y in ax.get_yticks()]
        else:
            yticklabels = ax.get_yticklabels()

        ax.set_yticklabels(yticklabels,
                        rotation=self.axes_style['ytick_rotation'],
                        font_properties=self.axes_style['tick_fontproperties'])

        # Do final drawing
        if self.axes_style['use_tightlayout']:
            plt.tight_layout()
        plt.draw()

def make_styled_logo(style_file=None,
                     style_dict=None,
                     print_params=True,
                     print_warnings=True,
                     *args, **user_kwargs):
    """
    Description:

        Generates a logo using default parameters specified in a style file that
        can be overwritten by the user. For detailed information on all
        possible arguments, see make_logo()

    Return:

        logo (logomaker.Logo): a rendered logo.

    Args:

        style_file (str): file containing default keyword arguments.

        style_dict (dict): dictionary containing style specifications.
            Overrides style_file specifications.

        print_params (bool): whether to print the specified parameters
            to stdout.

        print_warnings (bool): whether to print warnings to stderr.

        args (list): standard args list passed by user

        user_kwargs (dict): user-specified keyword arguments specifying style.
            Overrides both style_file and style_dict specifications.
    """

    # Copy kwargs explicitly specified by user
    kwargs = user_kwargs

    # If user provides a style dict
    if style_dict is not None:
        kwargs = dict(style_dict, **kwargs)

    # If user provides a parameters file
    if style_file is not None:
        file_kwargs = load_parameters(style_file, print_params, print_warnings)
        kwargs = dict(file_kwargs, **kwargs)

    # Make logo
    logo = make_logo(*args, **kwargs)

    # Return logo to user
    return logo


def load_parameters(file_name, print_params=True, print_warnings=True):
    """
    Description:

        Fills a dictionary with parameters parsed from specified file.

    Arg:

        file_name (str): Name of file containing parameter assignments.

        print_params (bool): whether to print the specified parameters
            to stdout.

        print_warnings (bool): whether to print warnings to stderr.

    Return:

        params_dict (dict): Dictionary containing parameter assignments
            parsed from parameters file
    """

    # Create dictionary to hold results
    params_dict = {}

    # Create regular expression for parsing parameter file lines
    pattern = re.compile(
        '^\s*(?P<param_name>[\w]+)\s*[:=]\s*(?P<param_value>.*)$'
    )

    # Quit if file_name is not specified
    if file_name is None:
        return params_dict

    # Open parameters file
    try:
        file_obj = open(file_name, 'r')
    except IOError:
        print('Error: could not open file %s for reading.' % file_name)
        raise IOError

    # Process each line of file and store resulting parameter values
    # in params_dict
    params_dict = {}
    prefix = '' # This is needed to parse multi-line files
    for line in file_obj:

        # removing leading and trailing whitespace
        line = prefix + line.strip()

        # Ignore lines that are empty or start with comment
        if (len(line) == 0) or line.startswith('#'):
            continue

        # Record current line plus a space in prefix, then continue to next
        # if line ends in a "\"
        if line[-1] == '\\':
            prefix += line[:-1] + ' '
            continue

        # Otherwise, clean prefix and continue with this parsing
        else:
            prefix = ''

        # Parse line using pattern
        match = re.match(pattern, line)

        # If line matches, record parameter name and value
        if match:
            param_name = match.group('param_name')
            param_value_str = match.group('param_value')

            # Evaluate parameter value as Python literal
            try:
                params_dict[param_name] = ast.literal_eval(param_value_str)
                if print_params:
                    print('[set] %s = %s' % (param_name, param_value_str))

            except ValueError:
                if print_warnings:
                    print(('Warning: could not set parameter "%s" because ' +
                          'could not interpret "%s" as literal.') %
                          (param_name, param_value_str))
            except SyntaxError:
                if print_warnings:
                    print(('Warning: could not set parameter "%s" because ' +
                          'of a syntax error in "%s".') %
                          (param_name, param_value_str))


        elif print_warnings:
            print('Warning: could not parse line "%s".' % line)

    return params_dict


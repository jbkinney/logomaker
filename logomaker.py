from __future__ import division
import numpy as np
import pandas as pd
import ast
import inspect
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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

from data import load_alignment

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
              baseline_width=.5,
              vline_width=1,
              vline_color='gray',
              show_vlines=None,
              show_binary_yaxis=False,

              # Standard axes formatting
              xlim=None,
              xticks=None,
              xtick_spacing=None,
              xtick_anchor=0,
              xticklabels=None,
              xtick_rotation=None,
              xtick_length=None,
              xlabel=None,
              ylim=None,
              yticks=None,
              yticklabels=None,
              ytick_rotation=None,
              ytick_length=None,
              ylabel=None,
              title=None,
              left_spine=None,
              right_spine=None,
              top_spine=None,
              bottom_spine=None,
              rcparams={}):
    """
    Description:

        Returns a logo representing the data matrix provided.

    Returns:

        logo (logomaker.Logo): Logo object. Draw by using logo.draw(ax).

    Args:

        matrix (pd.DataFrame): Data matrix used to make the logo. Row names are
            the positions, column names are the characters. In what follows, L
            refers to the number of rows.

        parameters_file (str): Name of a file containing parameters. Entries in
            this file override any other keyword arguments provided to
            this function.

        matrix_type: Type of matrix provided. Value can be 'counts',
            'probability', 'enrichment', or 'information'.

        logo_type (str): Type of logo to display. Value can be 'counts',
            'probability', 'enrichment', 'information', or
            None. Defaults to matrix_type if provided.

        background: [WRITE]

        pseudocount (float): For converting a counts matrix to a probability
            matrix. Must be >= 0.

        enrichment_logbase (str): Logarithm to use when computing enrichment.
            Value can be '2', '10', or 'e'. [IMPLEMENT]

        information_units (str): Units to use when computing information logos.
            Values can be 'bits' or 'nats'. [IMPLEMENT]

        colors (str, list, or dict): Interior colors of logo characters. Can
            take a variety of inputs:
            - string, specifying a LogoMaker color scheme. Options are,
                + For DNA/RNA: 'classic', 'grays', 'base_paring'.
                    [IMPLEMENT]
                + For protein: 'hydrophobicity', 'chemistry', 'charge'.
                    [IMPLEMENT]
            - string, listing a color name  such as 'k' or 'tomato'
            - string, listing a colormap name such as 'viridis' or 'Purples'
            - list, specifying an RGB color or RGBA color.
            - dictionary) mapping characters to colors, such as
                {'A': 'green',
                'C':[ 0.,  0.,  1.],
                'G':'y', 'T':[ 1.,  0.,  0.,  1.]}
                [IMPLEMENT CHECKING FOR PROPER CHARS]

        characters (str, list, or dict): Characters to be used in the logo. Can
            take a variety of inputs:
            - string, such as 'ACGT', listing the matrix columns to be used.
            - list, such as ['A','C','G','T'], listing the matrix columns to
                be used. LogoMaker provides pre-specified lists, including,
                + DNA: uppercase deoxynuclotides
                + RNA: uppercase ribonucleotides
                + PROTEIN: uppercase amino acids
                + PROTEIN_STOP: same as PROTEIN but with '*' added
                + dna, rna, protein, protein_stop: same as above but lowercase
            - dictionary, such as {'T':'U'}, listing which characters to
            rename. LogoMaker provides pre-specified dictionaries, including
                + to_DNA: transform to uppercase deoxynuclotides
                + to_RNA: transform to uppercase ribonucleotides
                + to_PROTEIN: transform to uppercase amino acids
                + to_dna, to_rna, to_protein: same as above but transform to
                    lowercase characters

        alpha (float): Opacity of logo characters. Values are restricted to
            interval [0, 1].

        edgecolors (str, list, or dict): Edge colors of logo characters. Same
            inputs as parameter "colors".

        edgewidth (float): Width of logo character edges. Values are restricted
            to be  >= 0.

        boxcolors (str, list): Color of the box in which each logo character is
            drawn. Can take a variety of inputs:
            - string, listing a color name  such as 'k' or 'tomato'
            - list, specifying an RGB color or RGBA color.

        boxalpha (float): Opacity of each logo character bounding box. Values
            are restricted to interval [0, 1].

        highlight_sequence (str, pd.DataFrame): Logo characters to highlight.
            Can take a variety of inputs:
            - string, listing a sequence with the same length and characters
                as logo.
            - dataframe, having the same rows and columns as matrix, with
                boolean elements indicating which characters to highlight at
                which positions
                [IMPLEMENT!]

        highlight_colors (str, list, or dictionary): Interior colors of
            highlighted logo characters. Same inputs as parameter "colors".

        highlight_alpha (float): Opacity of each highlighted logo character.
            Values are restricted to interval [0, 1].

        highlight_edgecolors (str, list, or dict): Edge colors of highlighted
            logo characters. Same inputs as parameter "colors".

        highlight_edgewidth (float): Width of highlighted logo character edges.
            Values are restricted to be >= 0.

        highlight_boxcolors (string, list, or dictionary): Color of the box in
            which each highlighted logo character is drawn.Same inputs as
            parameter "colors".

        highlight_boxalpha (float in interval [0, 1]): Opacity of each
            highlighted logo character bounding box.

        hpad (float): Horizonal padding for each logo character. Quanitifies
            whitespace, evenly split across both sides, as a fraction of
            character width.

        vpad (float): Vertical padding for each logo character. Quanitifies
            whitespace, evenly split on top and bottom, as a fraction of
            character height.

        axes_style (string): Styles the logo axes. Options are 'classic',
            'rails', 'light_rails', 'naked', and 'everything'.

        font_family (string or list of strings): The logo character font name.
            Specifically, the value passed as the 'family' parameter when
            calling the matplotlib.font_manager.FontProperties constructor.
            From matplotlib documentation:
                "family: A list of font names in decreasing order of priority.
                The items may include a generic font family name, either
                'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.
                In that case, the actual font to be used will be looked up from
                the associated rcParam in matplotlibrc."

        font_weight (string or float): The logo character font weight.
            Specifically, the value passed as the 'weight' parameter in the
            matplotlib.font_manager.FontProperties constructor. From matplotlib
            documentation:
                "weight: A numeric value in the range 0-1000 or one of
                'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                'extra bold', 'black'."

        font_file (string): The file specifying the logo character font.
            Specifically, the value passed as the 'fname' parameter in the
            matplotlib.font_manager.FontProperties constructor.

        font_style (string): The logo character font style. Specifically, the
            value passed as the 'style' parameter in the
            matplotlib.font_manager.FontProperties constructor. From matplotlib
            documentation:
                "style: Either 'normal', 'italic' or 'oblique'."

        font_properties (matplotlib.font_manager.FontProperties): The logo
            character front properties; overrides other font_xxx parameters.

        stack_order (string): Order in which to stack characters at the same
            position. Either 'big_on_top', 'small_on_top', or 'fixed'.

        use_transparency (boolean): Option to shade characters according to
            absolute height.

        max_alpha_val (float >= 0.0): Absolute matrix element value
            corresponding to opacity of 1.0. If None is passed, this is set to
            the largest absolute matrix element.

        below_shade (float): Amount to darken any characters drawn below the
            baseline. E.g. a value of 0.8 will cause RBG values to be reduced
            to 80% of their initial value. Restricted to interval [0, 1].

        below_alpha: (float): Amount to reduce the opacity of characters drawn
            below the baseline. E.g., a value of 0.8 will cause alpha values to
            be reduced to 80% of their initial value.

        below_flip (boolean): Whether to flip upside down any characters that
            are drawn below the baseline.

        baseline_width (float): Width of the logo baseline. Restricted to >= 0.

        xlim (tuple): Overrides the default value of (-0.5, L-0.5).

        xticks (list): Overrides automatic setting of range(L).

        xticklabels (list): Overrides default value given by the matrix
            positions.

        xlabel (string): Overrides value determined by "logo_style".

        ylim (tuple): Overrides automatic determination by matplotlib.

        yticks (list): Overrides automatic determination by matplotlib.

        yticklabels (list): Overrides automatic determination by matplotlib.

        ylabel (string): Overrides value determined by "logo_type".
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
        if ylabel is None:
            ylabel = 'counts'
    elif logo_type == 'probability':
        if ylim is None:
            ylim = [0, 1]
        if ylabel is None:
            ylabel = 'probability'
    elif logo_type == 'information':
        if ylim is None and (background is None):
            ylim = [0, np.log2(matrix.shape[1])]
        if ylabel is None:
            ylabel = 'information\n(%s)' % information_units
    elif logo_type == 'enrichment':
        if ylabel is None:
            if enrichment_logbase == 2:
                ylabel = '$\log_2$ enrichment'
            elif enrichment_logbase == 10:
                ylabel = '$\log_{10}$ enrichment'
            elif enrichment_logbase == np.e:
                ylabel = '$\ln$ enrichment'
            else:
                assert False, 'Error: invalid choice of enrichment_logbase=%f'\
                              % enrichment_logbase
            if enrichment_centering:
                ylabel = 'centered\n' + ylabel
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
        if xticks is None:
            xticks = list(matrix.index)
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
        if xticks is None:
            xticks = list(matrix.index)
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
        if show_vlines is None:
            show_vlines = True

    # Set xticks
    if xticks is not None:
        xticks = xticks
    elif xtick_spacing is not None:
        xticks = [pos for pos in positions if \
                  (pos - xtick_anchor) % xtick_spacing == 0.0]
    else:
        xticks = positions

    # Set tick labels and label rotation
    if xticklabels is None:
        xticklabels = xticks
    if yticklabels is None:
        yticklabels = yticks
    if xtick_rotation is None:
        xtick_rotation = 0
    if ytick_rotation is None:
        ytick_rotation = 0

    if title is None:
        title = ''

    # Set axes style
    axes_style = {
        'baseline_width': baseline_width,
        'vline_width': vline_width,
        'vline_color': to_rgba(vline_color),
        'show_vlines': show_vlines,
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
        'ytick_length': ytick_length,
        'ytick_rotation': ytick_rotation
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

    # Set RC parameters
    for key, value in rcparams.items():
        mpl.rcParams[key] = value

    # If user specifies a figure size, make figure and axis,
    # draw logo, then return all three
    if figsize is not None:

        fig, ax = plt.subplots(figsize=figsize)
        logo.draw(ax)
        if save_to_file:
            fig.savefig(save_to_file)
        plt.draw()

        return logo, ax, fig

    # If draw_now, get current axes, draw logo, and return both
    elif draw_now:
        ax = plt.gca()
        logo.draw(ax)

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

        # Set limits
        ax.set_xlim(self.axes_style['xlim'])
        ax.set_ylim(self.axes_style['ylim'])

        # Draw x-axis annotation
        if self.axes_style['xticks'] is not None:
            ax.set_xticks(self.axes_style['xticks'])

        if self.axes_style['xticklabels'] is not None:
            ax.set_xticklabels(self.axes_style['xticklabels'],
                               rotation=self.axes_style['xtick_rotation'])

        if self.axes_style['xlabel'] is not None:
            ax.set_xlabel(self.axes_style['xlabel'])

        if self.axes_style['yticks'] is not None:
            ax.set_yticks(self.axes_style['yticks'])

        if self.axes_style['yticklabels'] is not None:
            ax.set_yticklabels(self.axes_style['yticklabels'],
                               rotation=self.axes_style['ytick_rotation'])

        if self.axes_style['ylabel'] is not None:
            ax.set_ylabel(self.axes_style['ylabel'])

        if self.axes_style['title'] is not None:
            ax.set_title(self.axes_style['title'])

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


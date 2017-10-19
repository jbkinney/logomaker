from __future__ import division
import numpy as np
import pandas as pd
import ast
import inspect
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

# Valid values for matrix_type and logo_type
LOGOMAKER_TYPES = [None, 'counts', 'probability', 'enrichment', 'energy',
                   'information']

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


def actual_kwargs():
    """
    Decorator that provides the wrapped function with an attribute 'actual_kwargs'
    containing just those keyword arguments actually passed in to the function.
    found here:
    https://stackoverflow.com/questions/1408818/getting-the-the-keyword-arguments-actually-passed-to-a-python-method
    """
    def decorator(function):
        def inner(*args, **kwargs):

            # Record names of all function arguments
            inner.arg_names = inspect.getargspec(function)[0]

            # Record names of keywords passed by user
            inner.user_kwargs = kwargs

            return function(*args, **kwargs)
        return inner
    return decorator

@actual_kwargs()
def make_logo(matrix,
              parameters_file=None,
              matrix_type=None,
              logo_type=None,
              background=None,
              pseudocount=1.0,
              energy_gamma=1.0,
              energy_units='$k_B T$',
              enrichment_logbase=2,
              information_units='bits',
              colors='black',
              characters=None,
              alpha=1.,
              edgecolors='none',
              edgewidth=0.,
              boxcolors='white',
              boxalpha=0.,
              highlight_sequence=None,
              highlight_colors=None,
              highlight_alpha=None,
              highlight_edgecolors=None,
              highlight_edgewidth=None,
              highlight_boxcolors=None,
              highlight_boxalpha=None,
              hpad=0.,
              vpad=0.,
              axes_style='classic',
              font_family=None,
              font_weight=None,
              font_file=None,
              font_style=None,
              font_properties=None,
              stack_order='big_on_top',
              use_transparency=False,
              max_alpha_val=None,
              below_shade=1.,
              below_alpha=1.,
              below_flip=True,
              baseline_width=.5,
              xlim=None,
              xticks=None,
              xticklabels=None,
              xlabel='position',
              ylim=None,
              yticks=None,
              yticklabels=None,
              ylabel=None):
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
            'probability', 'enrichment', 'energy', 'information', or
            None. Defaults to the value of matrix.logomaker_type if that
            exists and is valid, then to None.

        logo_type (str): Type of logo to display. Value can be 'counts',
            'probability', 'enrichment', 'energy', 'information', or
            None. Defaults to matrix_type if provided, then to value of
            matrix.logomaker_type if that exists and is valid, then to None.

        background: [WRITE]

        pseudocount (float): For converting a counts matrix to a probability
            matrix. Must be >= 0. [IMPLEMENT]

        energy_gamma (float): For conversion from log enrichment to energy.
            [IMPLEMENT]

        energy_units (str): Units to display in ylabel of energy logo.
            [IMPLEMENT]

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

        ylabel (string): Overrides value determined by "logo_style".

    """

    # Read in args from parameters_file
    file_kwargs = load_parameters(parameters_file)

    # Get list of user-specified arguments
    user_kwargs = make_logo.user_kwargs

    # Report on user-specified and file-specified variables
    reports = []
    for arg_name in make_logo.arg_names:
        # Report argumet if set by user
        if arg_name in user_kwargs:
            arg_value = user_kwargs[arg_name]
            reports.append('[user] %s : %s' %(arg_name, repr(arg_value)))

        # Otherwise, report and set argument specified in file
        elif arg_name in file_kwargs:
            arg_value = file_kwargs[arg_name]

            # WARNING: input to exec needs to be safe!
            exec_str = '%s = %s' % (arg_name, repr(arg_value))
            exec(exec_str)
            reports.append('[file] %s : %s' % (arg_name, repr(arg_value)))

        # Otherwise, report argument specified by default
        # else:
        #     reports.append('[default] %s : %s' % \
        #                    (arg_name, repr(eval(arg_name))))

    # Print report contents
    reports.sort()
    print '### Parameters ###'
    for report in reports:
        print report

    # Validate matrix
    matrix = data.validate_mat(matrix)

    # Check logo_type
    if not (logo_type in LOGOMAKER_TYPES):
        print 'Warning: invalid logo_type = %s. Using None.' % repr(logo_type)
        logo_type = None

    # Set matrix_type if it is None and the user didn't specify it to be so
    if (matrix_type is None):

        # If user specified matrix_type = None
        if ('matrix_type' in make_logo.user_kwargs) and not (logo_type is None):
            logo_type = None
            print 'Warning: matrix_type is None, so setting logo_type=None.'

        # If matrix specifies a matrix type
        elif ('logomaker_type' in matrix.__dict__):
            if matrix.logomaker_type in LOGOMAKER_TYPES:
                matrix_type = matrix.logomaker_type
            else:
                print 'Warning: invalid matrix.logomaker_type = %s.' % \
                      repr(matrix.logomaker_type) +' Using matrix_type = None.'

    # Get background matrix
    bg_mat = data.set_bg_mat(background, matrix)

    if logo_type == 'counts':
        # Transform input matrix to freq_mat
        matrix = data.transform_mat(matrix,
                                    from_type=matrix_type,
                                    to_type='counts',
                                    background=bg_mat)
        ymax = matrix.values.sum(axis=1).max()

        # Change default plot settings
        if ylim is None:
            ylim = [0, ymax]
        if ylabel is None:
            ylabel = 'counts'

    elif logo_type == 'probability':
        # Transform input matrix to freq_mat
        matrix = data.transform_mat(matrix,
                                    from_type=matrix_type,
                                    to_type='probability',
                                    background=bg_mat)

        # Change default plot settings
        if ylim is None:
            ylim = [0, 1]
        if ylabel is None:
            ylabel = 'probability'

    elif logo_type == 'information':
        # Transform input matrix to info_mat
        matrix = data.transform_mat(matrix,
                                    from_type=matrix_type,
                                    to_type='information',
                                    background=bg_mat)

        # Change default plot settings
        if ylim is None and (background is None):
            ylim = [0, np.log2(matrix.shape[1])]
        if ylabel is None:
            ylabel = 'information\n(bits)'

    elif logo_type == 'enrichment':
        # Transform input matrix to weight_mat
        matrix = data.transform_mat(matrix,
                                    from_type=matrix_type,
                                    to_type='enrichment',
                                    background=bg_mat)

        # Change default plot settings
        if ylabel is None:
            ylabel = '$\log_2$\nenrichment'

    elif logo_type == 'energy':
        # Transform input matrix to weight_mat
        matrix = data.transform_mat(matrix,
                                    from_type=matrix_type,
                                    to_type='energy',
                                    background=background)

        # Change default plot settings
        if ylabel is None:
            ylabel = '- energy\n($k_B T$)'

    elif logo_type is None:
        matrix = data.validate_mat(matrix)
        ylabel = ''

    else:
        assert False, 'Error! logo_type %s is invalid' % logo_type

    # Record kwargs for Logo constructor
    kwargs_for_logo = {}
    for arg_name in make_logo.arg_names:
        kwargs_for_logo[arg_name] = eval(arg_name)

    # Create Logo instance
    logo = Logo(**kwargs_for_logo)
    logo.logo_type = logo_type

    # Return Logo instance to user
    return logo


# Logo base class
class Logo:
    def __init__(self, **kwargs):

        # Record user font input
        self.in_font_file = kwargs['font_file']
        self.in_font_style = kwargs['font_style']
        self.in_font_weight = kwargs['font_weight']
        self.in_font_family = kwargs['font_family']
        self.in_font_properties = kwargs['font_properties']

        # If user supplies a FontProperties object, validate it
        if self.in_font_properties is not None:
            assert isinstance(self.in_font_properties, FontProperties),\
                'Error: font_properties is not an instance of FontProperties.'
            self.font_properties = self.in_font_properties.copy()

        # Otherwise, create a FontProperties object based on user's input
        else:
            self.font_properties = FontProperties(family=self.in_font_family,
                                                  weight=self.in_font_weight,
                                                  fname=self.in_font_file,
                                                  style=self.in_font_style)
        # Set data
        self.in_df = kwargs['matrix'].copy()

        # Characters:
        # Restrict to provided characters if string or list of characters
        # Transform to provided characters if dictionary
        self.in_characters = kwargs['characters']
        if self.in_characters is None:
            self.df = self.in_df.copy()
        elif isinstance(self.in_characters, dict):
            self.df = self.in_df.rename(columns=self.in_characters)
        elif isinstance(self.in_characters, (str, unicode, list, np.array)):
            characters = list(self.in_characters)
            self.df = self.in_df[characters]
        else:
            assert False, 'Error: cant interpret characters %s.' % \
                          repr(self.in_characters)

        self.poss = self.df.index.copy()
        self.chars = np.array([str(c) for c in self.df.columns])
        self.L = len(self.poss)

        # Set normal character format
        self.facecolors = kwargs['colors']
        self.edgecolors = kwargs['edgecolors']
        self.edgewidth = kwargs['edgewidth']
        self.alpha = float(kwargs['alpha'])
        self.boxcolors = kwargs['boxcolors']
        self.boxalpha = float(kwargs['boxalpha'])
        self.hpad = kwargs['hpad']
        self.vpad = kwargs['vpad']

        # Set normal character color dicts
        self.facecolors_dict = \
                color.get_color_dict(color_scheme=self.facecolors,
                                     chars=self.chars,
                                     alpha=self.alpha)
        self.edgecolors_dict = \
                color.get_color_dict(color_scheme=self.edgecolors,
                                     chars=self.chars,
                                     alpha=self.alpha)
        self.boxcolors_dict = \
                color.get_color_dict(color_scheme=self.boxcolors,
                                     chars=self.chars,
                                     alpha=self.boxalpha)

        # Set higlighted character format
        self.highlight_facecolors = kwargs['highlight_colors'] \
            if kwargs['highlight_colors'] is not None \
            else kwargs['colors']
        self.highlight_edgecolors = kwargs['highlight_edgecolors'] \
            if kwargs['highlight_edgecolors'] is not None  \
            else kwargs['edgecolors']
        self.highlight_edgewidth = kwargs['highlight_edgewidth'] \
            if kwargs['highlight_edgewidth'] is not None \
            else kwargs['edgewidth']
        self.highlight_alpha = float(kwargs['highlight_alpha']) \
            if kwargs['highlight_alpha'] is not None \
            else kwargs['alpha']
        self.highlight_boxcolors = kwargs['highlight_boxcolors'] \
            if kwargs['highlight_boxcolors'] is not None \
            else kwargs['boxcolors']
        self.highlight_boxalpha = float(kwargs['highlight_boxalpha']) \
            if kwargs['highlight_boxalpha'] is not None \
            else kwargs['boxalpha']

        # Set highlight character color dicts
        self.highlight_facecolors_dict = \
                color.get_color_dict(color_scheme=self.highlight_facecolors,
                                     chars=self.chars,
                                     alpha=self.highlight_alpha)
        self.highlight_edgecolors_dict = \
                color.get_color_dict(color_scheme=self.highlight_edgecolors,
                                     chars=self.chars,
                                     alpha=self.highlight_alpha)
        self.highlight_boxcolors_dict = \
                color.get_color_dict(color_scheme=self.highlight_boxcolors,
                                     chars=self.chars,
                                     alpha=self.highlight_alpha)

        # Set wild type sequence
        self.highlight_sequence = kwargs['highlight_sequence']
        if self.highlight_sequence  is not None:
            assert isinstance(self.highlight_sequence , basestring)
            assert len(self.highlight_sequence ) == len(self.poss)
            assert set(list(str(self.highlight_sequence))) == set(self.chars),\
                'Error: highlight_sequence %s contains invalid characters'\
                %self.highlight_sequence
            self.use_highlight = True
        else:
            self.use_highlight = False

        # Set other character styling
        self.logo_style = kwargs['axes_style']
        self.stack_order = kwargs['stack_order']
        self.use_transparency = kwargs['use_transparency']
        self.neg_shade = float(kwargs['below_shade'])
        self.neg_alpha = float(kwargs['below_alpha'])
        self.neg_flip = kwargs['below_flip']
        self.max_alpha_val = kwargs['max_alpha_val']
        self.use_transparency = kwargs['use_transparency']

        # Compute characters and box
        self.compute_characters()

        # Set x axis params
        self.xlim = [self.bbox.xmin, self.bbox.xmax]\
            if kwargs['xlim'] is None else kwargs['xlim']
        self.xticks = range(len(self.poss)) \
            if kwargs['xticks'] is None else kwargs['xticks']
        self.xticklabels = self.poss\
            if kwargs['xticklabels'] is None else kwargs['xticklabels']
        self.xlabel = kwargs['xlabel']

        # Set y axis params
        self.ylim = [self.bbox.ymin, self.bbox.ymax]\
            if kwargs['ylim'] is None else kwargs['ylim']
        self.ylabel = kwargs['ylabel']
        self.yticks = kwargs['yticks']
        self.yticklabels = kwargs['yticklabels']

        # Set other formatting parameters
        self.floor_line_width = kwargs['baseline_width']


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
            elif self.stack_order == 'fixed':
                indices = range(len(vals))
            else:
                assert False, 'Error: unrecognized stack_order value %s.'%\
                              self.stack_order
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
                    facecolor *= np.array([shade, shade, shade, alpha])
                    edgecolor *= np.array([shade, shade, shade, alpha])
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

        if self.yticks is not None:
            ax.set_yticks(self.yticks)
        if self.yticklabels is not None:
            ax.set_yticklabels(self.yticklabels)


def load_parameters(file_name):
    """
    helper function that will parse the upload parameters file
    so that the parameters can be displayed on screen

    Returns parameters specified as strings.

    parameters
    ----------
    :parameter parameters_filename (str): name of file containing parameters

    """
    params = {}

    # Quit if file_name is not specified
    if file_name is None:
        return params

    # try read parameters file
    try:
        # try opening file
        fileObj = open(file_name,'r')

        """ the following snippet tries to read parameters from file into a dict called params """
        try:
            for line in fileObj:
                # removing leading and trailing whitespace
                line = line.strip()
                # ignore comments in the parameters file
                # the use of hash to display comments in the
                # params file is a design choice by the author
                if not line.startswith("#"):
                    # split lines by colon separator
                    key_value = line.split(":")
                    # print lists only of format key-value, i.e. ignore inputs \
                    # with multiple values for the same key:{value_1,value_2}
                    if len(key_value) == 2:
                        params[key_value[0].strip()] = key_value[1].strip()

            """ 
            parse read-in parameters correctly, i.e. change the value of the appropriate 
            key to the appropriate type
            The following needs to be done for all the parameters in the constructor
            """
            for key, value in params.iteritems():
                params[key] = ast.literal_eval(value) # Much safer than eval()

        except (RuntimeError, ValueError) as pe:
            print('some went wrong parsing the parameters file', pe.message)

    except IOError as e:
        print("Something went wrong reading the parameters file: ", e.strerror,
              e.filename)

    # the following will change your dict to a pandas df
    # param_df = pandas.DataFrame(params, index=[0])
    return params
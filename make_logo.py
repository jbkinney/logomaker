from __future__ import division
import numpy as np
import inspect
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, FontManager
import matplotlib as mpl
from validate import validate_parameter, validate_mat
from Logo import Logo
import data
import color

# Create global font manager instance. This takes a second or two
font_manager = FontManager()

def remove_none_from_dict(d):
    """ Removes None values from a dictionary """
    assert isinstance(d, dict), 'Error: d is not a dictionary.'
    return dict([(k, v) for k, v in d.items() if v is not None])

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
              rcparams={},

              # Grid line formatting
              show_gridlines=False,
              gridline_axis=None,
              gridline_width=None,
              gridline_color=None,
              gridline_alpha=None,
              gridline_style=None,

              # Baseline formatting
              show_baseline=True,
              baseline_width=None,
              baseline_color=None,
              baseline_alpha=None,
              baseline_style=None,

              # x-axis formatting
              xlim=None,
              xticks=None,
              xtick_spacing=None,
              xtick_anchor=0,
              xticklabels=None,
              xtick_rotation=None,
              xtick_length=None,
              xtick_format=None,
              xlabel=None,

              # y-axis formatting
              show_binary_yaxis=False,
              ylim=None,
              yticks=None,
              yticklabels=None,
              ytick_rotation=None,
              ytick_length=None,
              ytick_format=None,
              ylabel=None,

              # Other axis formatting
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
              ):
    """
    Description:

        Generate a logo based on user-specified parameters

    Returns:

        logo (logomaker.Logo): Logo object.
            logo.ax: Axes logo is drawn on. Set if figsize or draw_now=True
                is passed.
            logo.fig: Figure logo is drawn on. Set if figsize is set.

    Arguments:

        #######################################################################
        ### Matrix transformation

        matrix (pd.DataFrame): Data matrix used to make the logo. Row names are
            the positions, column names are the characters, and values are
            floats (or ints). If the matrix_type is set to 'counts',
            'probability', or 'information', all elements of matrix must be
            floats >= 0. In what follows, L refers to the number of matrix rows
            and C refers to the number of matrix columns.

        matrix_type (str in set, None): Type of data passed in matrix. If str,
            value must be in {'counts','probability', 'enrichment',
            'information'}. If None, a generic logo is created. Default None.

        logo_type (str in set, None): Type of logo to display. If str, value
            must be in {'counts','probability', 'enrichment', 'information'}.
            If None, defaults to matrix_type. Default None.

        background (array, dict, pd.DataFrame, None): Only used when creating
            logos of logo_type='enrichment' from matrices of matrix_type=
            'counts' or matrix_type='probability'. Specifies the background
            probability of each character at each position. Different value
            types are interpreted in different ways:
            - None: Each character in each column of matrix is assumed to
                occur with equal background probability
            - array: Must contain floats >= 0 and be of length C. If so, these
                float values are interpreted as the background probabilities of
                the characters corresponding to each column of matrix.
                E.g., for a GC content of 60% and matrix columns ['A','C','G',
                'T'], one can pass,
                    background=[0.2, 0.3, 0.3, 0.2]
            - dict: All characters specified by matrix must be keys of this
                dictionary, and the corresponding values must be floats >= 0.
                If so, the value is interpreted as the relative background
                probability of the character corresponding to each key. E.g.,
                for a GC content of 60% one can pass
                    background = { 'A':0.2, 'C':0.3, 'G':0.3, 'T':0.2}
            - pd.DataFrame, 1 row: Columns must list the same chaarcters as
                the columns of matrix, and values must be floats >= 0. If so,
                each float is interpreted as the relative background
                probability of the character corresponding to each column.
                E.g., for a GC content of 60% one can pass a DataFrame that
                that looks like:
                        'A'     'C'     'G'     'T'
                 0      0.2     0.3     0.3     0.2
            - pd.DataFrame, L rows: Columns must list the same characters as
                the columns of matrix, and values must be floats >= 0. If so,
                the float in each row and column is interpreted as the relative
                background probability of the corresponding character at that
                corresponding position. This option is particularly useful
                when

        pseudocount (float >= 0): For converting a counts matrix to a
            probability matrix. Default 1.

        enrichment_logbase (float in set): Logarithm to use when computing
            enrichment. Value must be in {2, 10, np.e}. Default 2.

        enrichment_centering (bool): If True, log enrichment values at each
            position are shifted so that their mean value is 0. Default True.

        information_units (str in set): Units to use when computing information
            logos. Values must be in {'bits', 'nats'}. Default 'bits'.

        counts_threshold (float >= 0): The total number of sequences in an
            alignment that must have a non-deletion character for that position
            to be included in the resulting counts matrix and derived matrices.
            Default 0.

        #######################################################################
        ### Immediate drawing

        figsize ([float >= 0, float >=0], None): Size of figure in inches. If
            not None, a new figure object and axes object will be created and
            the Logo object will be drawn. The .fig and .ax attributes of the
            returned Logo object will store these figure and axes objects.
            Default None

        draw_now (bool): If True, the logo object will be drawn on plt.gca(),
            which will be stored in the .ax attribute of the returned Logo
            object. Default False

        save_to_file (str, None): If string, specifies the name of file that
            logo is saved to. File type is determined automatically from the
            extension of the file name. If None, no file is stored. Default
            None.

        #######################################################################
        ### Position choice

        position_range ([int >= 0, int >= 0], None): Rows of matrix to use when
            drawing logo. If None, all rows are used. Default None.

        shift_first_position_to (float): Position value to be assigned to the
            first row of matrix that is actually used. Default 1.

        #######################################################################
        ### Position choice

        sequence_type (str in set, None): Specifies the set of characters to
            used in the sequence logo. Non-sequene characters will be ignored.
            If str, must be in {'DNA', 'dna', 'RNA', 'rna', 'PROTEIN', or
            'protein'}. If None, this option is ignored. Default None.

        characters (str, None): Specifies the set of characters to be used in
            the sequence logo. If str, any of the non-whitespace characters
            listed in the string can be used. If None, this option is ignored.
            Overridden by sequence_type. Default None.

        ignore_characters (str, None): Specifies the set of characters to not
            be used in the sequence logo. If str, all of the characters listed
            will be ignored. If None, all characters will be used. Overridden
            by characters. Default '-.'.

        #######################################################################
        ### Logo character styling

        colors (color scheme):  Face color of logo characters. Default 'gray'.
            Here and in what follows a variable of type 'color' can
            take a variety of value types.
            - (str) A LogoMaker color scheme in which the color is determined
                by the specific character being drawn. Options are,
                + For DNA/RNA: 'classic', 'grays', 'base_paring'.
                + For protein: 'hydrophobicity', 'chemistry', 'charge'.
            - (str) A built-in matplotlib color name  such as 'k' or 'tomato'
            - (str) A built-in matplotlib colormap name such as  'viridis' or
                'Purples'. In this case, the color within the colormap will
                depend on the character being drawn.
            - (list) An RGB color (3 floats in interval [0,1]) or RGBA color
                (4 floats in interval [0,1]).
            - (dict) A dictionary mapping of characters to colors, in which
                case the color will depend on the character being drawn. E.g.,
                {'A': 'green',
                 'C': [ 0.,  0.,  1.],
                 'G': 'y',
                 'T': [ 1.,  0.,  0.,  0.5]}

        alpha (float in [0,1]): Opacity of logo character face color. Here and
            in what follows, if the corresponding color is specified by an
            RGBA array, this alpha value will be multiplied by the 'A' value
            that array to yield the final alpha value. Default 1.

        edgecolors (color scheme): Edge color of the logo characters. Default
            'black'.

        edgealpha (float in [0,1]): Opacity of character edge color. Default 1.

        edgewidth (float >= 0): Width of character edges. Default 0.

        boxcolors (color scheme): Face color of the box containing each logo
            character. Default 'white'.

        boxalpha (float in [0,1]): Opacity box face color. Default 0.

        boxedgecolors (color scheme): Edge color of the box containing each
            logo character. Default 'black'.

        boxedgealpha (float in [0,1]): Opacity of box edge coclor. Default 0.

        boxedgewidth (float >= 0): Width of box edges. Default 1.

        #######################################################################
        ### Highlighted logo character styling

        highlight_sequence (str, None): Sequence to highlight within the logo.
            If str, sequence must be the same length as the logo. The valid
            character at each position in this sequence will then be drawn
            using style parameters highlight_colors, highlight_alpha, etc.,
            instead of colors, alpha, etc. If None, no characters are
            highlighted. Default None.

            In what follows, each highlight_* parameter behaves behaves as the
            * parameter listed above, but is used only for highlighted
            characters. Each parameter can also be None, and in fact is None
            by default. Each highlight_* parameter, when None is passed,
            defaults to the value provided in the corresponding * parameter.

        highlight_colors (color scheme, None): See colors.

        highlight_alpha (float in [0,1], None): See alpha.

        highlight_edgecolors (color scheme, None): See edgecolors.

        highlight_edgealpha (float in [0,1], None): See edgealpha.

        highlight_edgewidth (float >= 0, None): See edge_width.

        highlight_boxcolors (color scheme, None: See boxcolors.

        highlight_boxalpha (float in [0,1], None): See boxalpha.

        highlight_boxedgecolors (color scheme, None): See boxedgecolors.

        highlight_boxedgealpha (float in [0,1], None): see boxedgealpha.

        highlight_boxedgewidth (float >= 0, None): See boxedgewidth.

        #######################################################################
        ### Logo character font

        # Logo characters #

        font_properties (FontProperties, None): The logo character font
            properties. If FontProperties, overrides all of the font_*
            parameters below. If None, is ignored. Default None.

        font_name (str, None): Name of the font used for logo characters. This
            name corresponds to a specific font_file, which is determined
            using the FontManager.findfont() method. If str, overrides the
            font_file, font_family, and maybe other parameters below. If None,
            is ignored. Default None

        font_file (str, None): The local file specifying the logo character
            font. Specifically, the value passed as the 'fname' parameter in
            the FontProperties constructor. Default None.

        font_family (str, list, None): The logo character font family name.
            Specifically, the value passed as the 'family' parameter when
            calling the FontProperties constructor. From matplotlib
            documentation:
                "family: A list of font names in decreasing order of priority.
                The items may include a generic font family name, either
                'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.
                In that case, the actual font to be used will be looked up from
                the associated rcParam in matplotlibrc."
            Default None.

        font_weight (str, float, None): The logo character font weight.
            Specifically, the value passed as the 'weight' parameter in the
            FontProperties constructor. From matplotlib documentation:
                "weight: A numeric value in the range 0-1000 or one of
                'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
                'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
                'extra bold', 'black'."
            Default None.

        font_style (str, None): The logo character font style. Specifically,
            the value passed as the 'style' parameter in the FontProperties
            constructor. From matplotlib documentation:
                "style: Either 'normal', 'italic' or 'oblique'."
            Default None.

        #######################################################################
        ### Character placement

        stack_order (str in set): Order in which to stack characters at the
            same position. Either 'big_on_top', 'small_on_top', or 'fixed'.
            Default 'big_on_top'.

        use_transparency (boolean): Option to render characters with an opacity
            proportional to character height. Default False.

        max_alpha_val (float >= 0.0, None): Absolute matrix element value
            corresponding to opacity of 1. If None, this is set to the largest
            absolute-value matrix element. Default None.

        below_shade (float in [0,1]): Amount by which to darken logo characters
            drawn below the baseline. E.g., a value of 0.8 will cause RBG
            values to be reduced to 80% of their initial value. Default 1.

        below_alpha (float in [0,1]): Amount by which to reduce the opacity of
            logo characters drawn below the baseline. E.g., a value of 0.8 will
            cause opacity values to be reduced to 80% of their initial value.
            Default 1.

        below_flip (bool): Whether to flip upside down all characters drawn
            below the baseline.

        hpad (float >= 0): Relative amount of empty space to include on the
            sides of each character within its bounding box. E.g., a value of
            0.2 will result in empty space totaling 20% of that character's
            width (10% on each side). Default 0.

        vpad (float >= 0): Relative amount of empty space to include above and
            below each character within its bounding box. E.g., a value of
            0.2 will result in empty space totaling 20% of that character's
            height (10% above and 10% below). Default 0.

        width (float in [0,1]): Width of each character bounding box in
            position units. Default 1.

        uniform_stretch (bool): If True, each logo character will be
            horizontally stretched by the same factor. This factor will be
            determined by the natrually widest character that appears in the
            logo, and the stretch factor will be computed to accomodate the
            value of hpad specified by the user. Most characters will thus be
            drawn with additional varying blank space on either side. Default
            False.

        max_stretched_character (str length 1, None): If a character is
            specified, logo characters will be horizontally stretched in the
            following manner: The specified character and characters naturally
            wider than it will be stretched according to width and hpad, while
            narrower characters will be stretched the same amount as the
            specified character. Specifying a value like 'A' allows wide
            characters like 'W' and 'A' to be stretched to fill the avaliable
            width while preventing naturally narrow characters like 'I' from
            being stretched an absurdly large amount.

        #######################################################################
        ### Axes formatting

        axes_type (str in set): Axes logo style. Value must be in {'classic',
            'rails', 'naked', 'everything'}. Default 'classic'.

        style_sheet (str, None): Matplotlib style sheet to use for default axis
            styling. This value is passed to plt.style.use(). Available style
            sheets are listed in plt.style.available. Examples include
                - 'classic': Standard Matplotlib style prior to v2.0
                - 'dark_background': White text/lines on black background
                - 'ggplot': Mimics the default style of ggplot in R
                - 'fivethirtyeight': Mimics the style used by
                    fivethirtyeight.com
                - 'seaborn-pastel', 'seaborn-darkgrid', etc: Mimics the style
                    defined in the popular seaborn plotting package.
            Ignored if None is passed. Default None.

        rcparams (dict): Default parameter values to used for matplotlib
            plotting. Warning: using this changes defaults for all subsequent
            plots, not just the current logo. Default {}.

        #######################################################################
        ### Gridline formatting

        show_gridlines (bool): Whether to show gridlines. Note: gridlines are
            plotted below logo characters but above logo bounding boxes.
            Default False.

        gridline_axis (str in set, None): If str, specifies axes on which to
            plot gridlines value in {'x', 'y', 'both'}. Passed as the 'axis'
            argument to ax.grid() if not None. Default None.

        gridline_width (float >= 0, None): If not None, specifies the width of
            plotted gridlines. Is passed as the 'linewidth' argument to
            ax.grid() if not None. Default None.

        gridline_color (color): If not None, specifies the color of the
            gridlines. Is passed as the 'color' argument to ax.grid() if not
            None. Default None.

        gridline_alpha (float in [0,1], None): If not None, specifies the
            opacity of the gridlines. Is passed as the 'alpha' argument to
            ax.grid() if not None. Default None.

        gridline_style (str, None): If not None, specifies gridline line style.
            Is passed as the 'linestyle' argument to ax.grid if not None.
            Default None.

        #######################################################################
        ### Baseline formatting

        show_baseline (bool): Whether to show the baseline at y=0. Note: the
            baseline is plotted above logo characters. Default True.

        baseline_width (float >= 0, None): If not None, specifies the width of
            plotted baseline. Is passed as the 'linewidth' argument to
            ax.axhline() if not None. Default None.

        baseline_color (color): If not None, specifies the color of the
            baseline. Is passed as the 'color' argument to ax.axhline() if not
            None. Default None.

        baseline_alpha (float in [0,1], None): If not None, specifies the
            opacity of baseline. Is passed as the 'alpha' argument to
            ax.axhline() if not None. Default None.

        baseline_style (str, None): If not None, specifies baseline line style.
            Is passed as the 'linestyle' argument to ax.axhline() if not None.
            Default None.

        #######################################################################
        ### x-axis formatting

        xlim ([float, float], None): x-axis limits. Determined automatically if
            None. Default None.

        xticks (array, None): Location of tick marks on x-axis. Overrides
            xtick_spacing and xtick_anchor if not None. Default None.

        xtick_spacing (float, None): Spacing between x-axis tick marks.
            Tickmarks drawn at xtick_anchor + z*xtick_spacing for all integers
            if value is not None. Overrides axes_type if not None. Default
            None.

        xtick_anchor (float): Determines positioning of xticks as described
            above. Default 0.

        xticklabels (array, None): Values to display below x-axis tickmarks.
            Labels are determined automatically if None. Default None.

        xtick_rotation (float, None): Angle in degrees at which to draw x-axis
            tick labels. Angle is determined automatically if None. Default
            None.

        xtick_length (float >= 0, None): Length of x-axis tick marks. Length is
            determined automatically if None. Default None.

        xtick_format (str, None): Formatting string used for making x-axis
            labels. Overridden by xticklabels. Ignored if None. Default None.

        xlabel (str, None): Text to display below the x-axis. Determined
            automatically if None. Default None.

        #######################################################################
        ### y-axis formatting

        show_binary_yaxis (bool): If True, y-axis is labeled with '+' and '-'.
            in place of numerically labeled ticks. Default False.

        ylim ([float, float], None): y-axis limits. Determined automatically if
            None. Default None.

        yticks (array, None): Location of tick marks on y-axis. Overrides
            ytick_spacing and ytick_anchor if not None. Default None.

        yticklabels (array, None): Values to display below y-axis tickmarks.
            Labels are determined automatically if None. Default None.

        ytick_rotation (float, None): Angle in degrees at which to draw y-axis
            tick labels. Angle is determined automatically if None. Default
            None.

        ytick_length (float >= 0, None): Length of x-axis tick marks. Length is
            determined automatically if None. Default None.

        ytick_format (str, None): Formatting string used for making x-axis
            labels. Overridden by xticklabels. Ignored if None. Default None.

        ylabel (str, None): Text to display below the x-axis. Determined
            automatically if None. Default None.

        #######################################################################
        ### Other axes formatting

        title (str, None): Title of plot if not None. Default None.

        left_spine (bool, None): Whether to show the left axis spine. If None,
            spine choice is set by axes_type. Default None.

        right_spine (bool, None): Whether to show the right axis spine. If
            None, spine choice is set by axes_type. Default None.

        top_spine (bool, None): Whether to show the top axis spine. If None,
            spine choice is set by axes_type. Default None.

        bottom_spine (bool, None): Whether to show the bottom axis spine. If
            None, spine choice is set by axes_type. Default None.

        use_tightlayout (bool): Whether to call plt.tight_layout() after
            logo is plotted. If called, this will reformat the plot to try and
            ensure that all plotted elements are visible. Note: this will
            reformat the entire figure, not just the logo axes.

            #######################################################################
            ### Default axes font

        axes_fontname (FontProperties, None): See font_name. Default to use for
            axes labels, axes tick labels, and title. Ignored if None. Default
            None.

        axes_fontfile (str, None): See font_file. Default to use for axes
            labels, axes tick labels, and title. Ignored if None. Default None.

        axes_fontfamily (str, None): See font_family. Default to use for
            axes labels, axes tick labels, and title. Ignored if None. Default
            None.

        axes_fontweight (str, float, None): See font_weight. Default to use for
            axes labels, axes tick labels, and title. Ignored if None. Default
            None.

        axes_fontstyle (str, None): See font_style. Default to use for axes
            labels, axes tick labels, and title. Ignored if None. Default None.

        axes_fontsize (str, float, None): Font size to be used for axes
            labels, axes tick labels, and title. Passed as 'size' parameter to
            the FontProperties constructor. From matplotlib documentation:
                "size: Either an relative value of 'xx-small', 'x-small',
                'small', 'medium', 'large', 'x-large', 'xx-large' or an
                 absolute font size, e.g., 12"
            Ignored if value is None. Default None.

        #######################################################################
        ### Tick label font

        tick_fontname (FontProperties, None): Overrides axes_fontname for tick
            label styling if value is not None. Default None.

        tick_fontfile (str, None): Overrides axes_fontfile for tick label
            styling if value is not None. Default None.

        tick_fontfamily (str, None): Overrides axes_fontfamily for tick label
            styling if value is not None. Default None.

        tick_fontweight (str, float, None): Overrides axes_fontweight for tick
            label styling if value is not None. Default None.

        tick_fontstyle (str, None): Overrides axes_fontstyle for tick label
            styling if value is not None. Default None.

        tick_fontsize (str, float, None): Overrides axes_fontsize for tick
            label styling if value is not None. Default None.

        #######################################################################
        ### Axis label font

        label_fontname (FontProperties, None): Overrides axes_fontname for axis
            label styling if value is not None. Default None.

        label_fontfile (str, None): Overrides axes_fontfile for axis label
            styling if value is not None. Default None.

        label_fontfamily (str, None): Overrides axes_fontfamily for axis label
            styling if value is not None. Default None.

        label_fontweight (str, float, None): Overrides axes_fontweight for axis
            label styling if value is not None. Default None.

        label_fontstyle (str, None): Overrides axes_fontstyle for axis label
            styling if value is not None. Default None.

        label_fontsize (str, float, None): Overrides axes_fontsize for axis
            label styling if value is not None. Default None.

        #######################################################################
        ### Title font

        title_fontname (FontProperties, None): Overrides axes_fontname for
            title styling if value is not None. Default None.

        title_fontfile (str, None): Overrides axes_fontfile for title styling
            if value is not None. Default None.

        title_fontfamily (str, None): Overrides axes_fontfamily for title
            styling if value is not None. Default None.

        title_fontweight (str, float, None): Overrides axes_fontweight for
            title label styling if value is not None. Default None.

        title_fontstyle (str, None): Overrides axes_fontstyle for title
            styling if value is not None. Default None.

        title_fontsize (str, float, None): Overrides axes_fontsize for title
            styling if value is not None. Default None.

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
    positions = np.arange(shift_first_position_to, shift_first_position_to + L)
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

    # Baseline styling
    baseline_dict = {
        'color': baseline_color,
        'alpha': baseline_alpha,
        'linewidth': baseline_width,
        'linestyle': baseline_style,
    }
    baseline_dict = remove_none_from_dict(baseline_dict)

    # Set axes_style dictionary
    axes_style = {
        'show_baseline': show_baseline,
        'baseline_dict': baseline_dict,
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

        logo.ax = ax
        logo.fig = fig

    # If draw_now, get current axes, draw logo, and return both
    elif draw_now:
        ax = plt.gca()
        logo.draw(ax)

        if use_tightlayout:
            plt.tight_layout()
            plt.draw()

        logo.ax = ax
        logo.fig = None

    # Otherwise, just return logo to user
    else:
        logo.ax = None
        logo.fig = None

    # Return logo to user
    return logo


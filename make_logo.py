from __future__ import division
import numpy as np
import inspect
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from validate import validate_parameter, validate_dataframe, \
    params_that_specify_colorschemes
from data import load_alignment, load_matrix, iupac_to_probability_mat, \
    counts_mat_to_probability_mat
from Logo import Logo
import data
import color
from load_meme import load_meme
from documentation_parser import document_function
import os

import pdb
import sys

default_fig_width = 8
default_fig_height_per_line = 2

def remove_none_from_dict(d):
    """ Removes None values from a dictionary """
    assert isinstance(d, dict), 'Error: d is not a dictionary.'
    return dict([(k, v) for k, v in d.items() if v is not None])

def make_logo(dataframe=None,

              # Matrix transformation (make_logo only)
              matrix_type=None,
              logo_type=None,
              background=None,
              pseudocount=1.0,
              enrichment_logbase=2,
              center_columns=False,
              information_units='bits',
              counts_threshold=None,
              negate_matrix=False,

              # Immediate drawing (make_logo only)
              figsize=None,
              draw_now=True,
              save_to_file=None,
              dpi=300,

              # Character choice
              sequence_type=None,
              characters=None,
              ignore_characters='.-',

              # Dictionary containing styling options for characters
              character_style_dict = None,

              # Highlighted character formatting
              highlight_style_dict = None,

              #highlight_sequence=None,
              highlight_bgconsensus=False,
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
              highlight_zorder=None,

              # Full height formatting
              fullheight=None,
              fullheight_colors=None,
              fullheight_alpha=None,
              fullheight_edgecolors=None,
              fullheight_edgewidth=None,
              fullheight_edgealpha=None,
              fullheight_boxcolors=None,
              fullheight_boxalpha=None,
              fullheight_boxedgecolors=None,
              fullheight_boxedgewidth=None,
              fullheight_boxedgealpha=None,
              fullheight_zorder=None,
              fullheight_vsep=None,
              fullheight_width=None,

              # Character font
              font_properties=None,
              font_file=None,
              font_family=('Arial Rounded MT Bold', 'Arial', 'sans'),
              font_weight='bold',
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
              width=1.,
              vsep=0.,
              uniform_stretch=False,
              max_stretched_character='A',
              remove_flattened_characters=True,

              # Special axes formatting
              axes_type='classic',
              style_sheet=None,
              #rcparams={},
              rcparams=None,

              # Scalebar styling
              show_scalebar=None,
              scalebar_length=None,
              scalebar_linewidth=None,
              scalebar_color=None,
              scalebar_text=None,
              scalebar_x=None,
              scalebar_ymin=None,
              scalebar_texthalignment=None,
              scalebar_textvalignment=None,
              scalebar_textrotation=None,

              # Grid line formatting
              show_gridlines=None,
              gridline_axis=None,
              gridline_width=None,
              gridline_color=None,
              gridline_alpha=None,
              gridline_style=None,

              # Baseline formatting
              show_baseline=None,
              baseline_width=None,
              baseline_color=None,
              baseline_alpha=None,
              baseline_style=None,
              baseline_zorder=None,

              # vlines formatting
              vline_positions=(),
              vline_width=None,
              vline_color=None,
              vline_alpha=None,
              vline_style=None,
              vline_zorder=None,

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
              show_binary_yaxis=None,
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
              use_tightlayout=True,

              # Default axes font
              axes_fontfile=None,
              axes_fontfamily='sans',
              axes_fontweight=None,
              axes_fontstyle=None,
              axes_fontsize=10,

              # tick font
              tick_fontfile=None,
              tick_fontfamily=None,
              tick_fontweight=None,
              tick_fontstyle=None,
              tick_fontsize=None,

              # label font
              label_fontfile=None,
              label_fontfamily=None,
              label_fontweight=None,
              label_fontstyle=None,
              label_fontsize=None,

              # title font
              title_fontfile=None,
              title_fontfamily=None,
              title_fontweight=None,
              title_fontstyle=None,
              title_fontsize=None,
              ):

    """

    Generate a logo based on user-specified parameters

    Parameters
    ----------

    background_mattype: (str)
     Type of background matrix loaded from background_matcsvfile. \n
     Must  be 'counts' or 'probability'. Default None.

    character_boxedgecolors: (color scheme
        Edge color of the box containing each logo character. Default None.

    xlim: ([float, float], None):
        x-axis limits. Determined automatically if None. Default None.

    baseline_style: (str, list, None)
        If not None, specifies baseline line style. Is passed as \n
        the 'linestyle' argument to ax.axhline() if not None. \n
        Default None.

    axes_fontfamily: (str, list, None)
        See font_family. Default to use for axes labels, axes \n
        tick labels, and title. Ignored if None. Default None.

    matrix_type: (str in set, None)
        Type of data passed in matrix. If str, value must be in \n
        {'counts','probability', 'enrichment', 'information'}. \n
        If None, this parameter is either set automatically \n
        (e.g. by specifying iupac_string) or a generic logo is  \n
        created. Default None.

    highlight_boxedgealpha: (float in [0,1], None)
        see boxedgealpha. Default None.

    baseline_width: (float >= 0, None)
        If not None, specifies the width of plotted baseline. Is \n
        passed as the 'linewidth' argument to ax.axhline() if not \n
        None. Default None.

    character_colors: (color scheme)
        Face color of logo characters. Default 'gray'. Here and in \n
        what follows a variable of type 'color' can take a variety of value types. \n
         - (str) A LogoMaker color scheme in which the color is determined
             by the specific character being drawn. Options are, \n
             + For DNA/RNA: 'classic', 'grays', 'base_paring'.
             + For protein: 'hydrophobicity', 'chemistry', 'charge'.
         - (str) A built-in matplotlib color name  such as 'k' or 'tomato'
         - (str) A built-in matplotlib colormap name such as  'viridis' or
             'Purples'. In this case, the color within the colormap will
             depend \n
              on the character being drawn.
         - (list) An RGB color (3 floats in interval [0,1]) or RGBA color
             (4 floats in interval [0,1]).
         - (dict) A dictionary mapping of characters to colors, in which
             case the color will depend \n
             on the character being drawn. E.g., \n
             {'A': 'green','C': [ 0.,  0.,  1.], 'G': 'y', 'T': [ 1.,  0.,  0.,  0.5]} \n
         Default None.

    below_flip: (bool)
        Whether to flip upside down all characters drawn below the baseline. \n
        Default None.

    scalebar_length: (float, None)
        If not None, specifies the  length of the scalebar in y-axis units. \n
        Default None.

    background_ctcol: (literal, None)
        If csv_file or background_csvfile is specified, this specifies \n
        the name of the column that lists the number of counts for each \n
        sequence. Defaults to ct_col if None is specified. Default None.

    show_position_zero: (bool)
        If False, no xticklabel 0 will be shown. Instead, \n
        xticklabels for positions >= 0 are all increased by 1; the x-axis will \n
        thus be labeled using ...-3, -2, -1, 1, 2, 3,... . This is because some \n
        biological numbering convensions avoid the use of zero. This option only \n
        affects xticklabels, not the positions listed in the matrix dataframe. \n
        Default None.

    fullheight_edgecolors: (color scheme)
        See edgecolors. Default None.

    ylabel: (str, None)
        Text to display below the x-axis. Determined automatically if None. \n
        Default None.

    vline_width: (float >= 0, None)
        If not None, specifies the width of plotted vlines. Is passed \n
        as the 'linewidth' argument to ax.axhline() if not None. \n
        Default None.

    character_boxcolors: (color scheme)
        Face color of the box containing each logo character. Default None.

    position_range: ([int >= 0, int >= 0], None)
        Rows of matrix to use when drawing logo. If None, all rows are used. \n
        Default None.

    below_shade: (float in [0,1])
        Amount by which to darken logo characters drawn below the baseline. \n
        E.g., a value of 0.8 will cause RBG values to be reduced to 80% \n
        of their initial value. Default None.

    axes_type: (str in set)
        Axes logo style. Value must be in \n
        {'classic', 'rails', 'naked', 'everything, 'vlines'}. Default None.

    tick_fontfamily: (str, list, None)
        Overrides axes_fontfamily for tick label styling if value is \n
        not None. Default None.

    vline_color: (color, None)
        If not None, specifies the color of the vlines. Is passed as \n
        the 'color' argument to ax.axhline() if not None. Default None.

    font_weight: (str, float, None)
        The logo character font weight. Specifically, the value passed \n
        as the 'weight' parameter in the FontProperties constructor. \n
        From matplotlib documentation: \n
             "weight: A numeric value in the range 0-1000 or one of \n
             'ultralight', 'light', 'normal', 'regular', 'book', 'medium', \n
             'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy', \n
             'extra bold', 'black'." Default None. \n

    show_baseline (bool): 
        Whether to show the baseline at y=0. Note: the \n
         baseline is plotted above logo characters. Default None. \n

    xticklabels (array, None): 
        Values to display below x-axis tickmarks. Labels are \n
        determined automatically if None. Default None. \n

    background (array, dict, pd.DataFrame, None): 
        Only used when creating logos of logo_type='enrichment' from \n 
        matrices of  matrix_type = 'counts' or matrix_type='probability'. \n 
        Specifies the background probability of each character at each position. \n
        Different value types are interpreted in different ways: \n
         - None: Each character in each column of matrix is assumed to \n
           occur with equal background probability \n
         - array: Must contain floats >= 0 and be of length C. If so, these \n
           float values are interpreted as the background probabilities of \n
           the characters corresponding to each column of matrix. \n
           E.g., for a GC content of 60% and matrix columns ['A','C','G','T'], \n
           one can pass, background=[0.2, 0.3, 0.3, 0.2] \n

         - dict: All characters specified by matrix must be keys of this \n
           dictionary, and the corresponding values must be floats >= 0. \n
           If so, the value is interpreted as the relative background \n
           probability of the character corresponding to each key. E.g., \n
           for a GC content of 60% one can pass background = \n
           { 'A':0.2, 'C':0.3, 'G':0.3, 'T':0.2} \n

         - pd.DataFrame, 1 row: Columns must list the same chaarcters as \n
           the columns of matrix, and values must be floats >= 0. If so, \n
           each float is interpreted as the relative background \n
           probability of the character corresponding to each column. \n
           E.g., for a GC content of 60% one can pass a DataFrame that \n
           that looks like: 'A'     'C'     'G'     'T' \n
           0      0.2     0.3     0.3     0.2 \n

         - pd.DataFrame, L rows: Columns must list the same characters as \n
           the columns of matrix, and values must be floats >= 0. If so, \n
           the float in each row and column is interpreted as the relative \n
           background probability of the corresponding character at that \n
           corresponding position. This option is particularly useful \n
           when Default None. \n

    xtick_spacing (float, None):
        Spacing between x-axis tick marks. Tickmarks drawn at \n
        xtick_anchor + z*xtick_spacing for all integers if value \n
        is not None. Overrides axes_type if not None Default None. \n

    xtick_length (float >= 0, None):
        Length of x-axis tick marks. Length is determined automatically \n
        if None. Default None. \n

    draw_now (bool):
        If True, a new figure will be created of size figsize \n
        and the logo will be drawn. If not, the logo will not be \n
        drawn. Default None. \n

    ignore_characters (str, None):
        Specifies the set of characters to not be used in the \n
        sequence logo. If str, all of the characters listed \n
        will be ignored. If None, all characters will be used. \n
        Overridden by characters. Default None. \n

    use_tightlayout (bool):
        Whether to call plt.tight_layout() after logo is plotted. \n
        If called, this will reformat the plot to try and \n
        ensure that all plotted elements are visible. Note: \n
        this will reformat the entire figure, not just the \n
        logo axes. Default None. \n

    tick_fontweight (str, float, None):
        Overrides axes_fontweight for tick label styling if \n
        value is not None. Default None. \n

    title_fontstyle (str, None):
        Overrides axes_fontstyle for title styling if value is not \n
        None. Default None. \n

    fullheight_width (float >= 0):
        Overrides width when applied to fullheight positions \n
        Default None. \n

    logo_type (str in set, None):
        Type of logo to display. If str, value must be in \n
        {'counts','probability', 'enrichment', 'information'}. \n
        If None, defaults to matrix_type. Default None. \n

    counts_threshold (float >= 0, None):
        The total number of sequences in an alignment that \n
        must have a non-deletion character for that position \n
        to be included in the resulting counts matrix and \n
        derived matrices. If this is not None, positions with \n
        counts falling below this threshold will be removed from \n
        the matrix. Regardless of whether any positions are \n
        removed, matrix positions will be renumbered starting \n
        from zero. Default None. \n

    dpi (int):
        The resolution at which to save the logo image if writing \n
        to a non-vector format. Default None. \n

    baseline_zorder (float, None):
        Specifies how in-front baseline is drawn relative to logo \n
        characters and other plotted objects. Larger values \n
        correspond to more in-frontness. Default None. \n

    use_transparency (boolean):
        Option to render characters with an opacity proportional \n
        to character height. Default None. \n

    character_boxalpha (float in [0,1]):
        Opacity box face color. Default None. \n

    scalebar_textrotation (float, None):
        If not None, specifies the rotation angle of the scalebar \n
        label text. Default None. \n

    highlight_zorder (float):
        See zorder. Default None. \n

    csv_delimiter (bool, None):
        The delimiter character to use when parsing CSV files. \n
        If None, the default value for pandas.read_csv(delimneter=) \n
        is used. Default None. \n

    background_seqcsvfile (str, None):
        Name of file containing background sequences and (optionally) \n
        background sequence counts in column-separated value format. \n
        Defaults to sequences_csvfile if None is passed. Default None. \n

    meme_motifnum (int > 0, None):
        The number of the motif to be loaded from the MEME file used \n
        when meme_file is specified. The first motif is specified by \n
        a value of 1. Ignored if None is passed. Default None. \n

    sequences_csvfile (str, None):
        Name of file containing sequence and (optionally) sequence counts \n
        in column-separated value format. Ignored if None is passed. \n
        Default None.

    matrix (pd.DataFrame, None):
        Data matrix used to make the logo. Row names are \n
        the positions, column names are the characters, and values \n
        are floats (or ints). If the matrix_type is set to 'counts', \n
        'probability', or 'information', all elements of matrix must \n
        be floats >= 0. In what follows, L refers to the number of \n
        matrix rows and C refers to the number of matrix columns. \n
        Default None. \n

    scalebar_ymin (float, None):
        If not None, specifies the y-cooridnate of the bottom of the \n
        scalebar. Default None. \n

    fullheight_boxedgecolors (color scheme):
        See boxedgecolors. Default None. \n

    ytick_format (str, None):
        Formatting string used for making x-axis labels. Overridden \n
        by xticklabels. Ignored if None. Default None. \n

    center_columns (bool):
        If True, values at each position are shifted so that their \n
        mean value is 0. Default None. \n

    sequence_type (str in set, None):
        Specifies the set of characters to used in the sequence logo. \n
        Non-sequene characters will be ignored. If str, must be in \n
        {'DNA', 'dna', 'RNA', 'rna', 'PROTEIN', or 'protein'}. \n
        If None, this option is ignored. Default None. \n

    ct_col (literal, None):
        If csv_file is specified, this specifies the name \n
        of the column that lists the number of counts for each sequence. \n
        If  csv_file is specified and this is not, a count of 1 will be \n
        assumed for each listed sequence. Ignored if None is passed. \n
        Default None.\n

    uniform_stretch (bool):
        If True, each logo character will be horizontally stretched \n
        by the same factor. This factor will be determined by the \n
        natrually widest character that appears in the logo, and the \n
        stretch factor will be computed to accomodate the \n
        value of hpad specified by the user. Most characters will \n
        thus be drawn with additional varying blank space on either \n
        side. Default None.
        
    highlight_edgewidth (float >= 0, None): 
        See edge_width. Default None. \n
        
    fullheight_colors (color scheme): 
        See colors. Default None. \n
        
    font_properties (FontProperties, None): 
        The logo character font properties. If FontProperties, overrides \n 
        all of the font_* parameters below. If None, is ignored. \n
         Default None. \n
         
    csv_index_col (literal, None): 
        Name of column to user for matrix positions. If None, the default \n 
        value for pandas.read_csv(index_col=) is used. \n
         Default None. \n
         
    information_units (str in set): 
        Units to use when computing information logos. Values must \n 
        be in {'bits', 'nats'}. Default None. \n
        

    baseline_alpha (float in [0,1], None):
        If not None, specifies the opacity of baseline. Is passed \n
        as the 'alpha' argument to ax.axhline() if not None. \n
         Default None. \n

    font_family (str, list, None):
        The logo character font family name. Specifically, the value \n
        passed as the 'family' parameter when calling the FontProperties \n
        constructor. From matplotlib documentation: \n

             "family: A list of font names in decreasing order of priority. \n
             The items may include a generic font family name, either \n
             'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'. \n
             In that case, the actual font to be used will be looked up from \n
             the associated rcParam in matplotlibrc." \n
        Default None. \n

    highlight_sequence (str, None):
        Sequence to highlight within the logo. If str, sequence must \n
        be the same length as the logo. The valid character at each \n
        position in this sequence will then be drawn using style parameters \n
        highlight_colors, highlight_alpha, etc., instead of colors, alpha, \n
        etc. If None, no characters are highlighted.  In what follows, each \n
        highlight_* parameter behaves behaves as the * parameter listed \n
        above, but is used only for highlighted characters. Each parameter \n
        can also be None, and in fact is None by default. Each highlight_* \n
        parameter, when None is passed, defaults to the value provided in the \n
        corresponding * parameter. Default None. \n

    scalebar_texthalignment (str, None):
        If not None, specifies the horizonotal alignment of the scalebar. \n
        See mpl.text.Text() documentation for options. Default None. \n

    character_boxedgealpha (float in [0,1]):
        Opacity of box edge color. Default None. \n

    highlight_edgealpha (float in [0,1], None):
        See edgealpha. Default None. \n

    tick_fontfile (str, None):
        Overrides axes_fontfile for tick label styling if value is not None. \n
        Default None. \n

    character_edgecolors (color scheme):
        Edge color of the logo characters. Default None. \n

    stack_order (str in set):
        Order in which to stack characters at the same position. \n
        Can be 'big_on_top', 'small_on_top', 'fixed_going_up', 'fixed_going_down'. \n
        Default None. \n

    xtick_anchor (float):
        Determines positioning of xticks as described above. \n
         Default None. \n

    fullheight_edgealpha (float in [0,1]):
        See edgealpha. Default None. \n

    meme_motifname (str, None):
        The name of the motif to be loaded from the MEME \n
        file used when meme_file is specified. Ignored if None \n
        is passed. Default None. \n

    yticklabels (array, None):
        Values to display below y-axis tickmarks. Labels are \n
        determined automatically if None. Default None. \n

    show_gridlines (bool, None):
        Whether to show gridlines. Note: gridlines are plotted below \n
        logo characters but above logo bounding boxes. Default None. \n

    highlight_bgconsensus (bool):
        If True, the consensus sequence of the background probability \n
        matrix is used as highlight_sequence. Default None. \n

    seq_col (literal, None):
        If csv_file is specified, this specifies the name of the column \n
        that lists sequences. Must be passed if csv_file is specified. \n
         Default None. \n

    show_binary_yaxis (bool, None):
        If True, y-axis is labeled with '+' and  '-'. in place \n
        of numerically labeled ticks. Overrides ylim, yticks, and \n
        yticklabels. Default None. \n

    vline_zorder (float, None):
        Specifies how in-front vlines are drawn relative to logo \n
        characters and other plotted objects. Larger values correspond \n
        to more in-frontness. Default None. \n

    title_fontfamily (str, list, None):
        Overrides axes_fontfamily for title styling if value is not None. \n
        Default None. \n

    fullheight_alpha (float in [0,1]):
        See alpha.  Default None.

    fullheight_boxalpha (float in [0,1]):
        See boxalpha. Default None.

    csv_header (literal, None):
        Row to use as header when parsing CSV file. If None, \n
        the default value for pandas.read_csv(header=) is used. \n
        Default None. \n

    meme_file (str, None):
        MEME file from which to load a motif and (optionally) \n
        a background model as well. The resulting matrix_type is 'probability'. \n
        The first motif within the MEME file is loaded if neither meme_motifname \n
        nor meme_motifnum is specified.  A background model will be loaded \n
        automatically if any is detected. Ignored if None is passed. \n
        Default None. \n

    axes_fontstyle (str, None):
        See font_style. Default to use for axes  labels, axes \n
        tick labels, and title. Ignored if None. Default None. \n

    character_edgewidth (float >= 0):
        Width of character edges. Default None.

    style_sheet (str, None):
        Matplotlib style sheet to use for default axis styling. \n
        This value is passed to plt.style.use(). Available style \n
        sheets are listed in plt.style.available. Examples include \n
             - 'classic': Standard Matplotlib style prior to v2.0 \n
             - 'dark_background': White text/lines on black background \n
             - 'ggplot': Mimics the default style of ggplot in R \n
             - 'fivethirtyeight': Mimics the style used by \n
               fivethirtyeight.com \n
             - 'seaborn-pastel', 'seaborn-darkgrid', etc: Mimics the style \n
                defined in the popular seaborn plotting package. \n
        Ignored if None is passed.  Default None. \n

    font_file (str, None):
        The local file specifying the logo character font. Specifically, \n
        the value passed as the 'fname' parameter in the FontProperties \n
        constructor. Default None. \n

    hpad (float >= 0):
        Relative amount of empty space to include on the \n
        sides of each character within its bounding box. E.g., a value of \n
        0.2 will result in empty space totaling 20% of that character's \n
        width (10% on each side). Default None. \n

    tick_fontsize (str, float, None):
        Overrides axes_fontsize for tick label styling if value is not None. \n
         Default None. \n

    axes_fontweight (str, float, None):
        See font_weight. Default to use for axes labels, axes \n
        tick labels, and title. Ignored if None. Default None. \n

    matrix_csvfile (str, None):
        Name of file containing matrix values in CSV format. \n
        Ignored if None is passed. Default None. \n

    pseudocount (float >= 0):
        For converting a counts matrix to a probability matrix. \n
        Default None. \n

    title_fontsize (str, float, None):
        Overrides axes_fontsize for title styling if value is not \n
        None. Default None. \n

    highlight_boxcolors (color scheme, None):
        See boxcolors. Default None. \n

    csv_usecols (literal, None):
        Columns to use when parsing CSV file. If None, the default \n
        value for pandas.read_csv(use_cols=) is used Default None. \n

    background_csvkwargs (dict):
        If csv_file or background_csvfile is specified, this dictionary \n
        contains the keyword arguments to be passed to pandas.read_csv. \n
        For example, if the csv file uses whitespace to separate columns, \n
        one might pass csv_kwargs={'delim_whitespace':True}. Defaults to \n
        csv_kwargs if None is passed. Default None. \n

    character_boxedgewidth (float >= 0):
        Width of box edges. Default None. \n

    label_fontweight (str, float, None):
        Overrides axes_fontweight for axis label styling if value is \n
        not None. Default None. \n

    fullheight_boxedgealpha (float in [0,1]):
    see boxedgealpha. Default None. \n

    ytick_length (float >= 0, None):
        Length of x-axis tick marks. Length is determined automatically \n
        if None. Default None. \n

    fullheight_boxcolors (color scheme):
        See boxcolors. Default None. \n

    xticks (array, None):
        Location of tick marks on x-axis. Overrides xtick_spacing and \n
        xtick_anchor if not None. Default None. \n

    width (float in [0,1]):
        Width of each character bounding box in position units. \n
        Default None. \n

    gridline_alpha (float in [0,1], None):
        If not None, specifies the opacity of the gridlines. Is \n
        passed as the 'alpha' argument to ax.grid() if not None. \n
        Default None. \n

    vsep (float >= 0):
        Vertical separation between stacked characters and their \n
        bounding boxes, expressed as a fraction of the span of the y-axis. \n
        Half of this separation is subtracted from both the top and \n
        bottoms of each character's bounding box. Characters with  \n
        height smaller than this separation are rendered as empty \n
        bounding boxes. Default None. \n

    vline_alpha (float in [0,1], None):
        If not None, specifies the opacity of the vlines. \n
        Is passed as the 'alpha' argument to ax.axhline() if not \n
        None. Default None. \n

    character_edgealpha (float in [0,1]):
        Opacity of character edge color. Default None. \n

    axes_fontfile (str, None):
        See font_file. Default to use for axes labels, axes \n
        tick labels, and title. Ignored if None. Default None. \n

    title_fontfile (str, None):
        Overrides axes_fontfile for title styling if value is \n
        not None. Default None. \n

    highlight_colors (color scheme, None):
        See colors. Default None.

    tick_fontstyle (str, None):
        Overrides axes_fontstyle for tick label styling if value is \n
        not None. Default None. \n

    axes_fontsize (str, float, None):
        Font size to be used for axes labels, axes tick labels, and \n
        title. Passed as 'size' parameter to the FontProperties constructor. \n
        From matplotlib documentation: "size: Either an relative value of \n
        'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', \n
        'xx-large' or an absolute font size, e.g., 12"  Ignored if value is None. \n
         Default None. \n

    background_matcsvfile (str, None):
        Name of file containing background matrix in CSV format. This matrix \n
        must be of type 'counts' or 'probability', which is specified by the \n
        background_mattype parameter. Default None. \n

    max_positions_per_line (int > 0):
        The maximum number of positions per line. If  the number of \n
        positions within a matrix exceeds this number, the logo will \n
        be split over multiple lines. If figsize is None, a figure whose \n
        height is proportional to the number of lines will be created. \n
        Default None. \n

    fullheight_zorder (float):
        See zorder. Default None.

    scalebar_text (str, None):
        If not None, specifies the text displayed next to the scalebar. \n
         Default None. \n

    highlight_alpha (float in [0,1], None):
        See alpha. Default None. \n

    scalebar_textvalignment (str, None):
        If not None, specifies the vertical alignment of the scalebar. \n
        See mpl.text.Text() documentation for options. Default None. \n

    scalebar_x (float, None):
        If not None, specifies the x-position of the scalebar \n
        (which is vertical). Default None. \n

    highlight_edgecolors (color scheme, None):
        See edgecolors. Default None. \n

    fullheight_vsep (float >= 0):
        Overrides vsep when applied to fullheight positions. Default None. \n

    highlight_boxedgewidth (float >= 0, None):
        See boxedgewidth. Default None. \n

    gridline_axis (str in set, None):
        If str, specifies axes on which to plot gridlines value in \n
        {'x', 'y', 'both'}. Passed as the 'axis' argument to ax.grid() \n
        if not None. Default None. \n

    xlabel (str, None):
        Text to display below the x-axis. Determined automatically if None. \n
        Default None. \n

    label_fontfamily (str, list, None):
        Overrides axes_fontfamily for axis label styling if value is not \n
        None. Default None.

    left_spine (bool, None):
        Whether to show the left axis spine. If None, spine choice is set \n
        by axes_type. Default None. \n

    iupac_string (str, None):
        A string specifying a motif in IUPAC format. The resulting \n
        matrix_type is 'probability' and the resulting characters \n
        are 'ACGT'. The parameter sequence_type can be used in \n
        conjunction with this parameter to transform the characters to \n
        'dna', 'RNA', or 'rna.' Ignored if None is passed. \n
        Default None. \n

    character_alpha (float in [0,1]):
        Opacity of logo character face color. Here  and in what follows, \n
        if the corresponding color is specified by an RGBA array, this \n
        alpha value will be multiplied by the 'A' value that array to \n
        yield the final alpha value. Default None. \n

    highlight_boxedgecolors (color scheme, None):
        See boxedgecolors. Default None. \n

    fasta_file (str, None):
        FASTA file from which sequence counts are to be loaded. The \n
        resulting matrix_type is 'counts'. Default None. \n

    rcparams (dict):
        Default parameter values to used for matplotlib plotting. \n
        Warning: using this changes defaults for all subsequent plots, \n
        not just the current logo. Default None. \n

    gridline_style (str, list, None):
        If not None, specifies gridline line style. Is passed \n
        as the 'linestyle' argument to ax.grid if not None. \n
        Default None. \n

    remove_flattened_characters (bool):
        If vsep is set, some characters will be shrunk to size \n
        zero. This parameter determines whether or not these flattended \n
        characters will contribute to the height of a stack. Default None. \n

    character_zorder (float >= 0):
        Determines whether characters and their bounding boxes appear in \n
        front of or behind other features of the logo, such as \n
        gridlines. The higher zorder is, the more in-front these characters \n
        will be. Characters are always drawn immediately in front of their \n
        bounding boxes. Default None. \n

    xtick_format (str, None):
        Formatting string used for making x-axis labels. Overridden \n
        by xticklabels. Ignored if None. Default None. \n

    gridline_color (color, None):
        If not None, specifies the color of the gridlines. Is passed \n
        as the 'color' argument to ax.grid() if not None. Default None. \n

    baseline_color (color, None):
        If not None, specifies the color of the baseline. Is passed \n
        as the 'color' argument to ax.axhline() if not None. Default None. \n

    enrichment_logbase (float in set):
        Logarithm to use when computing enrichment. Value must be \n
        in {2, 10, np.e}. Default None. \n

    label_fontfile (str, None):
        Overrides axes_fontfile for axis label styling if value is not
        None. Default None. \n

    ylim ([float, float], None):
        y-axis limits. Determined automatically if None. Default None. \n
        show_scalebar (bool, None): If not None, specifies wheterh to
        show a scalebar. Default None. \n

    font_style (str, None):
        The logo character font style. Specifically, the value passed \n
        as the 'style' parameter in the FontProperties constructor. \n
        From matplotlib documentation: \n
        "style: Either 'normal', 'italic' or 'oblique'." \n
         Default None. \n

    vpad (float >= 0):
        Relative amount of empty space to include above and below each \n
        character within its bounding box. E.g., a value of \n
        0.2 will result in empty space totaling 20% of that character's \n
        height (10% above and 10% below). Default None. \n

    max_alpha_val (float >= 0.0, None):
        Absolute matrix element value corresponding to opacity of 1. \n
        If None, this is set to the largest absolute-value matrix element. \n
        Default None.

    xtick_rotation (float, None):
        Angle in degrees at which to draw x-axis tick labels. Angle \n
        is determined automatically if None. Default None. \n

    below_alpha (float in [0,1]):
        Amount by which to reduce the opacity of logo characters drawn \n
        below the baseline. E.g., a value of 0.8 will cause opacity \n
        values to be reduced to 80% of their initial value. Default None. \n

    shift_first_position_to (float):
        Position value to be assigned to the first row of matrix that is \n
        actually used. Default None. \n

    csv_kwargs (dict):
        If csv_file is specified, this dictionary contains the \n
        keyword arguments to be passed to pandas.read_csv. For \n
        example, if the csv file uses whitespace to separate columns, \n
        one might pass csv_kwargs={'delim_whitespace':True}. \n
        Default None.

    figsize ([float >= 0, float >=0], None): 
        Size of figure in inches. If not None, a default size \n
        for the figure will be used. If draw_now is True, a new \n 
        figure will be created of this size, and will be stored \n 
        in logo.fig. The axes on which the logo is drawn will be \n 
        saved to logo.ax. Default None. \n
        
    top_spine (bool, None): 
        Whether to show the top axis spine. If None, spine choice \n 
        is set by axes_type. Default None. \n
        
    highlight_boxalpha (float in [0,1], None): 
        See boxalpha. Default None. \n
        
    max_stretched_character (str length 1, None): 
        If a character is specified, logo characters will be horizontally \n 
        stretched in the following manner: The specified character and \n 
        characters naturally wider than it will be stretched according \n 
        to width and hpad, while narrower characters will be stretched  \n
        the same amount as the specified character. Specifying a value like \n
        'A' allows wide characters like 'W' and 'A' to be stretched to fill \n 
        the avaliable width while preventing naturally narrow characters \n 
        like 'I' from being stretched an absurdly large amount. \n
        Default None. \n
        
    fullheight_boxedgewidth (float >= 0): 
        See boxedgewidth. Default None. \n
        
    yticks (array, None): 
        Location of tick marks on y-axis. Overrides ytick_spacing and \n 
        ytick_anchor if not None. Default None. \n
        
    ytick_rotation (float, None): 
        Angle in degrees at which to draw y-axis tick labels. \n 
        Angle is determined automatically if None. Default None. \n
        
    label_fontsize (str, float, None): 
        Overrides axes_fontsize for axis label styling if value is \n 
        not None. Default None. \n
        
    vline_style (str, list, None): 
        If not None, specifies vline line style. Is passed as the \n 
        'linestyle' argument to ax.axhline() if not None. Default None. \n
        
    gridline_width (float >= 0, None): 
        If not None, specifies the width of plotted gridlines. \n 
        Is passed as the 'linewidth' argument to ax.grid() if not None. \n
        Default None. \n
        
    characters (str, None): 
        Specifies the set of characters to be used in the sequence logo. \n 
        If str, any of the non-whitespace characters listed in the \n 
        string can be used. If None, this option is ignored. Overridden \n 
        by sequence_type. Default None. \n
        
    scalebar_color (float, None): 
        If not None, specifies the color of the scalebar itself \n 
        (not the text). Default None. \n
        
    csv_delim_whitespace (bool, None): 
        Whether to interpret whitespace as the csv delimiter. \n
        If None, the default value for pandas.read_csv(delim_whitespace=) \n 
        is used. Default None. \n
        
    label_fontstyle (str, None): 
        Overrides axes_fontstyle for axis label styling if value is not 
        None. Default None. \n
        
    bottom_spine (bool, None): 
        Whether to show the bottom axis spine. If None, spine \n 
        choice is set by axes_type. Default None. \n
        
    title_fontweight (str, float, None): 
        Overrides axes_fontweight for title label styling if value \n 
        is not None. Default None.
        
    save_to_file (str, None): 
        If string, specifies the name of file that logo is saved to. \n 
        File type is determined automatically from the extension of the \n 
        file name. If None, no file is stored. Default None. \n

    scalebar_linewidth (float, None): 
        If not None, specifies the width of the scalebar in points. \n
        Default None. \n
        
    background_seqcol (literal, None): 
        If csv_file or background_csvfile is specified, this specifies \n 
        the name of the column that lists background sequences. Defaults to \n 
        seq_col. Default None. \n
        
    fullheight (dict, list, None): 
        Either a dictionary or a list. If a dictionary listing positions as \n 
        keys and characters as values, such as {3:'G', 4:'U'} those characters and \n 
        bounding box will be drawn at those positions using the full height of \n
        the y-axis. If a list specifying just positions is passed, only a box \n
        the full height of the y-axis will be drawn at those positions. \n
        In what follows, each fullheight_* parameter behaves behaves as the * \n
        parameter listed above. Unlike highlight_* parameters described immediately \n
        above, however, fullheight_* parameters do not default to the values of the \n
        corresponding * parameter if None is passed. Default None. \n
        
    vline_positions (array): 
        A list of x positions at which to draw vertical lines (henceforth 'vlines'). \n
        These lines are drawn with a different style than gridlines to allow \n
        users to mark specific locations within a logo. Default None. \n
        
    right_spine (bool, None): 
        Whether to show the right axis spine. If None, spine choice is set by \n 
        axes_type. Default None. \n
        
    title (str, None): 
        Title of plot if not None. Default None. \n
        
    negate_matrix (bool): 
        If true, all matrix values are multiplied by -1. \n
        Default None. \n
        
    fullheight_edgewidth (float >= 0): 
        See edge_width. Default None. \n

    Returns
    -------
    
    logo (a logomaker.Logo object). 
        The figure and axes on which the logo is drawn are saved in \n
        logo.fig and logo.ax respectively. (If a single-line logo is \n 
        drawn) If a multi-line logo is drawn: list of logomaker.Logo objects, \n
        one for each line. \n

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
    # matrix

    # Initialize background matrix to none
    bg_mat = None

    # Make sure that only one of the following is specified
    exclusive_list = ['dataframe']
    num_input_sources = sum([eval(x) is not None for x in exclusive_list])
    if num_input_sources != 1:
        assert False, \
            'Error: exactly one of the following must be specified: %s.' %\
            repr(exclusive_list)

    # If matrix is specified
    if dataframe is not None:
        dataframe = validate_dataframe(dataframe)


    else:
        assert False, 'This should never happen.'


    ######################################################################
    # matrix.columns

    # Filter matrix columns based on sequence and character specifications
    dataframe = data.filter_columns(matrix=dataframe,
                                    sequence_type=sequence_type,
                                    characters=characters,
                                    ignore_characters=ignore_characters)
    characters = dataframe.columns

    ######################################################################
    # matrix.index

    # If matrix_type is counts, and counts_theshold is not None,
    # remove positions with too few counts and renumber positions starting
    # at zero
    if matrix_type == 'counts' and counts_threshold is not None:
        position_counts = dataframe.values.sum(axis=1)
        dataframe = dataframe.loc[position_counts >= counts_threshold, :]
        dataframe['pos'] = range(len(dataframe))
        dataframe.set_index('pos', inplace=True, drop=True)


    # Enforce integer positions and set as index
    #dataframe['pos'] = dataframe['pos'].astype(int)
    #dataframe.set_index('pos', inplace=True, drop=True)
    #dataframe = validate_dataframe(dataframe)
    #positions = dataframe.index

    # Shift bg_mat positions too if bg_mat is specified
    #if bg_mat is not None:
        #bg_mat['pos'] = positions
        #bg_mat.set_index('pos', inplace=True, drop=True)
        #bg_mat = validate_dataframe(bg_mat)

    ######################################################################
    # matrix.values

    # Negate matrix values if requested
    if negate_matrix:
        dataframe *= -1.0

    # Set logo_type equal to matrix_type if is currently None
    if logo_type is None:
        logo_type = matrix_type
    logo_type = validate_parameter('logo_type', logo_type, None)

    # Get background matrix, only if it has not yet been set
    if bg_mat is None:
        bg_mat = data.set_bg_mat(background, dataframe)

    # Transform matrix:
    dataframe = data.transform_mat(matrix=dataframe,
                                   from_type=matrix_type,
                                   to_type=logo_type,
                                   pseudocount=pseudocount,
                                   background=bg_mat,
                                   enrichment_logbase=enrichment_logbase,
                                   center_columns=center_columns,
                                   information_units=information_units)

    # Set highlight sequence from background consensus if requested
    # Overrides highlight_sequence
    if highlight_bgconsensus and bg_mat is not None:
        cols = bg_mat.columns
        highlight_style_dict['highlight_sequence'] = ''
        for i, row in bg_mat.iterrows():
            highlight_style_dict['highlight_sequence'] += row.argmax()

    ######################################################################
    # font_properties

    # If font_properties is set directly by user, validate it
    if font_properties is not None:
        assert isinstance(font_properties, FontProperties), \
            'Error: font_properties is not an instance of FontProperties.'
    # Otherwise, create font_properties from other font information
    else:

        # Create properties
        font_properties = FontProperties(family=font_family,
                                         weight=font_weight,
                                         fname=font_file,
                                         style=font_style)

    ######################################################################
    # Set highlight style

    # Set higlighted character format
    highlight_colors = highlight_colors \
        if highlight_colors is not None \
        else character_style_dict['character_colors']
    highlight_alpha = float(highlight_alpha) \
        if highlight_alpha is not None \
        else character_style_dict['character_alpha']
    highlight_edgecolors = highlight_edgecolors \
        if highlight_edgecolors is not None \
        else character_style_dict['character_edgecolors']
    highlight_edgewidth = highlight_edgewidth \
        if highlight_edgewidth is not None \
        else character_style_dict['character_edgewidth']
    highlight_edgealpha = float(highlight_edgealpha) \
        if highlight_edgealpha is not None \
        else character_style_dict['character_edgealpha']
    highlight_boxcolors = highlight_boxcolors \
        if highlight_boxcolors is not None \
        else character_style_dict['character_boxcolors']
    highlight_boxalpha = float(highlight_boxalpha) \
        if highlight_boxalpha is not None \
        else character_style_dict['character_boxalpha']
    highlight_boxedgecolors = highlight_boxedgecolors \
        if highlight_boxedgecolors is not None \
        else character_style_dict['character_boxedgecolors']
    highlight_boxedgewidth = highlight_boxedgewidth \
        if highlight_boxedgewidth is not None \
        else character_style_dict['character_boxedgewidth']
    highlight_boxedgealpha = highlight_boxedgealpha \
        if highlight_boxedgealpha is not None \
        else character_style_dict['character_boxedgealpha']
    highlight_zorder = highlight_zorder \
        if highlight_zorder is not None \
        else character_style_dict['character_zorder']

    ######################################################################
    # Set fullheight style

    # If a list is passed, make characters transparent
    if isinstance(fullheight, np.ndarray):
        # Force characters to be transparent, since these are dummy
        # characters anyway
        fullheight_alpha = 0

        # Have box transparency default to 1, since that is all there
        # is to see
        if fullheight_boxalpha is None:
            fullheight_boxalpha = 1

        # Create dictionary with dummy characters
        keys = list(fullheight)
        vals = ['A']*len(fullheight)
        fullheight = dict(zip(keys, vals))

    # If None, default to empty dictionary
    elif fullheight is None:
        fullheight = {}

    fullheight_characters = set(fullheight.values())

    # Set fullheight character format
    fullheight_colors = fullheight_colors \
        if fullheight_colors is not None \
        else character_style_dict['character_colors']
    fullheight_alpha = float(fullheight_alpha) \
        if fullheight_alpha is not None \
        else character_style_dict['character_alpha']
    fullheight_edgecolors = fullheight_edgecolors \
        if fullheight_edgecolors is not None \
        else character_style_dict['character_edgecolors']
    fullheight_edgewidth = fullheight_edgewidth \
        if fullheight_edgewidth is not None \
        else character_style_dict['character_edgewidth']
    fullheight_edgealpha = float(fullheight_edgealpha) \
        if fullheight_edgealpha is not None \
        else character_style_dict['character_edgealpha']
    fullheight_boxcolors = fullheight_boxcolors \
        if fullheight_boxcolors is not None \
        else character_style_dict['character_boxcolors']
    fullheight_boxalpha = float(fullheight_boxalpha) \
        if fullheight_boxalpha is not None \
        else character_style_dict['character_boxalpha']
    fullheight_boxedgecolors = fullheight_boxedgecolors \
        if fullheight_boxedgecolors is not None \
        else character_style_dict['character_boxedgecolors']
    fullheight_boxedgewidth = fullheight_boxedgewidth \
        if fullheight_boxedgewidth is not None \
        else character_style_dict['character_boxedgewidth']
    fullheight_boxedgealpha = fullheight_boxedgealpha \
        if fullheight_boxedgealpha is not None \
        else character_style_dict['character_boxedgealpha']
    fullheight_zorder = fullheight_zorder \
        if fullheight_zorder is not None \
        else character_style_dict['character_zorder']
    fullheight_vsep = fullheight_vsep \
        if fullheight_vsep is not None \
        else vsep
    fullheight_width = fullheight_width \
        if fullheight_width is not None \
        else width

    ######################################################################
    # Modify colors and alpha values if either are None

    # If a color is not set, set alpha to 0
    # If a color is set and alpha is not, set alpha to 1

    colors_alpha_pairs = [
        ('character_style_dict["character_colors"]', 'character_style_dict["character_alpha"]'),
        ('character_style_dict["character_edgecolors"]', 'character_style_dict["character_edgealpha"]'),
        ('character_style_dict["character_boxcolors"]', 'character_style_dict["character_boxalpha"]'),
        ('character_style_dict["character_boxedgecolors"]', 'character_style_dict["character_boxedgealpha"]'),
        ('highlight_colors', 'highlight_alpha'),
        ('highlight_edgecolors', 'highlight_edgealpha'),
        ('highlight_boxcolors', 'highlight_boxalpha'),
        ('highlight_boxedgecolors', 'highlight_boxedgealpha'),
        ('fullheight_colors', 'fullheight_alpha'),
        ('fullheight_edgecolors', 'fullheight_edgealpha'),
        ('fullheight_boxcolors', 'fullheight_boxalpha'),
        ('fullheight_boxedgecolors', 'fullheight_boxedgealpha')
    ]

    for colors_varname, alpha_varname in colors_alpha_pairs:

        colors = eval(colors_varname)
        alpha = eval(alpha_varname)

        if colors is None:
            exec ('%s = "gray"' % colors_varname)
            exec ('%s = 0.0' % alpha_varname)
        elif alpha is None:
            exec ('%s = 1.0' % alpha_varname)

    ######################################################################
    # Set style dicts

    character_style = {
        'facecolors': color.get_color_dict(color_scheme=character_style_dict['character_colors'],
                                           chars=characters,
                                           alpha=character_style_dict['character_alpha']),
        'edgecolors': color.get_color_dict(color_scheme=character_style_dict['character_edgecolors'],
                                           chars=characters,
                                           alpha=character_style_dict['character_edgealpha']),
        'boxcolors': color.get_color_dict(color_scheme=character_style_dict['character_boxcolors'],
                                          chars=characters,
                                          alpha=character_style_dict['character_boxalpha']),
        'boxedgecolors': color.get_color_dict(
                                        color_scheme=character_style_dict['character_boxedgecolors'],
                                        chars=characters,
                                        alpha=character_style_dict['character_boxedgealpha']),
        'edgewidth': character_style_dict['character_edgewidth'],
        'boxedgewidth': character_style_dict['character_boxedgewidth'],
        'zorder': character_style_dict['character_zorder'],
    }

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
        'zorder': highlight_zorder,
    }


    fullheight_style = {
        'facecolors': color.get_color_dict(color_scheme=fullheight_colors,
                                           chars=fullheight_characters,
                                           alpha=fullheight_alpha),
        'edgecolors': color.get_color_dict(color_scheme=fullheight_edgecolors,
                                           chars=fullheight_characters,
                                           alpha=fullheight_edgealpha),
        'boxcolors': color.get_color_dict(color_scheme=fullheight_boxcolors,
                                          chars=fullheight_characters,
                                          alpha=fullheight_boxalpha),
        'boxedgecolors': color.get_color_dict(
                                        color_scheme=fullheight_boxedgecolors,
                                        chars=fullheight_characters,
                                        alpha=fullheight_boxedgealpha),
        'edgewidth': fullheight_edgewidth,
        'boxedgewidth': fullheight_boxedgewidth,
        'zorder': fullheight_zorder,
        'vsep': fullheight_vsep,
        'width': fullheight_width
    }

    '''
    ######################################################################
    # multi-line logos
    if L > max_positions_per_line:

        # Compute the number of lines needed
        num_lines = int(np.ceil(L / max_positions_per_line))

        # Set figsize
        fig_height = num_lines * default_fig_height_per_line
        if figsize is None:
            figsize = [default_fig_width, fig_height]

        # Pad matrix with zeros
        rows = dataframe.index[0] + \
               np.arange(L, num_lines * max_positions_per_line)
        for r in rows:
            dataframe.loc[r, :] = 0.0

        # If there is a background matrix, pad it with ones:
        if bg_mat is not None:
            for r in rows:
                bg_mat.loc[r, :] = 1. / bg_mat.shape[1]

        # If there is a highlight sequence, pad it too
        if highlight_sequence is not None:
            highlight_sequence = highlight_sequence + \
                                 ' ' * int(
                                     num_lines * max_positions_per_line - L)

        # Get arguments passed by user
        kwargs = dict(zip(names, user_values))

        # If 'random' was picked for any color, choose a specific color_dict
        # and set this across all logo lines
        for var_name in params_that_specify_colorschemes:
            if eval('%s == "random"' % var_name):
                these_chars = characters \
                              if not 'fullheight' in var_name \
                              else fullheight_characters
                kwargs[var_name] = color.get_color_dict(
                                        color_scheme='random',
                                        chars=these_chars,
                                        alpha=1)

        # Set ylim (will not be None)
        if ylim is None:
            values = dataframe.fillna(0).values
            ymax = (values * (values > 0)).sum(axis=1).max()
            ymin = (values * (values < 0)).sum(axis=1).min()
            ylim = [ymin, ymax]

        # Set style sheet:
        if style_sheet is not None:
            if style_sheet == 'default':
                mpl.rcdefaults()
            else:
                plt.style.use(style_sheet)

        # Create figure
        if draw_now:
            fig, axs = plt.subplots(num_lines, 1, figsize=figsize)

        logos = []
        for n in range(num_lines):

            # Section matrix
            start = int(n * max_positions_per_line)
            stop = int((n + 1) * max_positions_per_line)
            n_matrix = dataframe.iloc[start:stop, :]

            # If there is a background matrix, section it
            if bg_mat is not None:
                n_bgmat = bg_mat.iloc[start:stop, :]

            # If there is a highlight sequence, section it
            if highlight_sequence is not None:
                n_highlight_sequence = highlight_sequence[start:stop]
            else:
                n_highlight_sequence = None

            # Adjust kwargs
            n_kwargs = kwargs.copy()

            # Use only matrix and background as input, not files or iupac
            # To do this, first set all input variables to None
            for var_name in exclusive_list:
                n_kwargs[var_name] = None

            # Then pass sectioned matrices to matrics and background.
            #n_kwargs['matrix'] = n_matrix
            n_kwargs['dataframe'] = n_matrix
            n_kwargs['background'] = n_bgmat

            # Preserve matrix and logo type
            n_kwargs['matrix_type'] = logo_type
            n_kwargs['logo_type'] = logo_type

            # Pass sectioned highlight_sequence
            n_kwargs['highlight_sequence'] = n_highlight_sequence

            # Pass shifted shift_first_position_to
            n_kwargs['shift_first_position_to'] = dataframe.index[0] + start

            # Don't draw each individual logo. Wait until all are returned.
            n_kwargs['figsize'] = None
            n_kwargs['draw_now'] = False
            n_kwargs['ylim'] = ylim

            # Adjust annotation
            if n != 0:
                n_kwargs['title'] = ''
            if n != num_lines - 1:
                n_kwargs['xlabel'] = ''

            # Create logo
            logo = make_logo(**n_kwargs)
            if draw_now:
                logo.fig = fig
                logo.ax = axs[n]
                logo.draw(logo.ax)
            else:
                logo.fig = None
                logo.ax = None
            logos.append(logo)

        return logos
    '''

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
        'vsep': vsep,
        'width': width,
        'uniform_stretch': uniform_stretch,
        'max_stretched_character': max_stretched_character,
        'remove_flattened_characters': remove_flattened_characters,
    }

    ######################################################################
    # axes_style

    # Set style sheet:
    if style_sheet is not None:
        plt.style.use(style_sheet)

    # Modify ylim and ylabel according to logo_type
    if logo_type == 'counts':
        ymax = dataframe.values.sum(axis=1).max()
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
            ylim = [0, np.log2(dataframe.shape[1])]
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
                ylabel += '$(\ln)$ '
            else:
                assert False, 'Error: invalid choice of enrichment_logbase=%f'\
                              % enrichment_logbase
    else:
        if ylabel is None:
            ylabel = ''

    # Set ylim (will not be None)
    if ylim is None:
        values = dataframe.fillna(0).values
        ymax = (values * (values > 0)).sum(axis=1).max()
        ymin = (values * (values < 0)).sum(axis=1).min()
        ylim = [ymin, ymax]

    # Set xlim (will not be None)
    if xlim is None:
        xmin = dataframe.index.min() - .5
        xmax = dataframe.index.max() + .5
        xlim = [xmin, xmax]

    # Set xticks
    if xtick_spacing is None and (axes_type in ['classic', 'everything']):
        xtick_spacing = 1
    #if xticks is None and xtick_spacing is not None:
    #    xticks = [pos for pos in positions if
    #              (pos - xtick_anchor) % xtick_spacing == 0.0]

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
        if show_baseline is None:
            show_baseline = True
        if show_gridlines is None:
            show_gridlines = False

    elif axes_type == 'naked':
        if xticks is None:
            xticks = []
        if xtick_length is None:
            xtick_length = 0
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
        if show_baseline is None:
            show_baseline = True
        if show_gridlines is None:
            show_gridlines = False

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
        if show_baseline is None:
            show_baseline = True
        if show_gridlines is None:
            show_gridlines = False

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
        if show_baseline is None:
            show_baseline = True
        if show_gridlines is None:
            show_gridlines = False

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
        if show_gridlines is None:
            show_gridlines = True
        if gridline_axis is None:
            gridline_axis = 'x'
        if gridline_alpha is None:
            gridline_alpha = .5
        if show_baseline is None:
            show_baseline = True

    if axes_type == 'scalebar':
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
        if show_baseline is None:
            show_baseline = True
        if show_gridlines is None:
            show_gridlines = False
        if show_scalebar is None:
            show_scalebar = True

    # If showing binary yaxis, symmetrize ylim and set yticks to +/-
    if show_binary_yaxis:

        # Force set of ylim, yticks, and yticklabels
        y = np.max(np.abs([y for y in ylim]))
        ylim = [-y, y]
        yticks = [.5 * ylim[0], .5 * ylim[1]]
        yticklabels = ['$-$', '$+$']
        if ytick_length is None:
            ytick_length = 0

    # Set label rotation
    if xtick_rotation is None:
        xtick_rotation = 0
    if ytick_rotation is None:
        ytick_rotation = 0

    if title is None:
        title = ''

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

    # Set scalebar defaults
    if show_scalebar is None:
        show_scalebar = False
    if scalebar_text is None:
        scalebar_text = '1 unit'
    if scalebar_color is None:
        scalebar_color = mpl.rcParams['axes.edgecolor']
    if scalebar_linewidth is None:
        scalebar_linewidth = 2
    if scalebar_x is None:
        scalebar_x = xlim[0]-.5
    if scalebar_length is None:
        scalebar_length = 1
    if scalebar_ymin is None:
        scalebar_ymin = 0.5*(ylim[0] + ylim[1]) - .5
    if scalebar_texthalignment is None:
        scalebar_texthalignment = 'right'
    if scalebar_textvalignment is None:
        scalebar_textvalignment = 'center'
    if scalebar_textrotation is None:
        scalebar_textrotation = 90

    # Scalebar styling
    scalebar_linestyle = {
        'linewidth': scalebar_linewidth,
        'color': scalebar_color,
        'xloc': scalebar_x,
        'ymin': scalebar_ymin,
        'ymax': scalebar_ymin+scalebar_length,
    }
    scalebar_linestyle = remove_none_from_dict(scalebar_linestyle)

    scalebar_textstyle = {
        'y': scalebar_ymin + scalebar_length/2,
        'text': scalebar_text,
        'horizontalalignment': scalebar_texthalignment,
        'verticalalignment': scalebar_textvalignment,
        'rotation': scalebar_textrotation,
    }
    scalebar_textstyle = remove_none_from_dict(scalebar_textstyle)

    # This is what gets passed to Logo()
    scalebar_style = {
        'line_kwargs': scalebar_linestyle,
        'text_kwargs': scalebar_textstyle,
        'visible': show_scalebar}

    # Gridline styling
    gridline_dict = {
        'axis': gridline_axis,
        'alpha': gridline_alpha,
        'color': gridline_color,
        'linewidth': gridline_width,
        'linestyle': gridline_style,
        'visible': show_gridlines,
    }
    gridline_dict = remove_none_from_dict(gridline_dict)

    # Set baseline defaults
    if baseline_color is None:
        baseline_color = mpl.rcParams['axes.edgecolor']
    if baseline_alpha is None:
        baseline_alpha = 1
    if baseline_width is None:
        baseline_width = mpl.rcParams['axes.linewidth']
    if baseline_style is None:
        baseline_style = '-'
    if baseline_zorder is None:
        baseline_zorder = 10

    # Baseline styling
    baseline_dict = {
        'color': baseline_color,
        'alpha': baseline_alpha,
        'linewidth': baseline_width,
        'linestyle': baseline_style,
        'zorder': baseline_zorder
    }

    # Set vlines defaults
    if vline_color is None:
        vline_color = mpl.rcParams['axes.edgecolor']
    if vline_alpha is None:
        vline_alpha = 1
    if vline_width is None:
        vline_width = mpl.rcParams['axes.linewidth']
    if vline_style is None:
        vline_style = '-'
    if vline_zorder is None:
        vline_zorder = 30

    # vlines styling
    vline_dict = {
        'color': vline_color,
        'alpha': vline_alpha,
        'linewidth': vline_width,
        'linestyle': vline_style,
        'zorder': vline_zorder
    }

    # Set axes_style dictionary
    axes_style = {
        #'show_position_zero': show_position_zero,
        'show_binary_yaxis': show_binary_yaxis,
        'show_baseline': show_baseline,
        'baseline_dict': baseline_dict,
        'vline_positions': vline_positions,
        'vline_dict': vline_dict,
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
    logo = Logo(matrix=dataframe,
                #highlight_sequence=highlight_sequence,
                highlight_sequence=highlight_style_dict['highlight_sequence'],
                fullheight=fullheight,
                font_properties=font_properties,
                character_style=character_style,
                highlight_style=highlight_style,
                fullheight_style=fullheight_style,
                placement_style=placement_style,
                axes_style=axes_style,
                scalebar_style=scalebar_style)

    # Decorate logo
    logo.logo_type = logo_type
    logo.background = background
    logo.bg_mat = bg_mat

    ######################################################################
    # Optionally draw logo

    # Set RC parameters
    for key, value in rcparams.items():
        mpl.rcParams[key] = value

    # Set default figsize
    if figsize is None:
        figsize = [default_fig_width, default_fig_height_per_line]

    # If user specifies a figure size, make figure and axis,
    # draw logo, then return all three
    if draw_now:

        fig, ax = plt.subplots(figsize=figsize)
        logo.draw(ax)

        if use_tightlayout:
            plt.tight_layout()
            plt.draw()

        if save_to_file:
            fig.savefig(save_to_file, dpi=dpi)
        plt.draw()

        logo.ax = ax
        logo.fig = fig

    # Otherwise, just return logo to user without drawing
    else:
        logo.ax = None
        logo.fig = None

    # Return logo to user
    return logo


# Document make_logo
path = os.path.dirname(__file__)
#document_function(make_logo, '%s/make_logo_arguments.txt'%path)
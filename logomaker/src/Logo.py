from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes

# Import stuff from logomaker
from logomaker.src.Glyph import Glyph, list_font_names
from logomaker.src.validate import validate_matrix
from logomaker.src.error_handling import check, handle_errors
from logomaker.src.colors import get_color_dict, get_rgb
from logomaker.src.matrix import transform_matrix


class Logo:
    """
    Logo represents a basic logo, drawn on a specified axes object
    using a specified matrix, which is supplied as a pandas dataframe.

    attributes
    ----------

    df: (pd.DataFrame)
        A matrix specifying character heights and positions. Rows correspond
        to positions while columns correspond to characters. Column names
        must be single characters and row indices must be integers.

    color_scheme: (str, dict, or array with length 3)
        Specification of logo colors. Default is 'gray'. Can take a variety of
        forms.
         - (str) A built-in Logomaker color scheme in which the color of each
         character is determined that character's identity. Options are,
             + For DNA/RNA: 'classic', 'grays', or 'base_paring'.
             + For protein: 'hydrophobicity', 'chemistry', or 'charge'.
         - (str) A built-in matplotlib color name such as 'k' or 'tomato'
         - (list) An RGB array, i.e., 3 floats with values in the interval [0,1]
         - (dict) A dictionary that maps characters to colors, E.g.,
            {'A': 'blue',
             'C': 'yellow',
             'G': 'green',
             'T': 'red'}

    font_name: (str)
        The character font to use when rendering the logo. For a list of
        valid font names, run logomaker.list_font_names().

    stack_order: (str)
        Must be 'big_on_top', 'small_on_top', or 'fixed'. If 'big_on_top',
        stack characters away from x-axis in order of increasing absolute value.
        If 'small_on_top', stack glyphs away from x-axis in order of
        decreasing absolute value. If 'fixed', stack glyphs from top to bottom
        in the order that characters appear in the data frame.

    center_values: (bool)
        If True, the stack of characters at each position will be centered
        around zero. This is accomplished by subtracting the mean value
        in each row of the matrix from each element in that row.

    baseline_width: (float >= 0.0)
        Width of the logo baseline, drawn at value 0.0 on the y-axis.

    flip_below: (bool)
        If True, characters below the x-axis (which correspond to negative
        values in the matrix) will be flipped upside down.

    shade_below: (float in [0,1])
        The amount of shading to use for characters drawn below the x-axis.
        Larger numbers correspond to more shading (i.e., darker characters).

    fade_below: (float in [0,1])
        The amount of fading to use for characters drawn below the x-axis.
        Larger numbers correspond to more fading (i.e., more transparent
        characters).

    fade_probabilities: (bool)
        If True, the characters in each stack will be assigned an alpha value
        equal to their height. This option only makes sense if df is a
        probability matrix. For additional customization, use
        Logo.fade_glyphs_in_probability_logo().

    vpad: (float in [0,1])
        The whitespace to leave above and below each character within that
        character's bounding box. Note that, if vpad > 0, the height of the
        character's bounding box (and not of the character itself) will
        correspond to values in df.

    vsep: (float >= 0)
        Amount of whitespace to leave between the bounding boxes of rendered
        characters. Unlike vpad, vsep is NOT relative to character height.

    alpha: (float in [0,1])
        Opacity to use when rendering characters. Note that, if this is used
        together with fade_below or fade_probabilities, alpha will multiply
        existing opacity values.

    show_spines: (None or bool)
        Whether a box should be drawn around the logo. Note: if set to either
        True or False, the Logo will be drawn immediately and draw_now will
        be overridden. For additional customization of spines, use
        Logo.style_spines().

    ax: (matplotlib Axes object)
        The matplotlb Axes object on which the logo is drawn.

    zorder: (int >=0)
        This governs what other objects drawn on ax will appear in front or
        behind the rendered logo.

    figsize: ([float, float]):
        The default figure size for the rendered logo; only used if ax is
        not supplied by the user.

    draw_now: (bool)
        If True, the logo is rendered immediately after it is specified.
        Set to False if you wish to change the properties of any characters
        after initial specification.

    **kwargs:
        Additional key word arguments to send to the Glyph constructor.
    """

    @handle_errors
    def __init__(self,
                 df,
                 color_scheme=None,
                 font_name='sans',
                 stack_order='big_on_top',
                 center_values=False,
                 baseline_width=0.5,
                 flip_below=True,
                 shade_below=0.0,
                 fade_below=0.0,
                 fade_probabilities=False,
                 vpad=0.0,
                 vsep=0.0,
                 alpha=1.0,
                 show_spines=None,
                 ax=None,
                 zorder=0,
                 figsize=(10, 2.5),
                 draw_now=True,
                 **kwargs):

        # set class attributes
        self.df = df
        self.color_scheme = color_scheme
        self.font_name = font_name
        self.stack_order = stack_order
        self.center_values = center_values
        self.baseline_width = baseline_width
        self.flip_below = flip_below
        self.shade_below = shade_below
        self.fade_below = fade_below
        self.fade_probabilities = fade_probabilities
        self.vpad = vpad
        self.vsep = vsep
        self.alpha = alpha
        self.show_spines = show_spines
        self.zorder = zorder
        self.figsize = figsize
        self.ax = ax
        self.draw_now = draw_now

        # save other keyword arguments
        self.glyph_kwargs = kwargs

        # register logo as NOT having been drawn
        # This is changed to True after all Glyphs have been rendered
        self.has_been_drawn = False

        # perform input checks to validate attributes
        self._input_checks()

        # compute length
        self.L = len(self.df)

        # get list of characters
        self.cs = np.array([c for c in self.df.columns])
        self.C = len(self.cs)

        # get color dictionary
        # NOTE: this validates color_scheme; not self._input_checks()
        self.rgb_dict = get_color_dict(self.color_scheme, self.cs)

        # get list of positions
        self.ps = np.array([p for p in self.df.index])

        # center matrix if requested
        if self.center_values:
            self.df = transform_matrix(self.df, center_values=True)

        # compute characters
        self._compute_glyphs()

        # style glyphs below x-axis
        self.style_glyphs_below(shade=self.shade_below,
                                fade=self.fade_below,
                                draw_now=self.draw_now,
                                ax=self.ax)

        # fade glyphs by value if requested
        if self.fade_probabilities:
            self.fade_glyphs_in_probability_logo(v_alpha0=0,
                                                 v_alpha1=1,
                                                 draw_now=self.draw_now)

        # Either show or hide spines based on self.show_spines
        if self.show_spines is not None:
            self.style_spines(visible=self.show_spines)

        # Draw now if requested
        if self.draw_now:
            self.draw()

    def _input_checks(self):
        """
        Validate parameters passed to the Logo constructor EXCEPT for
        color_scheme; that is validated in the Logo constructor
        """

        # validate dataframe
        self.df = validate_matrix(self.df)

        # CANNOT validate color_scheme here; this is done in Logo constructor.

        # validate that font_name is a str
        check(isinstance(self.font_name, str),
              'type(font_name) = %s must be of type str' % type(self.font_name))

        # validate font_name value
        check(self.font_name in list_font_names(),
              'Invalid choice for font_name. For a list of valid choices, '
              'please call logomaker.list_font_names().')

        # validate stack_order
        valid_stack_orders = {'big_on_top', 'small_on_top', 'fixed'}
        check(self.stack_order in valid_stack_orders,
              'stack_order = %s; must be in %s.' %
              (self.stack_order, valid_stack_orders))

        # check that center_values is a boolean
        check(isinstance(self.center_values, bool),
              'type(center_values) = %s; must be of type bool.' %
              type(self.center_values))

        # check baseline_width is a number
        check(isinstance(self.baseline_width, (int, float)),
              'type(baseline_width) = %s must be of type number' %
              (type(self.baseline_width)))

        # check baseline_width >= 0.0
        check(self.baseline_width >= 0.0,
              'baseline_width = %s must be >= 0.0' % self.baseline_width)

        # check that flip_below is boolean
        check(isinstance(self.flip_below, bool),
              'type(flip_below) = %s; must be of type bool ' %
              type(self.flip_below))

        # validate that shade_below is a number
        check(isinstance(self.shade_below, (float, int)),
              'type(shade_below) = %s must be of type float' %
              type(self.shade_below))

        # validate that shade_below is between 0 and 1
        check(0.0 <= self.shade_below <= 1.0,
              'shade_below must be between 0 and 1')

        # validate that fade_below is a number
        check(isinstance(self.fade_below, (float, int)),
              'type(fade_below) = %s must be of type float' %
              type(self.fade_below))

        # validate that fade_below is between 0 and 1
        check(0.0 <= self.fade_below <= 1.0,
              'fade_below must be between 0 and 1')

        # validate that fade_probabilities is boolean
        check(isinstance(self.fade_probabilities, bool),
              'type(fade_probabilities) = %s; must be of type bool '
              % type(self.fade_probabilities))

        # validate that vpad is a number
        check(isinstance(self.vpad, (float, int)),
              'type(vpad) = %s must be of type float' % type(self.vpad))

        # validate that vpad is between 0 and 1
        check(0.0 <= self.vpad <= 1.0, 'vpad must be between 0 and 1')

        # validate that vsep is a number
        check(isinstance(self.vsep, (float, int)),
              'type(vsep) = %s; must be of type float or int ' %
              type(self.vsep))

        # validate that vsep is >= 0
        check(self.vsep >= 0,
              "vsep = %d must be greater than 0 " % self.vsep)

        # validate that alpha is a number
        check(isinstance(self.alpha, (float, int)),
              'type(alpha) = %s must be of type float' % type(self.alpha))

        # validate that alpha is between 0 and 1
        check(0.0 <= self.alpha <= 1.0, 'alpha must be between 0 and 1')

        # validate show_spines is None or boolean
        check(isinstance(self.show_spines, bool) or (self.show_spines is None),
              'show_spines = %s; show_spines must be None or boolean.'
              % repr(self.show_spines))

        # validate ax
        check(isinstance(self.ax, Axes) or (self.ax is None),
              'ax = %s; ax must be None or a matplotlib.Axes object.' %
              repr(self.ax))

        # validate zorder
        check(isinstance(self.zorder, (float, int)),
              'type(zorder) = %s; zorder must be a number.' %
              type(self.zorder))

        # validate that figsize is array=like
        check(isinstance(self.figsize, (tuple, list, np.ndarray)),
              'type(figsize) = %s; figsize must be array-like.' %
              type(self.figsize))
        self.figsize = tuple(self.figsize) # Just to pin down variable type.

        # validate length of figsize
        check(len(self.figsize) == 2, 'figsize must have length two.')

        # validate that each element of figsize is a number
        check(all([isinstance(n, (int, float)) and n > 0
                   for n in self.figsize]),
              'all elements of figsize array must be numbers > 0.')

        # check that draw_now is a boolean
        check(isinstance(self.draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' %
              type(self.draw_now))

    @handle_errors
    def style_glyphs(self,
                     color_scheme=None,
                     draw_now=True,
                     ax=None,
                     **kwargs):
        """
        Modifies the properties of all characters in a Logo.

        parameters
        ----------

        color_scheme: (str, dict, or array with length 3)
            Specification of logo colors. Default is 'gray'. Can take a variety of
            forms.
             - (str) A built-in Logomaker color scheme in which the color of each
             character is determined that character's identity. Options are,
                 + For DNA/RNA: 'classic', 'grays', or 'base_paring'.
                 + For protein: 'hydrophobicity', 'chemistry', or 'charge'.
             - (str) A built-in matplotlib color name such as 'k' or 'tomato'
             - (list) An RGB array, i.e., 3 floats with values in the interval [0,1]
             - (dict) A dictionary that maps characters to colors, E.g.,
                {'A': 'blue',
                 'C': 'yellow',
                 'G': 'green',
                 'T': 'red'}

        draw_now: (bool)
            Whether to re-draw modified logo on current Axes object.

        ax: (matplotlib Axes object)
            New Axes object, if any, on which to draw the Logo.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        # validate color_scheme and get dict representation;
        # set as self.rgb_dict, i.e., change the Logo's self-identified
        # color scheme
        if color_scheme is not None:
            self.color_scheme = color_scheme
            self.rgb_dict = get_color_dict(self.color_scheme, self.cs)

        # check that draw_now is a boolean
        check(isinstance(draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' %
              type(draw_now))

        # check ax
        check((ax is None) or isinstance(ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # update ax if axes are provided by the user
        self._update_ax(ax)

        # update glyph-specific attributes if they are passed as kwargs
        for key in ['zorder', 'vpad', 'font_name']:
            if key in kwargs.keys():
                self.__dict__[key] = kwargs[key]

        # modify all glyphs
        for g in self.glyph_list:

            # add color to kwargs dict, but only if user is updating color
            if color_scheme is not None:
                kwargs['color'] = self.rgb_dict[g.c]

            # set each glyph attribute
            g.set_attributes(**kwargs)

        # draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def fade_glyphs_in_probability_logo(self,
                                        v_alpha0=0.0,
                                        v_alpha1=1.0,
                                        draw_now=True,
                                        ax=None):

        """
        Fades glyphs in probability logo according to value.

        parameters
        ----------

        v_alpha0, v_alpha1: (number in [0,1])
            Matrix values marking values that are rendered using
            alpha=0 and alpha=1, respectively. These values must satisfy
            v_alpha0 < v_alpha1.

        draw_now: (bool)
            Whether to readraw modified Logo.

        ax: (matplotlib Axes object)
            New Axes object, if any, on which to draw the Logo.

        returns
        -------
        None
         """

        # validate alpha0
        check(isinstance(v_alpha0, (float, int)),
              'type(v_alpha0) = %s must be a number' %
              type(v_alpha0))

        # ensure that v_alpha0 is between 0 and 1
        check(0.0 <= v_alpha0 <= 1.0,
              'v_alpha0 must be between 0 and 1; value is %f.' % v_alpha0)

        # validate alpha1
        check(isinstance(v_alpha1, (float, int)),
              'type(v_alpha1) = %s must be a number' %
              type(v_alpha1))

        # ensure that v_alpha1 is between 0 and 1
        check(0.0 <= v_alpha1 <= 1.0,
              'v_alpha1 must be between 0 and 1; value is %f' % v_alpha1)

        # check that v_alpha0 < v_alpha1
        check(v_alpha0 < v_alpha1,
              'must have v_alpha0 < v_alpha1;'
              'here, v_alpha0 = %f and v_alpha1 = %f' % (v_alpha0, v_alpha1))

        # check that draw_now is a boolean
        check(isinstance(draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' %
              type(draw_now))

        # check ax
        check((ax is None) or isinstance(ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # update ax if axes are provided by the user
        self._update_ax(ax)

        # make sure matrix is a probability matrix
        self.df = validate_matrix(self.df, matrix_type='probability')

        # iterate over all positions and characters
        for p in self.ps:
            for c in self.cs:

                # grab both glyph and value
                v = self.df.loc[p, c]
                g = self.glyph_df.loc[p, c]

                # compute new alpha
                if v <= v_alpha0:
                    alpha = 0
                elif v >= v_alpha1:
                    alpha = 1
                else:
                    alpha = (v - v_alpha0) / (v_alpha1 - v_alpha0)

                # Set glyph attributes
                g.set_attributes(alpha=alpha)

        # Draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def style_glyphs_below(self,
                           shade=0.0,
                           fade=0.0,
                           flip=True,
                           draw_now=True,
                           ax=None,
                           **kwargs):

        """
        Modifies the properties of all characters drawn below the x-axis.

        parameters
        ----------

        shade: (number in [0,1])
            The amount to shade characters below the x-axis.

        fade: (number in [0,1])
            The amount to fade characters below the x-axis.

        flip: (bool)
            If True, characters below the x-axis will be flipped upside down.

        ax: (matplotlib Axes object)
            The Axes object on which to draw the logo.

        draw_now: (bool)
            Whether to readraw modified Logo.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes(), but only
            for characters below the x-axis.

        returns
        -------
        None
        """

        # validate shade
        check(isinstance(shade, (float, int)),
              'type(shade) = %s must be a number' %
              type(shade))

        # ensure that v_alpha0 is between 0 and 1
        check(0.0 <= shade <= 1.0,
              'shade must be between 0 and 1; value is %f.' % shade)

        # validate fade
        check(isinstance(fade, (float, int)),
              'type(fade) = %s must be a number' %
              type(fade))

        # ensure that fade is between 0 and 1
        check(0.0 <= fade <= 1.0,
              'fade must be between 0 and 1; value is %f' % fade)

        # check that flip is a boolean
        check(isinstance(flip, bool),
              'type(flip) = %s; must be of type bool ' %
              type(flip))

        # check ax
        check((ax is None) or isinstance(ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # update ax if axes are provided by the user
        self._update_ax(ax)

        # check that draw_now is a boolean
        check(isinstance(draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' %
              type(draw_now))

        # iterate over all positions and characters
        for p in self.ps:
            for c in self.cs:

                # check if matrix value is < 0
                v = self.df.loc[p, c]
                if v < 0:

                    # get glyph
                    g = self.glyph_df.loc[p, c]

                    # modify color and alpha
                    color = np.array(g.color) * (1.0 - shade)
                    alpha = g.alpha * (1.0 - fade)

                    # set glyph attributes
                    g.set_attributes(color=color,
                                     alpha=alpha,
                                     flip=flip,
                                     **kwargs)

        # draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def style_single_glyph(self, p, c, draw_now=True, ax=None, **kwargs):
        """
        Modifies the properties of a single character in Logo.

        parameters
        ----------

        p: (int)
            Position of modified glyph. Must index a row in the matrix df passed
            to the Logo constructor.

        c: (str of length 1)
            Character to modify. Must be the name of a column in the matrix df
            passed to the Logo constructor.

        draw_now: (bool)
            Specifies whether to readraw the modified Logo on the current Axes.
            Warning: setting this to True will slow down rendering.

        ax: (matplotlib Axes object)
            New Axes object, if any, on which to draw Logo.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        # validate p is an integer
        check(isinstance(p, int),
              'type(p) = %s must be of type int' % type(p))

        # check p is a valid position
        check(p in self.glyph_df.index,
              'p=%s is not a valid position' % p)

        # validate c is a str
        check(isinstance(c, str),
              'type(c) = %s must be of type str' % type(c))

        # validate that c has length 1
        check(len(c) == 1,
              'c = %s; must have length 1.' % repr(c))

        # check c is a valid character
        check(c in self.glyph_df.columns,
              'c=%s is not a valid character' % c)

        # check that draw_now is a boolean
        check(isinstance(draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' % type(draw_now))

        # check ax
        check((ax is None) or isinstance(ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # update ax if provided by the user.
        self._update_ax(ax)

        # Get glyph from glyph_df
        g = self.glyph_df.loc[p, c]

        # update glyph attributes
        g.set_attributes(**kwargs)

        # draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def style_glyphs_in_sequence(self,
                                 sequence,
                                 draw_now=True,
                                 ax=None,
                                 **kwargs):
        """
        Restyles the glyphs in a specific sequence.

        parameters
        ----------
        sequence: (str)
            A string the same length as the logo, specifying which character
            to restyle at each position. Characters in sequence that are not
            in the columns of the Logo's df are ignored.

        draw_now: (bool)
            Whether to readraw modified logo on the current Axes.

        ax: (matplotlib Axes object)
            New Axes object, if any, on which to draw logo.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        # validate sequence is a string
        check(isinstance(sequence, str),
              'type(sequence) = %s must be of type str' % type(sequence))

        # validate sequence has correct length
        check(len(sequence) == self.L,
              'sequence to restyle (length %d) ' % len(sequence) +
              'must have same length as logo (length %d).' % self.L)

        # check that draw_now is a boolean
        check(isinstance(draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' % type(draw_now))

        # check ax
        check((ax is None) or isinstance(ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # update Axes if provided by the user
        self._update_ax(ax)

        # for each position in the logo...
        for i, p in enumerate(self.glyph_df.index):

            # get character to highlight
            c = sequence[i]

            # modify the glyph corresponding character c at position p.
            # only modify if c is a valid character; if not, ignore position
            if c in self.cs:
                self.style_single_glyph(p, c, draw_now=False, **kwargs)

        # draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def highlight_position(self, p, **kwargs):

        """
        Draws a rectangular box highlighting a specific position.

        parameters
        ----------
        p: (int)
            Single position to highlight.

        **kwargs:
            Other parameters to pass to highlight_position_range()

        returns
        -------
        None
        """

        # validate p
        check(isinstance(p, int),
              'type(p) = %s must be of type int' % type(p))

        # If logo is not yet drawn, draw
        if not self.has_been_drawn:
            self.draw()

        # pass the buck to highlight_position_range
        self.highlight_position_range(pmin=p, pmax=p, **kwargs)

    @handle_errors
    def highlight_position_range(self,
                                 pmin,
                                 pmax,
                                 padding=0.0,
                                 color='yellow',
                                 edgecolor=None,
                                 floor=None,
                                 ceiling=None,
                                 zorder=-2,
                                 **kwargs):
        """
        Draws a rectangular box highlighting multiple positions within the Logo

        parameters
        ----------
        pmin: (int)
            Lowest position to highlight.
            
        pmax: (int)
            Highest position to highlight.
            
        padding: (number >= -0.5)
            Amount of padding to add on the left and right sides of highlight.
            
        color: (None or matplotlib color)
            Color to use for highlight. Can be a named matplotlib color or
            an RGB array.

        edgecolor: (None or matplotlib color)
            Color to use for highlight box edges. Can be a named matplotlib
            color or an RGB array.
            
        floor: (None number)
            Lowest y-axis extent of highlight box. If None, is set to
            ymin of the Axes object.
            
        ceiling: (None or number)
            Highest y-axis extent of highlight box. If None, is set to
            ymax of the Axes object.
            
        zorder: (number)
            This governs which other objects drawn on ax will appear in front or
            behind of the highlight. Logo characters are, by default, drawn in
            front of the highlight box.

        returns
        -------
        None
        """

        # draw if logo has not yet been drawn
        if not self.has_been_drawn:
            self.draw()

        # get ymin and ymax from Axes object
        ymin, ymax = self.ax.get_ylim()

        # validate pmin
        check(isinstance(pmin, (float, int)),
              'type(pmin) = %s must be a number' % type(pmin))

        # validate pmax
        check(isinstance(pmax, (float, int)),
              'type(pmax) = %s must be a number' % type(pmax))

        # Make sure pmin <= pmax
        check(pmin <= pmax,
              'pmin <= pmax not satisfied.')

        # validate that padding is a valid number
        check(isinstance(padding, (float, int)) and padding >= -0.5,
              'padding = %s must be a number >= -0.5' % repr(padding))

        # validate color
        if color is not None:
            color = get_rgb(color)

        # validate edegecolor
        if edgecolor is not None:
            edgecolor = get_rgb(edgecolor)

        # validate floor and set to ymin if None
        if floor is None:
            floor = ymin
        else:
            check(isinstance(floor, (float, int)),
                  'type(floor) = %s must be a number' % type(floor))

        # validate ceiling and set to ymax if None
        if ceiling is None:
            ceiling = ymax
        else:
            check(isinstance(ceiling, (float, int)),
                  'type(ceiling) = %s must be a number' % type(ceiling))

        # now that floor and ceiling are set, validate that floor <= ceiling
        check(floor <= ceiling,
              'must have floor <= ceiling; as is, floor = %f, ceiling = %s' %
              (floor, ceiling))

        # validate zorder
        check(isinstance(zorder, (float, int)),
              'type(zorder) = %s; must a float or int.' % type(zorder))

        # compute coordinates of highlight rectangle
        x = pmin - .5 - padding
        y = floor
        width = pmax - pmin + 1 + 2*padding
        height = ceiling - floor

        # specify rectangle
        patch = Rectangle(xy=(x, y),
                          width=width,
                          height=height,
                          facecolor=color,
                          edgecolor=edgecolor,
                          zorder=zorder,
                          **kwargs)

        # add rectangle to Axes
        self.ax.add_patch(patch)

    @handle_errors
    def draw_baseline(self,
                      zorder=-1,
                      color='black',
                      linewidth=0.5,
                      **kwargs):
        """
        Draws a horizontal line along the x-axis.

        parameters
        ----------

        zorder: (number)
            This governs what other objects drawn on ax will appear in front or
            behind the baseline. Logo characters are, by default, drawn in front
            of the baseline.

        color: (matplotlib color)
            Color to use for the baseline. Can be a named matplotlib color or an
            RGB array.

        linewidth: (number >= 0)
            Width of the baseline.

        **kwargs:
            Additional keyword arguments to be passed to ax.axhline()

        returns
        -------
        None
        """

        # validate zorder
        check(isinstance(zorder, (float, int)),
              'type(zorder) = %s; must a float or int.' % type(zorder))

        # validate color
        color = get_rgb(color)

        # validate that linewidth is a number
        check(isinstance(linewidth, (float, int)),
              'type(linewidth) = %s; must be a number ' % type(linewidth))

        # validate that linewidth >= 0
        check(linewidth >= 0, 'linewidth must be >= 0')

        # draw if logo has not yet been drawn
        if not self.has_been_drawn:
            self.draw()

        # Render baseline
        self.ax.axhline(zorder=zorder,
                        color=color,
                        linewidth=linewidth,
                        **kwargs)

    @handle_errors
    def style_xticks(self,
                     anchor=0,
                     spacing=1,
                     fmt='%d',
                     rotation=0.0,
                     **kwargs):
        """
        Formats and styles tick marks along the x-axis.

        parameters
        ----------

        anchor: (int)
            Anchors tick marks at a specific number. Even if this number
            is not within the x-axis limits, it fixes the register for
            tick marks.

        spacing: (int > 0)
            The spacing between adjacent tick marks

        fmt: (str)
            String used to format tick labels.

        rotation: (number)
            Angle, in degrees, with which to draw tick mark labels.

        **kwargs:
            Additional keyword arguments to be passed to ax.set_xticklabels()

        returns
        -------
        None
        """

        # validate anchor
        check(isinstance(anchor, int),
              'type(anchor) = %s must be of type int' % type(anchor))

        # validate spacing
        check(isinstance(spacing, int) and spacing > 0,
              'spacing = %s must be an int > 0' % repr(spacing))

        # validate fmt
        check(isinstance(fmt, str),
              'type(fmt) = %s must be of type str' % type(fmt))

        # validate rotation
        check(isinstance(rotation, (float, int)),
              'type(rotation) = %s; must be of type float or int ' %
              type(rotation))

        # draw if logo has not yet been drawn
        if not self.has_been_drawn:
            self.draw()

        # Get list of positions that span all positions in the matrix df
        p_min = min(self.ps)
        p_max = max(self.ps)
        ps = np.arange(p_min, p_max+1)

        # Compute and set xticks
        xticks = ps[(ps - anchor) % spacing == 0]
        self.ax.set_xticks(xticks)

        # Compute and set xticklabels
        xticklabels = [fmt % p for p in xticks]
        self.ax.set_xticklabels(xticklabels, rotation=rotation, **kwargs)

    @handle_errors
    def style_spines(self,
                     spines=('top', 'bottom', 'left', 'right'),
                     visible=True,
                     linewidth=1.0,
                     color='black',
                     bounds=None):
        """
        Styles the spines of the Axes object in which the logo is drawn.
        Note: "spines" refers to the edges of the Axes bounding box.

        parameters
        ----------

        spines: (tuple of str)
            Specifies which of the four spines to modify. The default value
            for this parameter lists all four spines.

        visible: (bool)
            Whether to show or not show the spines listed in the parameter
            spines.

        color: (matplotlib color)
            Color of the spines. Can be a named matplotlib color or an
            RGB array.

        linewidth: (float >= 0)
            Width of lines used to draw the spines.

        bounds: (None or [float, float])
            If not None, specifies the values between which a spine (or spines)
            will be drawn.

        returns
        -------
        None
        """

        # validate that spines is a set-like object
        check(isinstance(spines, (tuple, list, set)),
              'type(spines) = %s; must be of type (tuple, list, set) ' %
              type(spines))
        spines = set(spines)

        # validate that spines is a subset of a the valid spines choices
        valid_spines = {'top', 'bottom', 'left', 'right'}
        check(spines <= valid_spines,
              'spines has invalid entry; valid entries are: %s' %
              repr(valid_spines))

        # validate visible
        check(isinstance(visible, bool),
              'type(visible) = %s; must be of type bool ' %
              type(visible))

        # validate that linewidth is a number
        check(isinstance(linewidth, (float, int)),
              'type(linewidth) = %s; must be a number ' % type(linewidth))

        # validate that linewidth >= 0
        check(linewidth >= 0, 'linewidth must be >= 0')

        # validate color
        color = get_rgb(color)

        # validate bounds. If not None, validate entries.
        if bounds is not None:

            # check that bounds are of valid type
            bounds_types = (list, tuple, np.ndarray)
            check(isinstance(bounds, bounds_types),
                  'type(bounds) = %s; must be one of %s' % (
                  type(bounds), bounds_types))

            # check that bounds has right length
            check(len(bounds) == 2,
                  'len(bounds) = %d; must be %d' % (len(bounds), 2))

            # ensure that elements of bounds are numbers
            check(all([isinstance(bound, (float, int)) for bound in bounds]),
                  'bounds = %s; all entries must be numbers' % repr(bounds))

            # bounds entries must be sorted
            check(bounds[0] < bounds[1],
                  'bounds = %s; must have bounds[0] < bounds[1]' % repr(bounds))

        # draw if logo has not yet been drawn
        if not self.has_been_drawn:
            self.draw()

        # iterate over all spines
        for name, spine in self.ax.spines.items():

            # If name is in the set of spines to modify
            if name in spines:

                # Modify given spine
                spine.set_visible(visible)
                spine.set_color(color)
                spine.set_linewidth(linewidth)

                if bounds is not None:
                    spine.set_bounds(bounds[0], bounds[1])

    def draw(self, ax=None):
        """
        Draws characters on the Axes object 'ax' provided to the Logo
        constructor. Note: all previous content drawn on ax will be erased.

        parameters
        ----------
        ax: (Axes object)
            Axes on which to draw the logo. Defaults to the previously
            specified Axes or, if that was never set, to plt.gca().

        returns
        -------
        None
        """

        # update ax
        self._update_ax(ax)

        # if ax is still None, create figure
        if self.ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.ax = ax

        # clear previous content from ax
        self.ax.clear()

        # draw each glyph
        for g in self.glyph_list:
            g.draw(self.ax)

        # flag that this logo has indeed been drawn
        self.has_been_drawn = True

        # draw baseline
        self.draw_baseline(linewidth=self.baseline_width)

        # set xlims
        xmin = min([g.p - .5*g.width for g in self.glyph_list])
        xmax = max([g.p + .5*g.width for g in self.glyph_list])
        self.ax.set_xlim([xmin, xmax])

        # set ylims
        ymin = min([g.floor for g in self.glyph_list])
        ymax = max([g.ceiling for g in self.glyph_list])
        self.ax.set_ylim([ymin, ymax])

    def _update_ax(self, ax):
        """ Reset ax if user has passed a new one."""
        if ax is not None:
            self.ax = ax

    def _compute_glyphs(self):
        """
        Specifies the placement and styling of all glyphs within the logo.
        Note that glyphs can later be changed after this is called but before
        draw() is called.
        """
        # Create a dataframe of glyphs
        glyph_df = pd.DataFrame()

        # For each position
        for p in self.ps:

            # get values at this position
            vs = np.array(self.df.loc[p, :])

            # Sort values according to the order in which the user
            # wishes the characters to be stacked
            if self.stack_order == 'big_on_top':
                ordered_indices = np.argsort(vs)

            elif self.stack_order == 'small_on_top':
                tmp_vs = np.zeros(len(vs))
                indices = (vs != 0)
                tmp_vs[indices] = 1.0/vs[indices]
                ordered_indices = np.argsort(tmp_vs)

            elif self.stack_order == 'fixed':
                ordered_indices = np.array(range(len(vs)))[::-1]

            else:
                assert False, 'This line of code should never be called.'

            # Reorder values and characters
            vs = vs[ordered_indices]
            cs = [str(c) for c in self.cs[ordered_indices]]

            # Set floor
            floor = sum((vs - self.vsep) * (vs < 0)) + self.vsep/2.0

            # For each character
            for v, c in zip(vs, cs):

                # Set ceiling
                ceiling = floor + abs(v)

                # Set color
                this_color = self.rgb_dict[c]

                # Set whether to flip character
                flip = (v < 0 and self.flip_below)

                # Create glyph if height is finite
                glyph = Glyph(p, c,
                              ax=self.ax,
                              floor=floor,
                              ceiling=ceiling,
                              color=this_color,
                              flip=flip,
                              draw_now=False,
                              zorder=self.zorder,
                              font_name=self.font_name,
                              alpha=self.alpha,
                              vpad=self.vpad,
                              **self.glyph_kwargs)

                # Add glyph to glyph_df
                glyph_df.loc[p, c] = glyph

                # Raise floor to current ceiling
                floor = ceiling + self.vsep

        # Set glyph_df attribute
        self.glyph_df = glyph_df
        self.glyph_list = [g for g in self.glyph_df.values.ravel()
                           if isinstance(g, Glyph)]

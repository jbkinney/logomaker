from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb, to_rgba
import matplotlib.cm
import pdb

# Import stuff from logomaker
from logomaker.src.Glyph import Glyph
from logomaker.src import colors as lm_color
from logomaker.src.validate import validate_matrix, validate_probability_mat
import logomaker.src.validate as validate
from logomaker.src.error_handling import check, handle_errors

chars_to_colors_dict = {
    tuple('ACGT'): 'classic',
    tuple('ACGU'): 'classic',
    tuple('ACDEFGHIKLMNPQRSTVWY'): 'hydrophobicity',
}


class Logo:
    """
    Logo represents a basic logo, drawn on a specified axes object
    using a specified matrix.

    attributes
    ----------

    matrix: (pd.DataFrame)
        A matrix specifying character heights and positions. Note that
        positions index rows while characters index columns.

    colors: (str, dict, or array-like with length 3 or 4)
        Face color of logo characters. Default is 'gray'. Here and in
        what follows a variable of type 'color' can take a variety of message
        types.
         - (str) A Logomaker color scheme in which the color is determined
             by the specific character being drawn. Options are,
             + For DNA/RNA: 'classic', 'grays', 'base_paring'.
             + For protein: 'hydrophobicity', 'chemistry', 'charge'.
         - (str) A built-in matplotlib color name  such as 'k' or 'tomato'
         - (list) An RGB color (3 floats in interval [0,1]) or RGBA color
             (4 floats in interval [0,1]).
         - (dict) A dictionary mapping of characters to color_scheme, in which
             case the color will depend  on the character being drawn.
             E.g., {'A': 'green','C': [ 0.,  0.,  1.], 'G': 'y',
             'T': [ 1.,  0.,  0.,  0.5]}

    font_name: (str)
        The 'font_name' parameter to pass to FontProperties() when creating
        Glyphs.

    stack_order: (str)
        Must be 'big_on_top', 'small_on_top', or 'fixed. If 'big_on_top',
        stack glyphs away from x-axis in order of increasing absolute message.
        If 'small_on_top', stack glyphs away from x-axis in order of
        decreasing absolute message. If 'fixed', stack glyphs from top to bottom
        in the order that characters appear in the data frame. If 'flipped',
        stack glyphs in the opposite order as 'fixed'.

    center_values: (bool)
        If True, the stack of characters at each position will be centered
        around zero. This is accomplished by subtracting the mean message
        in each row of the matrix from each element in that row.

    baseline_width: (float >= 0.0)
        Width of the baseline.

    flip_below: (bool)
        If True, glyphs below the x-axis (which correspond to negative
        values in the matrix) will be flipped upside down.

    fade_probabilities: (bool)
        If True, the glyphs in each stack will then be assigned an alpha value
        equal to their height. Note: this option only makes sense if df is a
        probability matrix. For additional customization, use
        Logo.fade_glyphs_in_probability_logo().

    shade_below: (float in [0,1])
        The amount of shading underneath x-axis.

    fade_below: (float in [0,1])
        The amount of fading underneath x-axis.

    vsep: (float >= 0)
        Amount of whitespace to leave between rendered glyphs. Unlike vpad,
        vsep is NOT relative to glyph height. The vsep-sized margin between
        glyphs on either side of the x-axis will always be centered on the
        x-axis.

    show_spines: (bool)
        Whether should be shown on all four sides of the logo. If set to either
        True or false, will automatically set draw_now=True.
        For additional customization of spines, use Logo.style_spines().

    zorder: (int >=0)
        The order in which things are drawn.

    figsize: (number, number):
        The default figure size for logos; only needed if ax is not supplied.

    ax: (matplotlib Axes object)
        The axes object on which to draw the logo.

    draw_now: (bool)
        If True, the logo is rendered immediately after it is specified.
        Set to False if you wish to change the properties of any glyphs
        after initial specification, e.g. by running
        Logo.highlight_sequence().

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
                 vsep=0.0,
                 show_spines=None,
                 zorder=0,
                 figsize=(10, 2.5),
                 ax=None,
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
        self.vsep = vsep
        self.show_spines = show_spines
        self.zorder = zorder
        self.figsize = figsize
        self.ax = ax
        self.draw_now = draw_now

        # Register logo as NOT having been drawn
        self.has_been_drawn = False

        # perform input checks to validate attributes
        self._input_checks()

        # Compute length
        self.L = len(self.df)

        # Get list of characters
        self.cs = np.array([str(c) for c in self.df.columns])
        self.C = len(self.cs)

        # Get list of positions
        self.ps = np.array([int(p) for p in self.df.index])

        # Set color_scheme by identifying default or otherwise setting to gray
        if color_scheme is None:
            key = tuple(self.cs)
            color_scheme = chars_to_colors_dict.get(key, 'gray')
        self.color_scheme = color_scheme

        # If fade_probabilities is True, make df a probability matrix
        if fade_probabilities:
            self.df = validate_probability_mat(self.df)

        # Save other attributes
        self.ax = ax
        self.center_values = center_values
        self.flip_below = flip_below
        self.vsep = vsep
        self.zorder = zorder
        self.figsize = tuple(figsize)
        self.glyph_kwargs = kwargs

        # Note: Logo does NOT expect df to change after it is passed
        # to the constructor. But one can change character attributes
        # before drawing.

        # Set flag for whether Logo has been drawn
        self.has_been_drawn = False

        # Fill NaN values of matrix_df with zero
        if self.center_values:
            self.df.loc[:, :] = self.df.values - \
                                self.df.values.mean(axis=1)[:, np.newaxis]

        # Compute color dictionary
        self.rgb_dict = lm_color._get_color_dict(
                                    color_scheme=self.color_scheme,
                                    chars=self.cs)

        # Compute characters.
        self._compute_glyphs()

        # Style glyphs below x-axis
        self.style_glyphs_below(shade=self.shade_below,
                                fade=self.fade_below,
                                draw_now=self.draw_now,
                                ax=self.ax)

        # Fade glyphs by value if requested
        if self.fade_probabilities:
            self.fade_glyphs_in_probability_logo(v_alpha0=0,
                                                 v_alpha1=1,
                                                 draw_now=self.draw_now)

        # Either show or hide spines based on self.show_spines
        # Note: if show_spines is not None, logo must be drawn now.
        if self.show_spines is not None:
            self.style_spines(visible=self.show_spines)

        # Draw now if requested
        if self.draw_now:
            self.draw()

    def _input_checks(self):

        """
        check input parameters in the Logo constructor for correctness
        """

        # Validate dataframe
        self.df = validate_matrix(self.df)

        # check that center_values is a boolean
        check(isinstance(self.center_values, bool),
              'type(center_values) = %s; must be of type bool ' % type(self.center_values))


        # # check that color scheme is valid
        # if self.color_scheme is not None:
        #
        #     # if color scheme is specified as a string, check that string message maps
        #     # to a valid matplotlib color scheme
        #
        #     if type(self.color_scheme) == str:
        #
        #         # get allowed list of matplotlib color schemes
        #
        #         valid_color_strings = list(matplotlib.cm.cmap_d.keys())
        #         valid_color_strings.extend(['classic', 'grays', 'base_paring','hydrophobicity', 'chemistry', 'charge'])
        #
        #         check(self.color_scheme in valid_color_strings,
        #               # 'color_scheme = %s; must be in %s' % (self.color_scheme, str(valid_color_strings)))
        #               'color_scheme = %s; is an invalid color scheme. Valid choices include classic, chemistry, grays. '
        #               'A full list of valid color schemes can be found by '
        #               'printing list(matplotlib.cm.cmap_d.keys()). ' % self.color_scheme)
        #
        #     # otherwise limit the allowed types to tuples, lists, dicts
        #     else:
        #         check(isinstance(self.color_scheme,(tuple,list,dict)),
        #               'type(color_scheme) = %s; must be of type (tuple,list,dict) ' % type(self.color_scheme))
        #
        #         # check that RGB values are between 0 and 1 is
        #         # color_scheme is a list or tuple
        #
        #         if type(self.color_scheme) == list or type(self.color_scheme) == tuple:
        #
        #             check(all(i <= 1.0 for i in self.color_scheme),
        #                   'Values of color_scheme array must be between 0 and 1')
        #
        #             check(all(i >= 0.0 for i in self.color_scheme),
        #                   'Values of color_scheme array must be between 0 and 1')

        # check baseline_width is a number
        check(isinstance(self.baseline_width,(int,float)),
              'type(baseline_width) = %s must be of type number' %(type(self.baseline_width)))

        # check baseline_width >= 0.0
        check(self.baseline_width >= 0.0,
              'baseline_width = %s must be >= 0.0' % (self.baseline_width))

        # check that stack_order is valid
        check(self.stack_order in {'big_on_top', 'small_on_top', 'fixed', 'flipped'},
              'stack_order = %s; must be "big_on_top", "small_on_top", "fixed", "flipped".' % self.stack_order)

        # check that flip_below is a boolean
        check(isinstance(self.flip_below, bool),
            'type(flip_below) = %s; must be of type bool ' % type(self.flip_below))

        # validate shade_below
        check(isinstance(self.shade_below, (float, int)),
              'type(shade_below) = %s must be of type float' % type(self.shade_below))

        # ensure that shade_below is between 0 and 1
        check(0.0 <= self.shade_below <= 1.0, 'shade_below must be between 0 and 1')

        # validate fade_below
        check(isinstance(self.fade_below, (float, int)),
              'type(fade_below) = %s must be of type float' % type(self.fade_below))

        # ensure that fade_below is between 0 and 1
        check(0.0 <= self.fade_below <= 1.0, 'fade_below must be between 0 and 1')

        # ensure fade_probabilities is of type bool
        check(isinstance(self.fade_probabilities, bool),
              'type(fade_probabilities) = %s; must be of type bool ' % type(self.fade_probabilities))

        # validate vsep
        check(isinstance(self.vsep, (float, int)),
              'type(vsep) = %s; must be of type float or int ' % type(self.vsep))

        check(self.vsep >= 0, "vsep = %d must be greater than 0 " % self.vsep)

        # validate show_spines is a bool if its not none
        if self.show_spines is not None:
            check(isinstance(self.show_spines,bool), 'type(show_spines) = %s must be of type bool'%self.show_spines)

        # validate zorder
        check(isinstance(self.zorder, int),
              'type(zorder) = %s; must be of type or int ' % type(self.zorder))

        # the following check needs to be fixed based on whether the calling function
        # is the constructor, draw_baseline, or style_glyphs_below.
        # check(self.zorder >= 0, "zorder = %d must be greater than 0 " % self.zorder)

        # validate figsize
        check(isinstance(self.figsize, (tuple, list)),
              'type(figsize) = %s; must be of type (tuple,list) ' % type(self.figsize))

        check(len(self.figsize) == 2, 'The figsize array must have two elements')

        check(all([isinstance(n, (int,float)) for n in self.figsize]),
              'all elements of figsize array must be of type int')

        check(all(i > 0 for i in self.figsize),
              'Values of figsize array must be > 0')

        # validate ax. Need to go over this in code review
        #check(isinstance(self.ax,(None,matplotlib.axes._base._AxesBase)),
              #'ax needs to be None or a valid matplotlib axis object')

        # check that draw_now is a boolean
        check(isinstance(self.draw_now, bool),
              'type(draw_now) = %s; must be of type bool ' % type(self.draw_now))

        ### after this point, the function will check inputs that are not part of the constructor. ###
        ### so checking the existence of an attribute will become necessary. ###

        # validate fade
        if(hasattr(self,'fade')):
            check(isinstance(self.fade,(float,int)), 'type(fade) = %s must be of type float' % type(self.fade))

            # ensure that fade is between 0 and 1
            check(self.fade <= 1.0 and self.fade >= 0, 'fade must be between 0 and 1')

        # validate shade
        if (hasattr(self, 'shade')):
            check(isinstance(self.shade, (float, int)), 'type(shade) = %s must be of type float' % type(self.shade))

            # ensure that fade is between 0 and 1
            check(self.shade <= 1.0 and self.shade >= 0, 'shade must be between 0 and 1')

        if (hasattr(self, 'sequence')):
            check(isinstance(self.sequence, str), 'type(sequence) = %s must be of type str' % type(self.sequence))

        # validate p
        if (hasattr(self, 'p')):
            check(isinstance(self.p, int), 'type(p) = %s must be of type int' % type(self.p))

        # validate c
        if (hasattr(self, 'c')):
            check(isinstance(self.c, str), 'type(c) = %s must be of type str' % type(self.c))

        # validate spines
        if(hasattr(self,'spines')):
            check(isinstance(self.spines, (tuple, list)),
                  'type(spines) = %s; must be of type (tuple,list) ' % type(self.spines))

            # check if items of spines are valid if tuple
            valid_spines_tuple = ('top', 'bottom', 'left', 'right')
            if(isinstance(self.spines,tuple)):

                # ensure elements of spines are valid:
                # the following code checks if spines is a subset of a the valid spines choices.
                check(set(self.spines) <= set(valid_spines_tuple),
                      'choice of spine not valid, valid choices include: '+str(valid_spines_tuple))

            # check if items of spines are valid if list
            valid_spines_list = ['top', 'bottom', 'left', 'right']

            # ensure elements of spines are valid:
            # the following code checks if spines is a subset of a the valid spines choices.
            if (isinstance(self.spines, list)):
                check(set(self.spines) <= set(valid_spines_list),
                      'choice of spine not valid, valid choices include:'+str(valid_spines_list))

        # validate visible
        if(hasattr(self,'visible')):
            check(isinstance(self.visible,bool),
                  'type(visible) = %s; must be of type bool ' % type(self.visible))

        # validate linewidth
        if(hasattr(self,'linewidth')):

            check(isinstance(self.linewidth,(float,int)),
                  'type(linewidth) = %s; must be of type float ' % type(self.linewidth))

            check(self.linewidth>=0,'linewidth must be >= 0')

        # validate bounds
        if(hasattr(self,'bounds')):

            if self.bounds is not None :
                # check that bounds are of valid type
                bounds_types = (list, tuple, np.ndarray)
                check(isinstance(self.bounds, bounds_types),
                      'type(bounds) = %s; must be one of %s' % (type(self.bounds), bounds_types))

                # bounds has right length
                check(len(self.bounds) == 2,
                      'len(bounds) = %d; must be %d' %(len(self.bounds), 2))

                # ensure that elements of bounds are numbers
                check(isinstance(self.bounds[0],(float,int)) & isinstance(self.bounds[1],(float,int)),
                      'bounds = %s; entries must be numbers' %repr(self.bounds))

                # bounds entries must be sorted
                check(self.bounds[0] < self.bounds[1],
                      'bounds = %s; entries must be sorted' %repr(self.bounds))

        # validate anchor
        if(hasattr(self,'anchor')):
            check(isinstance(self.anchor,int),'type(anchor) = %s must be of type int' % type(self.anchor))

        # validate spacing
        if (hasattr(self, 'spacing')):
            check(isinstance(self.spacing, int), 'type(spacing) = %s must be of type int' % type(self.spacing))

            check(self.spacing>0, 'spacing must be > 0')

        # validate fmt
        if(hasattr(self,'fmt')):
            check(isinstance(self.fmt,str),'type(fmt) = %s must be of type str' % type(self.fmt))

        # validate rotation
        if (hasattr(self, 'rotation')):
            check(isinstance(self.rotation, (float, int)),
                      'type(rotation) = %s; must be of type float or int ' % type(self.rotation))

        # validate alpha0
        if(hasattr(self,'v_alpha0')):
            check(isinstance(self.v_alpha0, (float, int)),
                  'type(v_alpha0) = %s must be a number' % type(self.v_alpha0))

            # ensure that v_alpha0 is between 0 and 1
            check(self.v_alpha0 <= 1.0 and self.v_alpha0 >= 0, 'v_alpha0 must be between 0 and 1')

        # validate alpha1
        if (hasattr(self, 'v_alpha1')):
            check(isinstance(self.v_alpha1, (float, int)),
                  'type(v_alpha1) = %s must be a number' % type(self.v_alpha1))

            # ensure that v_alpha1 is between 0 and 1
            check(self.v_alpha1 <= 1.0 and self.v_alpha1 >= 0, 'v_alpha1 must be between 0 and 1')

        # validate pmin
        if(hasattr(self,'pmin')):
            check(isinstance(self.pmin, (float, int)),
                  'type(pmin) = %s must be a number' % type(self.pmin))

        # validate pmax
        if (hasattr(self, 'pmax')):
            check(isinstance(self.pmax, (float, int)),
                  'type(pmax) = %s must be a number' % type(self.pmax))

        # validate padding
        if (hasattr(self, 'padding')):
            check(isinstance(self.padding, (float, int)),
                  'type(padding) = %s must be a number' % type(self.padding))

            check(self.padding>-0.5,'padding must be > -0.5')

        # validate floor
        if (hasattr(self, 'floor')):
            if(self.floor is not None):
                check(isinstance(self.floor, (float, int)),
                      'type(floor) = %s must be a number' % type(self.floor))

        # validate ceiling
        if (hasattr(self, 'ceiling')):
            if(self.ceiling is not None):
                check(isinstance(self.ceiling, (float, int)),
                      'type(ceiling) = %s must be a number' % type(self.ceiling))

        # validate saliency
        #if (hasattr(self, 'saliency')):
        #    check(isinstance(self.saliency, type([])),
        #          'type(saliency) = %s must be a list' % type(self.saliency))



    @handle_errors
    def style_glyphs(self, colors=None, draw_now=True, ax=None, **kwargs):
        """
        Modifies the properties of all glyphs in a logo.

        parameter
        ---------

        color_scheme: (color scheme)
            Color specification for glyphs. See logomaker.Logo for details.

        draw_now: (bool)
            Whether to re-draw modified logo on current Axes.

        ax: (matplotlib Axes object)
            New axes, if any, on which to draw logo if draw_now=True.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        # set attributes
        self.color_scheme = colors
        self.draw_now = draw_now

        # Update ax if axes are provided by the user.
        self._update_ax(ax)

        self._input_checks()

        # Reset color_scheme if provided
        if colors is not None:
            self.color_scheme = colors

            # the following case represents an error that may occur if a user accidentally runs
            # style glyphs before running the logo constructor. The following check puts out a
            # clean message, no need for stack-trace. hasattr checks if self has attribute cs.
            check(hasattr(self,'cs'), 'Characters entered into style glyphs are None, please ensure'
                                   ' Logo ran correctly before running style_glyphs')

            self.rgb_dict = lm_color._get_color_dict(
                                    color_scheme=self.color_scheme,
                                    chars=self.cs)

        # Record zorder if this is provided
        if 'zorder' in kwargs.keys():
            self.zorder = kwargs['zorder']

        # Modify all glyphs
        for g in self.glyph_list:

            # Set each glyph attribute
            g.set_attributes(**kwargs)

            # If color_scheme is not None, this should override
            if colors is not None:
                this_color = self.rgb_dict[g.c]
                g.set_attributes(color=this_color)

        # Draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def fade_glyphs_in_probability_logo(self,
                                        v_alpha0=0,
                                        v_alpha1=1,
                                        draw_now=True,
                                        ax=None):

        """
        Fades glyphs in probability logo according to message

        parameter
        ---------

        v_alpha0 / v_alpha1: (number in [0,1])
            Matrix values marking alpha=0 and alpha=1

        draw_now: (bool)
            Whether to readraw modified logo on current Axes.

        ax: (matplotlib Axes object)
            New axes, if any, on which to draw logo if draw_now=True.

        returns
        -------
        None
         """

        # set attributes
        self.v_alpha0 = v_alpha0
        self.v_alpha1 = v_alpha1
        self.draw_now = draw_now

        # validate inputs
        self._input_checks()

        # Update ax if axes are provided by the user.
        self._update_ax(ax)

        # Make sure matrix is a probability matrix
        self.df = validate_probability_mat(self.df)

        # Iterate over all positions and characters
        for p in self.ps:
            for c in self.cs:

                # Grab both glyph and message
                v = self.df.loc[p, c]
                g = self.glyph_df.loc[p, c]

                # Compute new alpha
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
        Modifies the properties of all glyphs below the x-axis in a logo.

        parameter
        ---------

        shade: (float)
            The amount of shading underneath x-axis. Range is [0,1]

        fade: (float)
            The amount of fading underneath x-axis .Range is [0,1]

        flip: (bool)
            If True, the glyph will be rendered flipped upside down.

        ax: (matplotlib Axes object)
            The axes object on which to draw the logo.

        draw_now: (bool)
            If True, the logo is rendered immediately after it is specified.
            Set to False if you wish to change the properties of any glyphs
            after initial specification, e.g. by running
            Logo.highlight_sequence().

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        # set attributes
        self.shade = shade
        self.fade = fade
        self.flip = flip
        self.draw_now = draw_now

        # Update ax if axes are provided by the user.
        self._update_ax(ax)

        # validate inputs
        self._input_checks()

        # the following two checks ensure that the attributes cs and ps exist,
        # this could throw an error in jupyter notebooks if a user ran this function
        # with an incorrectly run Logo object.

        check(hasattr(self, 'cs'), 'Characters entered into are None, please ensure'
                                   ' Logo ran correctly before running style_glyphs_below')

        check(hasattr(self, 'ps'), 'positions entered into are None, please ensure'
                                   ' Logo ran correctly before running style_glyphs_below')

        # Iterate over all positions and characters
        for p in self.ps:
            for c in self.cs:

                # If matrix message is < 0
                v = self.df.loc[p, c]
                if v < 0:

                    #  Get glyph
                    g = self.glyph_df.loc[p, c]

                    # Modify color and alpha
                    color = np.array(g.color) * (1.0 - shade)
                    alpha = g.alpha * (1.0 - fade)

                    # Set glyph attributes
                    g.set_attributes(color=color,
                                     alpha=alpha,
                                     flip=flip,
                                     **kwargs)

        # Draw now if requested
        if draw_now:
            self.draw()

    @handle_errors
    def style_single_glyph(self, p, c, draw_now=False, ax=None, **kwargs):
        """
        Modifies the properties of a component glyph in a logo.

        parameter
        ---------

        p: (number)
            Position of modified glyph. Must index a row in the matrix passed
            to the Logo constructor.

        c: (str)
            Character of modified glyph. Must index a column in the matrix
            passed to the Logo constructor.

        draw_now: (bool)
            Whether to readraw modified logo on current Axes.

        ax: (matplotlib Axes object)
            New axes, if any, on which to draw logo if draw_now=True.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        # set attributes
        self.p = p
        self.c = c
        self.draw_now = draw_now
        self.ax = ax

        # validate inputs
        self._input_checks()

        # Update ax if axes are provided by the user.
        self._update_ax(ax)

        check(self.p in self.glyph_df.index,'Error: p=%s is not a valid position' % p)
        check(self.c in self.glyph_df.columns,'Error: c=%s is not a valid character' % c)

        # Get glyph from glyph_df
        g = self.glyph_df.loc[p, c]
        g.set_attributes(**kwargs)

        # using true will draw the entire logo one glyph at a time.
        # causes big slow down. I don't if it's good to keep this call here.
        if draw_now:
            self.draw()

    @handle_errors
    def style_glyphs_in_sequence(self,
                                 sequence,
                                 draw_now=True,
                                 ax=None,
                                 **kwargs):
        """
        Highlights a specified sequence by changing the parameters of the
        glyphs at each corresponding position in that sequence. To use this,
        first run constructor with draw_now=False.

        parameters
        ----------
        sequence: (str)
            A string the same length as the logo, specifying which character
            to highlight at each position.

        draw_now: (bool)
            Whether to readraw modified logo on current Axes.

        ax: (matplotlib Axes object)
            New axes, if any, on which to draw logo if draw_now=True.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None
        """

        self.sequence = sequence
        self.draw_now = draw_now

        # validate input
        self._input_checks()

        # Update Axes if axes are provided by the user.
        self._update_ax(ax)

        check(len(self.sequence) == self.L,
              'Error: sequence to highlight does not have same length as logo.')

        # For each position in the logo...
        for i, p in enumerate(self.glyph_df.index):

            # Get character to highlight
            c = self.sequence[i]

            # Modify the glyph corresponding character c at position p
            # Only modify if c is a valid character. If not, ignore position
            if c in self.cs:
                self.style_single_glyph(p, c, **kwargs)

        # Draw now
        if draw_now:
            self.draw()

    @handle_errors
    def highlight_position(self, p, **kwargs):

        """
        ** Can only modify Axes that has already been set. **

        parameters
        ----------
        p: (int)
            Single position to highlight

        **kwargs:
            Other parameters to pass to highlight_position_range()

        returns
        -------
        None
        """

        # set attributes
        self.p = p

        # validate inputs
        self._input_checks()

        # If not yet drawn, draw
        if not self.has_been_drawn:
            self.draw()

        #assert self.has_been_drawn, \
        #    'Error: Cannot call this function until Log0 has been drawn.'

        self.highlight_position_range(pmin=p, pmax=p, **kwargs)

    @handle_errors
    def highlight_position_range(self, pmin, pmax,
                                 padding=0.0,
                                 color='yellow',
                                 edgecolor=None,
                                 floor=None,
                                 ceiling=None,
                                 zorder=-2,
                                 **kwargs):
        """
        Highlights multiple positions
        ** Can only modify Axes that has already been set. **

        parameters
        ----------
        pmin: (number)
            Lowest position to highlight.
            
        pmax: (number)
            Highest position to highlight.
            
        padding: (number >= -0.5)
            Amount of padding on either side of highlighted positions to add.
            
        color: (matplotlib color)
            Matplotlib color.
            
        floor: (number)
            Lower-most extent of highlight. If None, is set to Axes ymin.
            
        ceiling: (number)
            Upper-most extent of highlight. If None, is set to Axes ymax.
            
        zorder: (number)
            Placement of highlight rectangle in Axes z-stack.

        **kwargs:
            Other parmeters to pass to highlight_single_position

        returns
        -------
        None
        """

        # set attributes
        self.pmin = pmin
        self.pmax = pmax
        self.padding = padding
        self.color = color
        self.edgecolor = edgecolor
        self.floor = floor
        self.ceiling = ceiling
        self.zorder = zorder

        # validate inputs
        self._input_checks()

        if (hasattr(self, 'has_been_drawn')):
            check(self.has_been_drawn == True, 'Cannot call this function until Logo has been drawn.')

        #assert self.has_been_drawn, \
        #    'Error: Cannot call this function until Log0 has been drawn.'

        # If floor or ceiling have not been specified, using Axes ylims
        ymin, ymax = self.ax.get_ylim()
        if floor is None:
            floor = ymin
        if ceiling is None:
            ceiling = ymax

        #assert floor < ceiling, \
        #    'Error: floor < ceiling not satisfied.'
        check(floor < ceiling,'Error: floor < ceiling not satisfied.')


        # Set coordinates of rectangle
        #assert pmin <= pmax, \
        #    'Error: pmin <= pmax not satisfied.'
        check(pmin <= pmax, 'pmin <= pmax not satisfied.')

        #assert padding >= -0.5, \
        #    'Error: padding >= -0.5 not satisfied'
        check(padding >= -0.5,'Error: padding >= -0.5 not satisfied')

        x = pmin - .5 - padding
        y = floor
        width = pmax - pmin + 1 + 2*padding
        height = ceiling-floor

        # Draw rectangle
        patch = Rectangle(xy=(x, y),
                          width=width,
                          height=height,
                          facecolor=color,
                          edgecolor=edgecolor,
                          zorder=zorder,
                          **kwargs)
        self.ax.add_patch(patch)

    @handle_errors
    def draw_baseline(self,
                      zorder=-1,
                      color='black',
                      linewidth=0.5,
                      **kwargs):
        """
        Draws a line along the x-axis.
        ** Can only modify Axes that has already been set. **

        parameters
        ----------

        zorder: (number)
            The z-stacked location where the baseline is drawn

        color: (matplotlib color)
            Color to use for the baseline

        linewidth: (float >= 0)
            Width of the baseline

        **kwargs:
            Additional keyword arguments to be passed to ax.axhline()


        returns
        -------
        None
        """

        # set attributes
        self.zorder = zorder
        self.color = color
        self.linewidth = linewidth

        # validate inputs
        self._input_checks()

        if (hasattr(self, 'has_been_drawn')):
            check(self.has_been_drawn == True, 'Cannot call this function until Logo has been drawn.')

        #assert self.has_been_drawn, \
        #    'Error: Cannot call this function until Log0 has been drawn.'

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
        ** Can only modify Axes that has already been set. **

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

        # set attributes
        self.anchor = anchor
        self.spacing = spacing
        self.fmt = fmt
        self.rotation = rotation

        # validate input
        self._input_checks()

        if (hasattr(self, 'has_been_drawn')):
            check(self.has_been_drawn == True, 'Error: Cannot call this function until Logo has been drawn.')

        # Get list of positions, ps, that spans all those in matrix_df
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
        Turns spines on an off.
        ** Can only modify Axes that has already been set. **

        parameters
        ----------

        spines: (tuple of str)
            Specifies which of the four spines to modify. Default lists
            all possible entries.

        visible: (bool)
            Whether or not a spine is drawn.

        color: (matplotlib color)
            Spine color.

        linewidth: (float >= 0)
            Spine width.

        bounds: ([float, float])
            Specifies the upper- and lower-bounds of a spine.

        **kwargs:
            Additional keyword arguments to be passed to ax.axhline()

        returns
        -------
        None
        """

        # set attributes
        self.spines = spines
        self.visible = visible
        self.linewidth = linewidth
        self.color = color
        self.bounds = bounds

        # validate inputs
        self._input_checks()

        # Draw if logo has not yet been drawn
        if not self.has_been_drawn:
            self.draw()

        # Iterate over all spines
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
        Draws glyphs on the axes object 'ax' provided to the Logo
        constructor

        parameters
        ----------
        None

        returns
        -------
        None
        """

        # Update ax
        self._update_ax(ax)

        # If ax is still None, create figure
        if self.ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.ax = ax

        # Clear previous content from ax
        self.ax.clear()

        # Flag that this logo has not been drawn
        self.has_been_drawn = False

        # Draw each glyph
        for g in self.glyph_list:
            g.draw(self.ax)

        # Flag that this logo has indeed been drawn
        self.has_been_drawn = True

        # Draw baseline
        self.draw_baseline(linewidth=self.baseline_width)

        # Set xlims
        xmin = min([g.p - .5*g.width for g in self.glyph_list])
        xmax = max([g.p + .5*g.width for g in self.glyph_list])
        self.ax.set_xlim([xmin, xmax])

        # Set ylims
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
        vsep = self.vsep

        # For each position
        for p in self.ps:

            # Get values at this position
            vs = np.array(self.df.loc[p, :])

            # Sort values and corresponding characters as desired
            if self.stack_order == 'big_on_top':
                ordered_indices = np.argsort(vs)

            elif self.stack_order == 'small_on_top':
                tmp_vs = np.zeros(len(vs))
                indices = (vs != 0)
                tmp_vs[indices] = 1.0/vs[indices]

                ordered_indices = np.argsort(tmp_vs)
            elif self.stack_order == 'fixed':
                ordered_indices = np.array(range(len(vs)))[::-1]

            elif self.stack_order == 'flipped':
                ordered_indices = np.array(range(len(vs)))

            else:
                assert False, 'This should not be possible.'

            # Reorder values and characters
            vs = vs[ordered_indices]
            cs = [str(c) for c in self.cs[ordered_indices]]

            # Set floor
            floor = sum((vs - vsep) * (vs < 0)) + vsep/2.0

            # For each character
            for n, v, c in zip(range(self.C), vs, cs):

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
                              font_family=self.font_name,
                              **self.glyph_kwargs)

                # Add glyph to glyph_df
                glyph_df.loc[p, c] = glyph

                # Raise floor to current ceiling
                floor = ceiling + vsep

        # Set glyph_df attribute
        self.glyph_df = glyph_df
        self.glyph_list = [g for g in self.glyph_df.values.ravel()
                           if isinstance(g, Glyph)]

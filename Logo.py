import numpy as np
from logomaker import Glyph
from logomaker import Matrix
from logomaker import color as lm_color
import pandas as pd
import matplotlib.pyplot as plt
import pdb


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

    negate: (bool)
        If True, all values in matrix are multiplied by -1. This can be
        useful when illustrating negative energy values in an energy matrix.

    center: (bool)
        If True, the stack of characters at each position will be centered
        around zero. This is accomplished by subtracting the mean value
        in each row of the matrix from each element in that row.

    colors: (color scheme)
        Face color of logo characters. Default 'gray'. Here and in
        what follows a variable of type 'color' can take a variety of value
        types.
         - (str) A Logomaker color scheme in which the color is determined
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
             case the color will depend  on the character being drawn.
             E.g., {'A': 'green','C': [ 0.,  0.,  1.], 'G': 'y',
             'T': [ 1.,  0.,  0.,  0.5]}

    flip_below: (bool)
        If True, glyphs below the x-axis (which correspond to negative
        values in the matrix) will be flipped upside down.

    vsep: (float > 0)
        Amount of whitespace to leave between rendered glyphs. Unlike vpad,
        vsep is NOT relative to glyph height. The vsep-sized margin between
        glyphs on either side of the x-axis will always be centered on the
        x-axis.

    ax: (matplotlib Axes object)
        The axes object on which to draw the logo.

    draw_now: (bool)
        If True, the logo is rendered immediately after it is specified.
        Set to False if you wish to change the properties of any glyphs
        after initial specification, e.g. by running
        Logo.highlight_sequence().

    """

    def __init__(self,
                 matrix,
                 ax=None,
                 negate=False,
                 center=False,
                 colors=None,
                 flip_below=True,
                 vsep=0.0,
                 draw_now=True,
                 zorder=0,
                 ):

        # Set matrix_df. How to do this will depend on whether self.matrix
        # is a pd.DataFrame or a Matrix object.
        assert isinstance(matrix, (pd.DataFrame, Matrix.Matrix)), \
            'Error: matrix must be either a pd.DataFrame object ' +\
            'or a Matrix object'

        # If user passed a dataframe, convert to a Matrix object
        if isinstance(matrix, pd.DataFrame):
            self.matrix = Matrix.Matrix(matrix)
        else:
            self.matrix = matrix
        self.matrix_df = self.matrix.df

        # Compute length
        self.L = len(self.matrix_df)

        # Get list of characters
        self.cs = np.array([str(c) for c in self.matrix_df.columns])
        self.C = len(self.cs)

        # Get list of positions
        self.ps = np.array([float(p) for p in self.matrix_df.index])

        # Set colors by identifying default or otherwise setting to gray
        if colors is None:
            key = tuple(self.cs)
            colors = chars_to_colors_dict.get(key,'gray')
        self.colors = colors

        # Save other attributes
        self.ax = ax
        self.negate = bool(negate)
        self.center = center
        self.flip_below = flip_below
        self.vsep = vsep
        self.draw_now = draw_now
        self.zorder = zorder

        # Negate values if requested
        if self.negate:
            self.matrix_df = -self.matrix_df

        # Note: Logo does NOT expect df to change after it is passed
        # to the constructor. But one can change character attributes
        # before drawing.

        # Fill NaN values of matrix_df with zero
        if self.center:
            self.matrix_df.loc[:, :] = self.matrix_df.values - \
                self.matrix_df.values.mean(axis=1)[:, np.newaxis]

        # Compute color dictionary
        self.rgba_dict = lm_color.get_color_dict(
                                    color_scheme=self.colors,
                                    chars=self.cs,
                                    alpha=1)

        # Compute characters.
        self._compute_characters()

        # Draw now if requested
        if self.draw_now:
            self.draw()


    def style_glyphs(self, colors=None, **kwargs):

        # Reset colors if provided
        if colors is not None:
            self.colors = colors
            self.rgba_dict = lm_color.get_color_dict(
                                    color_scheme=self.colors,
                                    chars=self.cs,
                                    alpha=1)

        # Record zorder if this is provided
        if 'zorder' in kwargs.keys():
            self.zorder = kwargs['zorder']

        # Modify all glyphs
        for g in self.glyph_list:

            # Set each glyph attribute
            g.set_attributes(**kwargs)

            # If colors is not None, this should override
            if colors is not None:
                this_color = self.rgba_dict[g.c][:3]
                g.set_attributes(color=this_color)


    def add_below_effects(self,
                          shade=0.0,
                          fade=0.0,
                          flip=True):
        for p in self.ps:
            for c in self.cs:
                v = self.matrix_df.loc[p, c]
                if v < 0:
                    g = self.glyph_df.loc[p, c]
                    #pdb.set_trace()
                    g.color = np.array(g.color) * (1.0 - shade)
                    g.alpha = g.alpha * (1.0 - fade)
                    g.flip = flip


    def style_single_glyph(self, p, c, draw_now=True, **kwargs):
        """
        Modifies the properties of a component glyph in a logo. If using this,
        set draw_now=False in the Logo constructor.

        parameter
        ---------

        p: (number)
            Position of modified glyph. Must index a row in the matrix passed
            to the Logo constructor.

        c: (str)
            Character of modified glyph. Must index a column in the matrix
            passed to the Logo constructor.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None

        """

        assert p in self.glyph_df.index, \
            'Error: p=%s is not a valid position' % p

        assert c in self.glyph_df.columns, \
            'Error: c=%s is not a valid character' % c

        # Get glyph from glyph_df
        g = self.glyph_df.loc[p, c]
        g.set_attributes(**kwargs)

        # Draw now
        if draw_now:
            self.draw()


    def style_glyphs_in_sequence(self, sequence, draw_now=True, **kwargs):
        """
        Highlights a specified sequence by changing the parameters of the
        glyphs at each corresponding position in that sequence. To use this,
        first run constructor with draw_now=False.

        parameters
        ----------
        sequence: (str)
            A string the same length as the logo, specifying which character
            to highlight at each position.

        **kwargs:
            Keyword arguments to pass to Glyph.set_attributes()

        returns
        -------
        None

        """

        assert len(sequence) == self.L, \
            'Error: sequence to highlight does not have same length as logo.'

        # Make sure that all sequence characters are in self.cs
        for c in sequence:
            assert c in self.cs, \
                'Error: sequence contains invalid character %s' % c

        # For each position in the logo...
        for i, p in enumerate(self.glyph_df.index):

            # Get character to highlight
            c = sequence[i]

            # Modify the glyph corresponding character c at position p
            self.style_single_glyph(p, c, **kwargs)

        # Draw now
        if draw_now:
            self.draw()

    def highlight_column(self, ax=None,
                         color='yellow',
                         width=1.0,
                         alpha=1.0,
                         floor=None,
                         ceiling=None):
        pass;


    def draw_baseline(self,
                      ax=None,
                      zorder=-1,
                      color='black',
                      linewidth=0.5,
                      **kwargs):
        """
        Draws a line along the x-axis.

        parameters
        ----------

        ax: (matplotlib Axes object)
            The axes object on which to draw the logo

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

        # Update ax
        self._update_ax(ax)

        # Render baseline
        self.ax.axhline(zorder=zorder,
                        color=color,
                        linewidth=linewidth,
                        **kwargs)


    def style_xticks(self, ax=None,
                     anchor=0,
                     spacing=1,
                     fmt='%d',
                     rotation=0.0,
                     **kwargs):
        """
        Formats and styles tick marks along the x-axis.

        parameters
        ----------

        ax: (matplotlib Axes object)
            The axes object on which to draw the logo

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

        # Update ax
        self._update_ax(ax)

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


    def style_spines(self, ax=None,
                     spines=('top', 'bottom', 'left', 'right'),
                     visible=True,
                     linewidth=1.0,
                     color='black',
                     bounds=None):
        """
        Turns

        parameters
        ----------

        ax: (matplotlib Axes object)
            The axes object on which to draw the logo

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

        # Update ax
        self._update_ax(ax)

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

        # If user passed ax, use that
        if ax is not None:
            self.ax = ax

        # If ax is not set, set to gca
        if self.ax is None:
            self.ax = plt.gca()

        # Draw each glyph
        for g in self.glyph_list:
            g.draw(self.ax)

        # Set xlims
        xmin = min([g.p - .5*g.width for g in self.glyph_list])
        xmax = max([g.p + .5*g.width for g in self.glyph_list])
        self.ax.set_xlim([xmin, xmax])

        # Set ylims
        ymin = min([g.floor for g in self.glyph_list])
        ymax = max([g.ceiling for g in self.glyph_list])
        self.ax.set_ylim([ymin, ymax])


    def _update_ax(self, ax):
        # If user passed ax, use that
        if ax is not None:
            self.ax = ax

        # If ax is not set, set to gca
        if self.ax is None:
            self.ax = plt.gca()


    def _compute_characters(self):
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

            # Get sorted values and corresponding characters
            vs = np.array(self.matrix_df.loc[p, :])
            ordered_indices = np.argsort(vs)
            vs = vs[ordered_indices]
            cs = [str(c) for c in self.cs[ordered_indices]]

            # Set floor
            floor = sum((vs - vsep) * (vs < 0)) + vsep/2.0

            # For each character
            for n, v, c in zip(range(self.C), vs, cs):

                # Set ceiling
                ceiling = floor + abs(v)

                # Set color
                rgba = self.rgba_dict[c]
                this_color = rgba[:3]

                # Set whether to flip character
                flip = (v < 0 and self.flip_below)

                # Create glyph if height is finite
                glyph = Glyph.Glyph(p, c,
                                    ax=self.ax,
                                    floor=floor,
                                    ceiling=ceiling,
                                    color=this_color,
                                    flip=flip,
                                    draw_now=False,
                                    zorder=self.zorder)

                # Add glyph to glyph_df
                glyph_df.loc[p, c] = glyph

                # Raise floor to current ceiling
                floor = ceiling + vsep

        # Set glyph_df attribute
        self.glyph_df = glyph_df
        self.glyph_list = [g for g in self.glyph_df.values.ravel() if isinstance(g, Glyph.Glyph)]


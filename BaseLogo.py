import numpy as np
from logomaker import Glyph
from logomaker import color
import pandas as pd


class  BaseLogo:
    """
    BaseLogo represents a basic logo, drawn on a specified axes object
    using a specified matrix.

    attributes
    ----------

    ax: (matplotlib Axes object)
        The axes object on which to draw the logo.

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

    edgecolor: (matplotlib color)
        Color to use for edges of all glyphs in logo.

    edgewidth: (float > 0)
        Width to use for edges of all glyphs in logo.

    font_family: (str)
        The font name to use when rendering glyphs. Specifically, this is
        the value passed as the 'family' parameter when calling the
        FontProperties constructor. From matplotlib documentation:
        "family: A list of font names in decreasing order of priority.
        The items may include a generic font family name, either
        'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.
        In that case, the actual font to be used will be looked up from
        the associated rcParam in matplotlibrc."
        Run logomaker.get_font_families() for list of (some but not all) valid
        options on your system.

    font_weight: (str)
        The font weight to use when rendering glyphs. Specifically, this is
        the value passed as the 'weight' parameter in the FontProperties
        constructor. From matplotlib documentation: "weight: A numeric
        value in the range 0-1000 or one of 'ultralight', 'light',
        'normal', 'regular', 'book', 'medium', 'roman', 'semibold',
        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'."


    alpha: (float in [0,1])
        Opacity of rendered glyphs.

    flip_below: (bool)
        If True, glyphs below the x-axis (which correspond to negative
        values in the matrix) will be flipped upside down.

    shade_below: (float in [0,1])
        If True, glyphs below the x-axis (which correspond to negative
        values in the matrix) will made darker by multiplying color RGB
        values by 1.0-shade_below.

    fade_below:
        If True, glyphs below the x-axis (which correspond to negative
        values in the matrix) will made more transparenty by multiplying
        alpha by 1.0-fade_below.

    width:
        Width of each rendered glyph in position units.

    vpad: (float in [0,1])
        Amount of whitespace to leave within the bounds of a glyph above
        and below the rendered character. Specifically, in a glyph of
        height h, a margin of size h*vpad/2 will be left blank both above
        and below the rendered character.

    vsep: (float > 0)
        Amount of whitespace to leave between rendered glyphs. Unlike vpad,
        vsep is NOT relative to glyph height. The vsep-sized margin between
        glyphs on either side of the x-axis will always be centered on the
        x-axis.

    dont_stretch_more_than: (str)
        This parameter limits the amount that a character will be
        horizontally stretched when rendering a glyph. Specifying an
        wide character such as 'W' corresponds to less potential stretching,
        while specifying a narrow character such as '.' corresponds to more
        stretching. Note that the specified character does not have to be
        a character rendered in the logo.

    draw_now: (bool)
        If True, the logo is rendered immediately after it is specified.
        Set to False if you wish to change the properties of any glyphs
        after initial specification, e.g. by running
        BaseLogo.highlight_sequence().

    """

    def __init__(self,
                 ax,
                 matrix,
                 negate=False,
                 center=False,
                 colors='classic',
                 edgecolor='black',
                 edgewidth=0.0,
                 font_family='sans',
                 font_weight='bold',
                 alpha=1.0,
                 flip_below=True,
                 shade_below=0.0,
                 fade_below=0.0,
                 width=1.0,
                 vpad=0.0,
                 vsep=0.0,
                 dont_stretch_more_than='A',
                 draw_now=True,
                 ):

        # Save attributes
        self.ax = ax
        self.matrix = matrix
        self.negate = bool(negate)
        self.colors = colors
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth
        self.alpha = alpha
        self.font_family = font_family
        self.font_weight = font_weight
        self.center = center
        self.flip_below = flip_below
        self.shade_below = shade_below
        self.fade_below = fade_below
        self.width = width
        self.vpad = vpad
        self.vsep = vsep
        self.dont_stretch_more_than = dont_stretch_more_than
        self.draw_now = draw_now

        # Set matrix_df. How to do this will depend on whether self.matrix
        # is a pd.DataFrame or a Matrix object.
        #if isinstance(self.matrix, pd.DataFrame):
        self.matrix_df = self.matrix.fillna(0.0).astype(float)
        #elif isinstance(self.matrix, Matrix.Matrix):
        #    self.matrix_df = self.matrix.df.fillna(0.0).astype(float)

        # Negate values if requested
        if self.negate:
            self.matrix_df = -self.matrix_df

        # Note: BaseLogo does NOT expect df to change after it is passed
        # to the constructor. But one can change character attributes
        # before drawing.

        # Fill NaN values of matrix_df with zero
        if self.center:
            self.matrix_df.loc[:, :] = self.matrix_df.values - \
                self.matrix_df.values.mean(axis=1)[:, np.newaxis]

        # Compute length
        self.L = len(self.matrix_df)

        # Get list of characters
        self.cs = np.array([str(c) for c in self.matrix_df.columns])
        self.C = len(self.cs)

        # Get list of positions
        self.ps = np.array([float(p) for p in self.matrix_df.index])

        # Compute color dictionary
        self.rgba_dict = color.get_color_dict(
                                    color_scheme=self.colors,
                                    chars=self.cs,
                                    alpha=1)

        # Compute characters.
        self._compute_characters()

        # Draw now if requested
        if self.draw_now:
            self.draw()


    def modify_glyph(self, p, c, params_dict):
        """
        Modifies the properties of a component glyph in a logo. If using this,
        set draw_now=False in the BaseLogo constructor.

        parameter
        ---------

        p: (number)
            Position of modified glyph. Must index a row in the matrix passed
            to the BaseLogo constructor.

        c: (str)
            Character of modified glyph. Must index a column in the matrix
            passed to the BaseLogo constructor.

        params_dict:
            A dict object specifying new values for certain glyph parameters.
            E.g. params_dict={'color':'cyan', 'edgewidth':1.0} will render
            the modified glyph in cyan with an edgewidth of 1.0.
        """

        assert p in self.glyph_df.index, \
            'Error: p=%s is not a valid position' % p

        assert c in self.glyph_df.columns, \
            'Error: c=%s is not a valid character' % c

        # Get glyph from glyph_df
        g = self.glyph_df.loc[p, c]

        # Only modify if it is indeed a glyph; e.g. it might be None if
        # the corresponding matrix value is exactly zero.
        if isinstance(g, Glyph.Glyph):

            # Set each glyph attribute
            for key, value in params_dict.items():
                g.__dict__[key] = value


    def highlight_sequence(self, sequence, params_dict):
        """
        Highlights a specified sequence by changing the parameters of the
        glyphs at each corresponding position in that sequence. To use this,
        first run constructor with draw_now=False.

        parameters
        ----------
        sequence: (str)
            A string the same length as the logo, specifying which character
            to highlight at each position.

        params_dict:
            A dict object specifying the values of highlighted glyphs. E.g.
            params_dict={'color':'cyan', 'edgewidth':1.0} will render
            highlighted glyphs in cyan with an edgewidth of 1.0.

        return
        ------
        None

        """

        assert len(sequence) == self.L, \
            'Error: sequence to highlight does not have same length as logo.'
        assert isinstance(params_dict, dict), \
            'Error: params_dict is not a dict object.'

        # For each position in the logo...
        for i, p in enumerate(self.glyph_df.index):

            # Get character to highlight
            c = sequence[i]

            # Modify the glyph corresponding character c at position p
            self.modify_glyph(p, c, params_dict)


    def draw(self):
        """
        Draws glyphs on the axes object 'ax' provided to the BaseLogo
        constructor

        parameters
        ----------
        None.

        return
        ------
        None.

        """

        # Draw each glyph
        glyphs = [g for g in self.glyph_df.values.ravel() if isinstance(g, Glyph.Glyph)]
        for g in glyphs:
            g.draw()

        # Set xlims
        xmin = min([g.p - .5 * g.width for g in glyphs])
        xmax = max([g.p + .5 * g.width for g in glyphs])
        self.ax.set_xlim([xmin, xmax])

        # Set ylims
        ymin = min([g.floor for g in glyphs])
        ymax = max([g.ceiling for g in glyphs])
        self.ax.set_ylim([ymin, ymax])


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
                color = rgba[:3]
                alpha = self.alpha

                # Set whether to flip character
                flip = (v < 0 and self.flip_below)

                # Shade and fade character if requested
                if v < 0:
                    shade = self.shade_below
                    fade = self.fade_below
                    color = color * (1.0-shade)
                    alpha = alpha * (1.0-fade)

                # Create glyph if height is finite
                if ceiling > floor:
                    glyph = Glyph.Glyph(self.ax, p, c,
                                        floor=floor,
                                        ceiling=ceiling,
                                        width=self.width,
                                        vpad=self.vpad,
                                        font_family=self.font_family,
                                        font_weight=self.font_weight,
                                        color=color,
                                        alpha=alpha,
                                        edgecolor=self.edgecolor,
                                        edgewidth=self.edgewidth,
                                        dont_stretch_more_than=\
                                                self.dont_stretch_more_than,
                                        flip=flip,
                                        draw_now=False)

                # If height is not finite, just put None
                else:
                    glyph = None

                # Add glyph to glyph_df
                glyph_df.loc[p, c] = glyph

                # Raise floor to current ceiling
                floor = ceiling + vsep

        # Set glyph_df attribute
        self.glyph_df = glyph_df


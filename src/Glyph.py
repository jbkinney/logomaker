from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox
from matplotlib.font_manager import FontManager, FontProperties
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import pdb

# Create global font manager instance. This takes a second or two
font_manager = FontManager()


def list_font_families():
    """
    Returns a list of valid font_name options for use, e.g., in Glyph or
    Logo constructors.

    parameters
    ----------
    None.

    returns
    -------
    List of valid font_name names. This will vary from system to system.

    """
    fontnames_dict = dict([(f.name,f.fname) for f in font_manager.ttflist])
    fontnames = list(fontnames_dict.keys())
    fontnames.sort()
    return fontnames


class Glyph:
    """
    A Glyph represents a character, drawn on a specified axes at a specified
    position, rendered using specified styling such as color and font.

    attributes
    ----------

    ax: (matplotlib Axes object)
        The axes object on which to draw the glyph.

    p: (number)
        Axes x-coordinate of glyph center_values.

    c: (str)
        Character represeted by glyph.

    floor: (float)
        Axes y-coordinate of glyph bottom. Must be < ceiling

    ceiling: (float)
        Axes y-coordinate of glyph top. Must be > floor.

    width: (float > 0)
        Axes x-span of glyph.

    vpad: (float in [0,1])
        Amount of whitespace to leave within the glyph bounding box above
        and below the rendered glyph itself. Specifically, in a glyph with
        height h = ceiling-floor, a margin of size h*vpad/2 will be left blank
        both above and below the rendered character.

    font_family: (str)
        The font name to use when rendering the glyph. Specifically, this is
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
        The font weight to use when rendering the glyph. Specifically, this is
        the value passed as the 'weight' parameter in the FontProperties
        constructor. From matplotlib documentation: "weight: A numeric
        value in the range 0-1000 or one of 'ultralight', 'light',
        'normal', 'regular', 'book', 'medium', 'roman', 'semibold',
        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'."

    color: (matplotlib color)
        Color to use for the face of the glyph.

    edgecolor: (matplotlib color)
        Color to use for edges of the glyph.

    edgewidth: (float > 0)
        Width to use for edges of all glyphs in logo.

    dont_stretch_more_than: (str)
        This parameter limits the amount that a character will be
        horizontally stretched when rendering tue glyph. Specifying an
        wide character such as 'W' corresponds to less potential stretching,
        while specifying a narrow character such as '.' corresponds to more
        stretching.

    flip: (bool)
        If True, the glyph will be rendered flipped upside down.

    mirror: (bool)
        If True, a mirror image of the glyph will be rendered.

    zorder: (number)
        Placement of glyph within the Axes z-stack.

    alpha: (float in [0,1])
        Opacity of the rendered glyph.

    draw_now: (bool)
        If True, the glyph is rendered immediately after it is specified.
        Set to False if you might wish to change the properties of this glyph
        after initial specification.

    """

    def __init__(self,
                 p,
                 c,
                 ax=None,
                 floor=None,
                 ceiling=None,
                 width=0.95,
                 vpad=0.00,
                 font_family='sans',
                 font_weight='bold',
                 color='gray',
                 edgecolor='black',
                 edgewidth=0.0,
                 dont_stretch_more_than='E',
                 flip=False,
                 mirror=False,
                 zorder=None,
                 alpha=1,
                 draw_now=True):

        # Do basic checks
        assert width > 0, 'Error: Must have width > 0'

        # Set attributes
        self.p = p
        self.floor = floor
        self.ceiling = ceiling
        self.width = width
        self.vpad = vpad
        self.c = c
        self.flip = flip
        self.mirror = mirror
        self.zorder = zorder
        self.dont_stretch_more_than = dont_stretch_more_than
        self.alpha = alpha
        self.color = to_rgb(color)
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth
        self.font_family = font_family
        self.font_weight = font_weight
        self.ax = ax

        # Draw now if requested
        if draw_now:
            self.draw()

    def set_attributes(self, **kwargs):
        """
        Safe way to set the attributes of a Glyph object

        parameters
        ----------
        **kwargs:
            Attributes and their values.

        """
        for key, value in kwargs.items():
            if key in ('color', 'edgecolor'):
                value = to_rgb(value)
            self.__dict__[key] = value

    def draw(self, ax=None):
        """
        Draws Glyph given current parameters.

        parameters
        ----------

        ax: (matplotlib Axes object)
            The axes object on which to draw the Glyph.

        returns
        -------
        None.

        """

        # Make patch
        self.patch = self._make_patch()

        # If user passed ax, use that
        if ax is not None:
            self.ax = ax

        # If ax is not set, set to gca
        if self.ax is None:
            self.ax = plt.gca()

        # Draw character
        if self.patch is not None:
            self.ax.add_patch(self.patch)

    def _make_patch(self):
        """
        Makes patch corresponding to char. Does NOT
        add this patch to an axes object, though; that is done by draw().
        """

        # Set height
        height = self.ceiling - self.floor

        # If height is zero, just return none
        if height == 0.0:
            return None

        # Set xmin
        try:
            xmin = self.p - self.width / 2
        except:
            pdb.set_trace()

        # Compute vpad
        vpad = self.vpad * height

        # Create bounding box, leaving requested amount of padding
        bbox = Bbox.from_bounds(xmin,
                                self.floor+vpad/2,
                                self.width,
                                height-vpad)

        # Set font properties
        font_properties = FontProperties(family=self.font_family,
                                         weight=self.font_weight)

        # Create raw path
        tmp_path = TextPath((0, 0), self.c, size=1,
                            prop=font_properties)

        # Create path for max_stretched_character
        msc_path = TextPath((0, 0), self.dont_stretch_more_than, size=1,
                            prop=font_properties)

        # If need to flip char, do it within tmp_path
        if self.flip:
            transformation = Affine2D().scale(sx=1, sy=-1)
            tmp_path = transformation.transform_path(tmp_path)

        # If need to mirror char, do it within tmp_path
        if self.mirror:
            transformation = Affine2D().scale(sx=-11, sy=1)
            tmp_path = transformation.transform_path(tmp_path)

        # Get bounding box for temporary character and max_stretched_character
        tmp_bbox = tmp_path.get_extents()
        msc_bbox = msc_path.get_extents()

        # Compute horizontal stretch and shift needed to center_values char
        hstretch_tmp = bbox.width / tmp_bbox.width
        hstretch_msc = bbox.width / msc_bbox.width
        hstretch = min(hstretch_tmp, hstretch_msc)
        char_width = hstretch * tmp_bbox.width
        char_shift = (bbox.width - char_width) / 2.0

        # Compute vertical stretch
        vstretch = bbox.height / tmp_bbox.height

        # THIS IS THE KEY TRANSFORMATION
        # 1. Translate char path so that lower left corner is at origin
        # 2. Scale char path to desired width and height
        # 3. Translate char path to desired position
        transformation = Affine2D() \
            .translate(tx=-tmp_bbox.xmin, ty=-tmp_bbox.ymin) \
            .scale(sx=hstretch, sy=vstretch) \
            .translate(tx=bbox.xmin + char_shift, ty=bbox.ymin)
        char_path = transformation.transform_path(tmp_path)

        # Compute char patch
        patch = PathPatch(char_path,
                          facecolor=self.color,
                          zorder=self.zorder,
                          alpha=self.alpha,
                          edgecolor=self.edgecolor,
                          linewidth=self.edgewidth)

        # Return patch
        return patch

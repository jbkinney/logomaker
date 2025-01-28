# explicitly set a matplotlib backend if called from python to avoid the
# 'Python is not installed as a framework... error'
import sys
if sys.version_info[0] == 2:
    import matplotlib
    matplotlib.use('TkAgg')

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox
import matplotlib.font_manager as fm
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from logomaker.src.error_handling import check, handle_errors
from logomaker.src.colors import get_rgb
import numpy as np
from logomaker.src.validate import validate_numeric

# Create global list of valid font weights
VALID_FONT_WEIGHT_STRINGS = [
    'ultralight', 'light', 'normal', 'regular', 'book',
    'medium', 'roman', 'semibold', 'demibold', 'demi',
    'bold', 'heavy', 'extra bold', 'black']


def list_font_names():
    """
    Returns a list of valid font_name options for use in Glyph or
    Logo constructors.

    parameters
    ----------
    None.

    returns
    -------
    fontnames: (list)
        List of valid font_name names. This will vary from system to system.

    """
    fontnames_dict = dict([(f.name, f.fname) for f in fm.fontManager.ttflist])
    fontnames = list(fontnames_dict.keys())
    fontnames.append('sans')  # This always exists
    fontnames.sort()
    return fontnames


class Glyph:
    """
    A Glyph represents a character, drawn on a specified axes at a specified
    position, rendered using specified styling such as color and font_name.

    attributes
    ----------

    p: (number)
        x-coordinate value on which to center the Glyph.

    c: (str)
        The character represented by the Glyph.

    floor: (number)
        y-coordinate value where the bottom of the Glyph extends to.
        Must be < ceiling.

    ceiling: (number)
        y-coordinate value where the top of the Glyph extends to.
        Must be > floor.

    ax: (matplotlib Axes object)
        The axes object on which to draw the Glyph.

    width: (number > 0)
        x-coordinate span of the Glyph.

    vpad: (number in [0,1))
        Amount of whitespace to leave within the Glyph bounding box above
        and below the actual Glyph. Specifically, in a glyph with
        height h = ceiling-floor, a margin of size h*vpad/2 will be left blank
        both above and below the rendered character.

    font_name: (str)
        The name of the font to use when rendering the Glyph. This is
        the value passed as the 'family' parameter when calling the
        matplotlib.font_manager.FontProperties constructor.

    font_weight: (str or number)
        The font weight to use when rendering the Glyph. Specifically, this is
        the value passed as the 'weight' parameter in the
        matplotlib.font_manager.FontProperties constructor.
        From matplotlib documentation: "weight: A numeric
        value in the range 0-1000 or one of 'ultralight', 'light',
        'normal', 'regular', 'book', 'medium', 'roman', 'semibold',
        'demibold', 'demi', 'bold', 'heavy', 'extra bold', 'black'."

    color: (matplotlib color)
        Color to use for Glyph face.

    edgecolor: (matplotlib color)
        Color to use for Glyph edge.

    edgewidth: (number >= 0)
        Width of Glyph edge.

    dont_stretch_more_than: (str)
        This parameter limits the amount that a character will be
        horizontally stretched when rendering the Glyph. Specifying a
        wide character such as 'W' corresponds to less potential stretching,
        while specifying a narrow character such as '.' corresponds to more
        stretching.

    flip: (bool)
        If True, the Glyph will be rendered upside down.

    mirror: (bool)
        If True, a mirror image of the Glyph will be rendered.

    zorder: (number)
        Placement of Glyph within the z-stack of ax.

    alpha: (number in [0,1])
        Opacity of the rendered Glyph.

    figsize: ([float, float]):
        The default figure size for the rendered glyph; only used if ax is
        not supplied by the user.
    """

    @handle_errors
    def __init__(self,
                 p,
                 c,
                 floor,
                 ceiling,
                 ax=None,
                 width=0.95,
                 vpad=0.00,
                 font_name='sans',
                 font_weight='bold',
                 color='gray',
                 edgecolor='black',
                 edgewidth=0.0,
                 dont_stretch_more_than='E',
                 flip=False,
                 mirror=False,
                 zorder=None,
                 alpha=1,
                 figsize=(1, 1)):

        # Set attributes
        self.p = p
        self.c = c
        self.floor = floor
        self.ceiling = ceiling
        self.ax = ax
        self.width = width
        self.vpad = vpad
        self.flip = flip
        self.mirror = mirror
        self.zorder = zorder
        self.dont_stretch_more_than = dont_stretch_more_than
        self.alpha = alpha
        self.color = color
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth
        self.font_name = font_name
        self.font_weight = font_weight
        self.figsize = figsize

        # Check inputs
        self._input_checks()

        # If ax is not set, set to current axes object
        if self.ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.ax = ax

        # Make patch
        self._make_patch()

    def set_attributes(self, **kwargs):
        """
        Safe way to set the attributes of a Glyph object

        parameters
        ----------
        **kwargs:
            Attributes and their values.
        """

        # remove drawn patch
        if (self.patch is not None) and (self.patch.axes is not None):
            self.patch.remove()

        # set each attribute passed by user
        for key, value in kwargs.items():

            # if key corresponds to a color, convert to rgb
            if key in ('color', 'edgecolor'):
                value = to_rgb(value)

            # save variable name
            self.__dict__[key] = value

        # remake patch
        self._make_patch()

    def draw(self):
        """
        Draws Glyph given current parameters.

        parameters
        ----------
        None.

        returns
        -------
        None.
        """

        # Draw character
        if self.patch is not None:
            self.ax.add_patch(self.patch)

    def _make_patch(self):
        """
        Returns an appropriately scaled patch object corresponding to
        the Glyph.
        """

        # Set height
        height = self.ceiling - self.floor

        # If height is zero, set patch to None and return None
        if height == 0.0:
            self.patch = None
            return None

        # Set bounding box for character,
        # leaving requested amount of padding above and below the character
        char_xmin = self.p - self.width / 2.0
        char_ymin = self.floor + self.vpad * height / 2.0
        char_width = self.width
        char_height = height - self.vpad * height
        bbox = Bbox.from_bounds(char_xmin,
                                char_ymin,
                                char_width,
                                char_height)

        # Set font properties of Glyph
        font_properties = fm.FontProperties(family=self.font_name,
                                            weight=self.font_weight)

        # Create a path for Glyph that does not yet have the correct
        # position or scaling
        tmp_path = TextPath((0, 0), self.c, size=1,
                            prop=font_properties)

        # Create create a corresponding path for a glyph representing
        # the max stretched character
        msc_path = TextPath((0, 0), self.dont_stretch_more_than, size=1,
                            prop=font_properties)

        # If need to flip char, do it within tmp_path
        if self.flip:
            transformation = Affine2D().scale(sx=1, sy=-1)
            tmp_path = transformation.transform_path(tmp_path)

        # If need to mirror char, do it within tmp_path
        if self.mirror:
            transformation = Affine2D().scale(sx=-1, sy=1)
            tmp_path = transformation.transform_path(tmp_path)

        # Get bounding box for temporary character and max_stretched_character
        tmp_bbox = tmp_path.get_extents()
        msc_bbox = msc_path.get_extents()

        # Compute horizontal stretch factor needed for tmp_path
        hstretch_tmp = bbox.width / tmp_bbox.width

        # Compute horizontal stretch factor needed for msc_path
        hstretch_msc = bbox.width / msc_bbox.width

        # Choose the MINIMUM of these two horizontal stretch factors.
        # This prevents very narrow characters, such as 'I', from being
        # stretched too much.
        hstretch = min(hstretch_tmp, hstretch_msc)

        # Compute the new character width, accounting for the
        # limit placed on the stretching factor
        char_width = hstretch * tmp_bbox.width

        # Compute how much to horizontally shift the character path
        char_shift = (bbox.width - char_width) / 2.0

        # Compute vertical stetch factor needed for tmp_path
        vstretch = bbox.height / tmp_bbox.height

        # THESE ARE THE ESSENTIAL TRANSFORMATIONS
        # 1. First, translate char path so that lower left corner is at origin
        # 2. Then scale char path to desired width and height
        # 3. Finally, translate char path to desired position
        # char_path is the resulting path used for the Glyph
        transformation = Affine2D() \
            .translate(tx=-tmp_bbox.xmin, ty=-tmp_bbox.ymin) \
            .scale(sx=hstretch, sy=vstretch) \
            .translate(tx=bbox.xmin + char_shift, ty=bbox.ymin)
        char_path = transformation.transform_path(tmp_path)

        # Convert char_path to a patch, which can now be drawn on demand
        self.patch = PathPatch(char_path,
                               facecolor=self.color,
                               zorder=self.zorder,
                               alpha=self.alpha,
                               edgecolor=self.edgecolor,
                               linewidth=self.edgewidth)

        # add patch to axes
        self.ax.add_patch(self.patch)

    def _input_checks(self):
        """
        check input parameters in the Logo constructor for correctness
        """
        # validate p
        self.p = validate_numeric(self.p, 'p')

        # check c is of type str
        check(isinstance(self.c, str),
              'type(c) = %s; must be of type str ' %
              type(self.c))

        # validate floor and ceiling
        self.floor = validate_numeric(self.floor, 'floor')
        self.ceiling = validate_numeric(self.ceiling, 'ceiling')

        # check floor <= ceiling
        check(self.floor <= self.ceiling,
              'must have floor <= ceiling. Currently, '
              'floor=%f, ceiling=%f' % (self.floor, self.ceiling))

        # check ax
        check((self.ax is None) or isinstance(self.ax, Axes),
              'ax must be either a matplotlib Axes object or None.')

        # validate width
        self.width = validate_numeric(self.width, 'width', min_val=0.0)

        # validate vpad
        self.vpad = validate_numeric(self.vpad, 'vpad', min_val=0.0, max_val=1.0, min_inclusive=True, max_inclusive=False)

        # validate font_name
        check(isinstance(self.font_name, str),
              'type(font_name) = %s must be of type str' % type(self.font_name))

        # check font_weight
        check(isinstance(self.font_weight, (str, int)),
              'type(font_weight) = %s should either be a string or an int' %
              (type(self.font_weight)))
        if isinstance(self.font_weight, str):
            check(self.font_weight in VALID_FONT_WEIGHT_STRINGS,
                  'font_weight must be one of %s' % VALID_FONT_WEIGHT_STRINGS)
        elif isinstance(self.font_weight, int):
            check(0 <= self.font_weight <= 1000,
                  'font_weight must be in range [0,1000]')

        # check color safely
        self.color = get_rgb(self.color)

        # validate edgecolor safely
        self.edgecolor = get_rgb(self.edgecolor)

        # Check that edgewidth is a number
        self.edgewidth = validate_numeric(self.edgewidth, 'edgewidth', min_val=0.0)

        # check dont_stretch_more_than is of type str
        check(isinstance(self.dont_stretch_more_than, str),
              'type(dont_stretch_more_than) = %s; must be of type str ' %
              type(self.dont_stretch_more_than))

        # check that dont_stretch_more_than is a single character
        check(len(self.dont_stretch_more_than)==1,
              'dont_stretch_more_than must have length 1; '
              'currently len(dont_stretch_more_than)=%d' %
              len(self.dont_stretch_more_than))

        # check that flip is a boolean
        check(isinstance(self.flip, (bool, np.bool_)),
              'type(flip) = %s; must be of type bool ' % type(self.flip))
        self.flip = bool(self.flip)

        # check that mirror is a boolean
        check(isinstance(self.mirror, (bool, np.bool_)),
              'type(mirror) = %s; must be of type bool ' % type(self.mirror))
        self.mirror = bool(self.mirror)

        # validate zorder
        if self.zorder is not None:
            self.zorder = validate_numeric(self.zorder, 'zorder')

        # Check alpha is a number
        self.alpha = validate_numeric(self.alpha, 'alpha', min_val=0.0, max_val=1.0)

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



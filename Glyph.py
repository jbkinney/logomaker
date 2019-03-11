from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D, Bbox
from matplotlib.font_manager import FontManager, FontProperties

# Create global font manager instance. This takes a second or two
font_manager = FontManager()

def get_fontnames_dict():
    ttf_dict = dict([(f.name,f.fname) for f in font_manager.ttflist])
    return ttf_dict

def get_font_families():
    fontnames_dict = get_fontnames_dict()
    fontnames = list(fontnames_dict.keys())
    fontnames.sort()
    return fontnames


class Glyph:
    """
    A Glyph represents to a character, drawn on a specified axes at a specified
    position, rendered using specified styling such as color and font.

    attributes
    ----------

    ax:

    p:

    c:

    floor:

    ceiling:

    width:

    vpad:

    font_family:

    font_weight:

    color:

    edgewidth:

    edgecolor:

    dont_stretch_more_than:

    flip:

    mirror:

    zorder:

    alpha:
    """

    def __init__(self,
                 ax,
                 p,
                 c,
                 floor=None,
                 ceiling=None,
                 width=1,
                 vpad=0.0,
                 font_family='sans',
                 font_weight='bold',
                 color='gray',
                 edgecolor='black',
                 edgewidth=0.0,
                 dont_stretch_more_than='A',
                 flip=False,
                 mirror=False,
                 zorder=None,
                 alpha=1,
                 draw_now=True):

        # Do basic checks
        assert width > 0, 'Error: Must have width > 0'

        # Set floor and ceiling to axes ylim value if None is passed for either
        ax_ymin, ax_ymax = ax.get_ylim()
        if floor is None:
            floor = ax_ymin
        if ceiling is None:
            ceiling = ax_ymax
        assert ceiling > floor, 'Error: Must have ceiling > floor'

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
        self.color = color
        self.edgecolor = edgecolor
        self.edgewidth = edgewidth
        self.font_family = font_family
        self.font_weight = font_weight
        self.ax = ax

        # Draw now if requested
        if draw_now:
            self.draw()

    def draw(self):
        '''
        Draws Glyph given current parameters except for the following,
        which are ignored after the constructor is called: xmin, xmax, height.
        '''

        # Make patch
        self.patch = self._make_patch()

        # Draw character
        self.ax.add_patch(self.patch)


    def _make_patch(self):
        '''
        Makes patch corresponding to char. Does NOT yet
        add patch to an axes object, though
        '''

        # Set xmin and height
        xmin = self.p - self.width / 2
        height = self.ceiling - self.floor

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

        # Compute horizontal stretch and shift needed to center char
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
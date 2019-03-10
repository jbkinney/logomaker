from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.transforms import Affine2D, Bbox
from matplotlib.font_manager import FontManager, FontProperties

import numpy as np
import pdb

# Create global font manager instance. This takes a second or two
font_manager = FontManager()

def get_fontnames_dict():
    ttf_dict = dict([(f.name,f.fname) for f in font_manager.ttflist])
    return ttf_dict

def get_fontnames():
    fontnames_dict = get_fontnames_dict()
    fontnames = list(fontnames_dict.keys())
    fontnames.sort()
    return fontnames


class Glyph:
    '''
    Draws a glyph with specified attributes. Requires axes object to run.

    Tips:
    font_family='Arial Rounded MT Bold'
    '''

    def __init__(self,
                 ax,
                 p,
                 c,
                 floor=None,
                 ceiling=None,
                 width=1,
                 font_family='sans',
                 font_weight='bold',
                 color='black',
                 max_horizontal_stretch=None,
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
        self.c = c
        self.flip = flip
        self.mirror = mirror
        self.zorder = zorder
        self.max_hstretch = max_horizontal_stretch
        self.alpha = alpha
        self.color = color
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

        # Create bounding box
        bbox = Bbox.from_bounds(xmin,
                                self.floor,
                                self.width,
                                height)

        # Set font properties
        font_properties = FontProperties(family=self.font_family,
                                         weight=self.font_weight)

        # Create raw path
        tmp_path = TextPath((0, 0), self.c, size=1,
                            prop=font_properties)

        # If need to flip char, do it within tmp_path
        if self.flip:
            transformation = Affine2D().scale(sx=1, sy=-1)
            tmp_path = transformation.transform_path(tmp_path)

        # If need to mirror char, do it within tmp_path
        if self.mirror:
            transformation = Affine2D().scale(sx=-11, sy=1)
            tmp_path = transformation.transform_path(tmp_path)

        # Get bounding box for temporary char path
        tmp_bbox = tmp_path.get_extents()

        # Compute horizontal stretch and shift needed to center char
        hstretch = bbox.width / tmp_bbox.width
        if self.max_hstretch is not None:
            hstretch = min(hstretch, self.max_hstretch)
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
                          edgecolor=None,
                          linewidth=0)

        # Return patch
        return patch


#######################################################################
# Code before 19.03.19 below

class Character:
    def __init__(self,
                 c,
                 xmin,
                 ymin,
                 width,
                 height,
                 font_properties,
                 flip,
                 facecolor,
                 edgecolor,
                 linewidth,
                 boxcolor,
                 boxedgecolor,
                 boxedgewidth,
                 hpad,
                 vpad,
                 max_hstretch,
                 zorder=None):
        assert width > 0
        assert height > 0

        self.c = c
        self.bbox = Bbox.from_bounds(xmin, ymin, width, height)
        self.font_properties = font_properties
        self.flip = flip

        # Set tranparency
        self.facecolor=facecolor
        self.edgecolor=edgecolor
        self.linewidth=linewidth
        self.boxcolor=boxcolor
        self.boxedgecolor=boxedgecolor
        self.boxedgewidth=boxedgewidth
        self.hpad = hpad
        self.vpad = vpad
        self.max_hstretch = max_hstretch
        self.zorder = zorder


    def draw(self, ax):
        '''
        Draws the char on a specified axes
        :param ax: matplotlib axes object
        :return: None
        '''
        # Calculate char & box patches
        put_char_in_box(ax=ax,
                        char=self.c,
                        bbox=self.bbox,
                        flip=self.flip,
                        facecolor=self.facecolor,
                        edgecolor=self.edgecolor,
                        linewidth=self.linewidth,
                        font_properties=self.font_properties,
                        boxcolor=self.boxcolor,
                        boxedgecolor=self.boxedgecolor,
                        boxedgewidth=self.boxedgewidth,
                        hpad=self.hpad,
                        vpad=self.vpad,
                        max_hstretch=self.max_hstretch,
                        zorder=self.zorder)


def put_char_in_box(ax,
                    char,
                    bbox,
                    facecolor,
                    edgecolor,
                    linewidth,
                    boxcolor,
                    boxedgecolor,
                    boxedgewidth,
                    font_properties,
                    flip,
                    hpad,
                    vpad,
                    max_hstretch,
                    zorder):

    # Create raw path
    tmp_path = TextPath((0, 0), char, size=1, prop=font_properties)

    # If need to flip char, do it within tmp_path
    if flip:
        transformation = Affine2D().scale(sx=1, sy=-1)
        tmp_path = transformation.transform_path(tmp_path)

    # Get bounding box for temporary char path
    tmp_bbox = tmp_path.get_extents()

    # Redefine tmp_bbox with padding as requested
    x0 = tmp_bbox.xmin - .5*hpad*tmp_bbox.width
    x1 = tmp_bbox.xmax + .5*hpad*tmp_bbox.width
    y0 = tmp_bbox.ymin - .5*vpad*tmp_bbox.height
    y1 = tmp_bbox.ymax + .5*vpad*tmp_bbox.height
    tmp_bbox = Bbox([[x0, y0], [x1, y1]])

    # Compute horizontal stretch and shift needed to center char
    hstretch = min(bbox.width / tmp_bbox.width, max_hstretch)
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
        .translate(tx=bbox.xmin+char_shift, ty=bbox.ymin)
    char_path = transformation.transform_path(tmp_path)

    # Compute patch for box containing char
    box_patch = Rectangle((bbox.xmin, bbox.ymin),
                          bbox.width, bbox.height,
                          facecolor=boxcolor,
                          edgecolor=boxedgecolor,
                          linewidth=boxedgewidth,
                          zorder=zorder)
    ax.add_patch(box_patch)

    # Compute char patch
    char_patch = PathPatch(char_path,
                           facecolor=facecolor,
                           edgecolor=edgecolor,
                           linewidth=linewidth,
                           zorder=zorder)
    ax.add_patch(char_patch)

    # Return patches to user
    return char_patch, box_patch

# Returns hstretch values for the individual characters
def get_stretch_vals(chars, width=1, height=1, font_properties=None,
                     hpad=0, vpad=0):

    # Stores stretch output 
    hstretch_dict = {}
    vstretch_dict = {}

    # Comptue hstretch and vstretch for each char
    for char in chars:

        # Create raw path
        tmp_path = TextPath((0, 0), char, size=1, prop=font_properties)

        # Get bounding box for temporary char path
        tmp_bbox = tmp_path.get_extents()

        # Redefine tmp_bbox with padding as requested
        x0 = tmp_bbox.xmin - .5 * hpad * tmp_bbox.width
        x1 = tmp_bbox.xmax + .5 * hpad * tmp_bbox.width
        y0 = tmp_bbox.ymin - .5 * vpad * tmp_bbox.height
        y1 = tmp_bbox.ymax + .5 * vpad * tmp_bbox.height
        tmp_bbox = Bbox([[x0, y0], [x1, y1]])

        # Compute horizontal and vertical stretch values
        hstretch = width / tmp_bbox.width
        vstretch = height / tmp_bbox.height

        # Store stretch values
        hstretch_dict[char] = hstretch
        vstretch_dict[char] = vstretch

    return hstretch_dict, vstretch_dict
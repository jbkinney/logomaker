from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.transforms import Affine2D, Bbox

import numpy as np
import pdb

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
                 max_hstretch):
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


    def draw(self, ax):
        '''
        Draws the character on a specified axes
        :param ax: matplotlib axes object
        :return: None
        '''
        # Calculate character & box patches
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
                        max_hstretch=self.max_hstretch)


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
                    max_hstretch):

    # Create raw path
    tmp_path = TextPath((0, 0), char, size=1, prop=font_properties)

    # If need to flip character, do it within tmp_path
    if flip:
        transformation = Affine2D().scale(sx=1, sy=-1)
        tmp_path = transformation.transform_path(tmp_path)

    # Get bounding box for temporary character path
    tmp_bbox = tmp_path.get_extents()

    # Redefine tmp_bbox with padding as requested
    x0 = tmp_bbox.xmin - .5*hpad*tmp_bbox.width
    x1 = tmp_bbox.xmax + .5*hpad*tmp_bbox.width
    y0 = tmp_bbox.ymin - .5*vpad*tmp_bbox.height
    y1 = tmp_bbox.ymax + .5*vpad*tmp_bbox.height
    tmp_bbox = Bbox([[x0, y0], [x1, y1]])

    # Compute horizontal stretch and shift needed to center character
    hstretch = min(bbox.width / tmp_bbox.width, max_hstretch)
    char_width = hstretch * tmp_bbox.width
    char_shift = (bbox.width - char_width) / 2.0

    # Compute vertical stretch
    vstretch = bbox.height / tmp_bbox.height

    # THIS IS THE KEY TRANSFORMATION
    # 1. Translate character path so that lower left corner is at origin
    # 2. Scale character path to desired width and height
    # 3. Translate character path to desired position
    transformation = Affine2D() \
        .translate(tx=-tmp_bbox.xmin, ty=-tmp_bbox.ymin) \
        .scale(sx=hstretch, sy=vstretch) \
        .translate(tx=bbox.xmin+char_shift, ty=bbox.ymin)
    char_path = transformation.transform_path(tmp_path)

    # Compute patch for box containing character
    box_patch = Rectangle((bbox.xmin, bbox.ymin),
                          bbox.width, bbox.height,
                          facecolor=boxcolor,
                          edgecolor=boxedgecolor,
                          linewidth=boxedgewidth,
                          zorder=-3)
    ax.add_patch(box_patch)

    # Compute character patch
    char_patch = PathPatch(char_path,
                           facecolor=facecolor,
                           edgecolor=edgecolor,
                           linewidth=linewidth,
                           zorder=3)
    ax.add_patch(char_patch)

    # Return patches to user
    return char_patch, box_patch

# Returns hstretch values for the individual characters
def get_stretch_vals(chars, width=1, height=1, font_properties=None,
                     hpad=0, vpad=0):

    # Stores stretch output 
    hstretch_dict = {}
    vstretch_dict = {}

    # Comptue hstretch and vstretch for each character
    for char in chars:

        # Create raw path
        tmp_path = TextPath((0, 0), char, size=1, prop=font_properties)

        # Get bounding box for temporary character path
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
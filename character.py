from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties 
import numpy as np
import os
import pdb

from utils import Box



class Character:
    def __init__(self, c, x, y, w, h, color,
                 font_properties=None,
                 flip=False,
                 shade=1,
                 alpha=1, edgecolor='none'):
        assert w > 0
        assert h > 0

        self.c = c
        self.box = Box(x, x + w, y, y + h)
        self.font_properties = font_properties
        self.flip = flip

        # Set color
        try:
            self.color = np.array(to_rgba(color)) * \
                         np.array([shade, shade, shade, 1])
        except:
            assert False, 'Error! Unable to interpret color %s' % repr(color)

        # Set tranparency
        self.color[3] = alpha
        self.edgecolor=edgecolor

    def draw(self, ax):

        # Define character bounding box
        bbox = list(self.box.bounds)
        if self.flip:
            bbox[2:] = bbox[2:][::-1]

        # Draw character
        put_char_in_box(ax, self.c, bbox, \
                        facecolor=self.color,
                        edgecolor=self.edgecolor,
                        font_properties=self.font_properties)



def put_char_in_box(ax, char, bbox, facecolor='k', \
                    edgecolor='none', 
                    font_properties=None, 
                    zorder=0, \
                    font_weight='bold', font_style='normal'):

    # Create raw path
    path = TextPath((0, 0), char, size=1, prop=font_properties)

    # Get params from bounding box
    try:
        set_xlb, set_xub, set_ylb, set_yub = tuple(bbox)
    except:
        pdb.set_trace()
    set_w = set_xub - set_xlb
    set_h = set_yub - set_ylb

    # Rescale path vertices to fit in bounding box
    num_vertices = len(path._vertices)
    raw_xs = [x[0] for x in path._vertices]
    raw_ys = [x[1] for x in path._vertices]
    raw_xlb = min(raw_xs)
    raw_ylb = min(raw_ys)
    raw_w = max(raw_xs) - raw_xlb
    raw_h = max(raw_ys) - raw_ylb
    set_xs = set_xlb + (raw_xs - raw_xlb) * (set_w / raw_w)
    set_ys = set_ylb + (raw_ys - raw_ylb) * (set_h / raw_h)

    # Reset vertices of path
    path._vertices = [100 * np.array([set_xs[i], set_ys[i]]) for i in range(num_vertices)]

    # Make and display patch
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                      zorder=zorder)
    ax.add_patch(patch)
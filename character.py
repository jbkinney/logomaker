from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties, findSystemFonts
import numpy as np
import os
import pdb

from utils import Box

DEFAULT_FONT = 'Arial Rounded Bold'

# Build font_file_dict
font_file_list = \
    findSystemFonts(fontpaths=None, fontext='ttf')

# Build FONT_FILE_DICT
FONT_FILE_DICT = {}
for font_file in font_file_list:
    base_name = os.path.basename(font_file)
    font_name = os.path.splitext(base_name)[0]

    # Test whether font name is a string
    if isinstance(font_name, (str, unicode)):
        FONT_FILE_DICT[str(font_name)] = font_file

# Get list of font names and sort
FONTS = list(FONT_FILE_DICT.keys())
FONTS.sort()

class Character:
    def __init__(self, c, x, y, w, h, color,
                 font_name=DEFAULT_FONT,
                 flip=False,
                 shade=1,
                 alpha=1, edgecolor='none'):
        assert w > 0
        assert h > 0

        self.c = c
        self.box = Box(x, x + w, y, y + h)
        self.font_name = font_name
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
                        font_name=self.font_name)

def get_fonts():
    return FONTS

def put_char_in_box(ax, char, bbox, facecolor='k', \
                    edgecolor='none', font_name=None, zorder=0):
    # Get default font properties if none specified
    if font_name is None:
        font_properties = FontProperties(family='sans-serif', weight='bold')
    elif font_name in FONT_FILE_DICT:
        font_file = FONT_FILE_DICT[font_name]
        font_properties = FontProperties(fname=font_file)
    else:
        assert False, 'Error: unable to interpret font name %s' % font_name

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
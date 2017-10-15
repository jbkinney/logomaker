from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties, findSystemFonts, findfont
import numpy as np
import os
import pdb

from utils import Box

# Build dictionary mapping font names to font files
font_files = findSystemFonts()
FONT_FILE_DICT = {}
for font_file in font_files:
    try:
        font_name = str(FontProperties(fname=font_file).get_name())
    except:
        print 'Failed on %s'%font_file
        continue
    FONT_FILE_DICT[font_name] = font_file

# Specify default font and add to dictionary
default_font_file = findfont('bold')
DEFAULT_FONT = str(FontProperties(fname=default_font_file).get_name())
FONT_FILE_DICT[DEFAULT_FONT] = default_font_file

# Get list of font names and sort
FONTS = list(FONT_FILE_DICT.keys())
FONTS.sort()

def get_default_font():
    ''' Returns the default chosen font'''
    return DEFAULT_FONT

def get_fonts():
    ''' Returns a list of all available fonts'''
    return FONTS

class Character:
    def __init__(self, c, x, y, w, h, color,
                 font_name=None,
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


def validate_font(font_name):
    # Get default font properties if none specified
    if font_name is None:
        font_name = DEFAULT_FONT 

    # If font is not in FONT_FILE_DICT, revert to default
    elif not font_name in  FONT_FILE_DICT:
        print 'Warning: unrecognized font %s; using default font %s.'%\
            (font_name, DEFAULT_FONT)
        font_name = DEFAULT_FONT
    return font_name    

def put_char_in_box(ax, char, bbox, facecolor='k', \
                    edgecolor='none', font_name=None, zorder=0, \
                    font_weight='bold', font_style='normal'):

    # Specify font properties from font name
    font_name = validate_font(font_name)
    font_file = FONT_FILE_DICT[font_name]
    font_properties = FontProperties(fname=font_file, 
        style=font_style, weight=font_weight)

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
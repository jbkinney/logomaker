from __future__ import division
import numpy as np
import pandas as pd
import ast
import inspect
import re
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, FontManager
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgba
import matplotlib as mpl
import pdb

from validate import validate_parameter, validate_mat

from data import load_alignment
from Logo import Logo
from make_logo import make_logo
from make_styled_logo import make_styled_logo

def get_fontnames():
    """ Return a list of available font names. """
    font_names = [f.name for f in font_manager.ttflist] + \
                 [f.name for f in font_manager.afmlist]
    font_names = list(set(font_names))
    font_names.sort()
    return font_names




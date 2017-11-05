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

import validate
from validate import validate_parameter, validate_mat

from data import load_alignment
from Logo import Logo
from make_logo import make_logo
from character import get_fontnames_dict, get_fontnames
from make_styled_logo import make_styled_logo

from load_meme import load_meme
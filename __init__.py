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
from functools import wraps
import sys
import os

# method that helps complete the implementation of
# Controlled Error for logomaker.
def logomaker_excepthook(type, value, traceback):
    print(value)


# this message prints out a message
# and stops execution and hides traceback.
def ControlledError(message):
        sys.excepthook = logomaker_excepthook
        raise Exception(message)


def check(condition, message):
    '''
    Checks a condition; raises a ControlledError with message if condition fails.
    :param condition:
    :param message:
    :return: None
    '''
    if not condition:
        raise ControlledError(message)

# Dummy class
class Dummy():
    def __init__(self):
        pass

def handle_errors(func):
    """
    Decorator function to handle logomaker errors
    """


    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Get should_fail debug flag
        should_fail = kwargs.pop('should_fail', None)


        try:
            # Execute function
            result = func(*args, **kwargs)

            error = False

            # If function didn't raise error, process results
            if should_fail is True:
                print('MISTAKE: Succeeded but should have failed.')
                mistake = True

            elif should_fail is False:
                print('Success, as expected.')
                mistake = False

            elif should_fail is None:
                mistake = False

            else:
                print('FATAL: should_fail = %s is not bool or None' %
                      should_fail)
                sys.exit(1)

        except (ControlledError) as e:

            error = True

            if should_fail is True:
                print('Error, as expected: ', e.__str__())
                mistake = False

            elif should_fail is False:
                print('MISTAKE: Failed but should have succeeded: ', e.__str__())
                mistake = True

            # Otherwise, just print an error and don't return anything
            else:
                print('Error: ', e.__str__())

        # If not in debug mode
        if should_fail is None:

            # If error, exit
            if error:
                # sys.exit(1)
                return

            # Otherwise, just return normal result
            else:
                return result

        # Otherwise, if in debug mode
        else:

            # If this is a constructor, set 'mistake' attribute of self
            if func.__name__ == '__init__':
                assert len(args) > 0
                args[0].mistake = mistake
                return None

            # Otherwise, create dummy object with mistake attribute
            else:
                obj = Dummy()
                obj.mistake = mistake
                return obj


    return wrapped_func

# this circumvents a non-logomaker error with python 3. Will debug.
import warnings
warnings.filterwarnings("ignore")

# module imports
from logomaker import validate
from logomaker.validate import validate_parameter, validate_dataframe

from logomaker.data import load_alignment
from logomaker.Logo import Logo
from logomaker.make_logo import make_logo
from logomaker.character import get_fontnames_dict, get_fontnames
from logomaker.make_styled_logo import make_styled_logo
from logomaker.load_meme import load_meme

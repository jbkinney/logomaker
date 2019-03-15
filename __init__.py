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


# Define error handling
class ControlledError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


'''

*** Note ***
The following implementation will not work with the decorator handle_errors.
Using it without the handle error decorator shows the entire stacktrace in 
jupyter, but not when running python directly, so I suggest we use the simpler
implementation of Controlled Error.

I suggest we remove this commented code after code review. 

-AT 
  

# this message prints out a message
# and stops execution and hides traceback.
def ControlledError(message):

        sys.excepthook = logomaker_excepthook
        raise Exception(message)

        
# method that helps complete the implementation of
# Controlled Error for logomaker.
def logomaker_excepthook(type, value, traceback):
    sys.tracebacklimit = 0
    print(value)
        
'''


def check(condition, message):

    """
    Checks a condition; raises a ControlledError with message if condition fails.
    :param condition:
    :param message:
    :return: None
    """

    if not condition:
        raise ControlledError(message)


# Dummy class.
# Need to justify leaving this in here. AT
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

# Rename useful stuff from within Logomaker

from logomaker.src.Logo import Logo
from logomaker.src.Glyph import Glyph
from logomaker.src.Glyph import list_font_families

from logomaker.src.data import transform_matrix
from logomaker.src.data import center_matrix
from logomaker.src.data import normalize_matrix
from logomaker.src.data import iupac_to_matrix
from logomaker.src.data import alignment_to_matrix

from logomaker.src.validate import validate_matrix
from logomaker.src.validate import validate_probability_mat
from logomaker.src.validate import validate_information_mat
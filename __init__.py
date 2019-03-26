from __future__ import division
from functools import wraps
import sys


class LogomakerError(Exception):
    """
    Class used by Logomaker to handle errors.

    parameters
    ----------

    message: (str)
        The message passed to check(). This only gets passed to the
        LogomakerError constructor when the condition passed to check() is
        False.
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class DebugResult:
    """
    Container class for debugging results.
    """
    def __init__(self, result, mistake):
        self.result = result
        self.mistake = mistake


def check(condition, message):

    """
    Checks a condition; raises a LogomakerError with message if condition
    evaluates to False

    parameters
    ----------

    condition: (bool)
        A condition that, if false, halts Logomaker execution and raises a
        clean error to user

    message: (str)
        The string to show user if condition is False.

    returns
    -------
    None
    """

    if not condition:
        raise LogomakerError(message)


def handle_errors(func):
    """
    Decorator function to handle Logomaker errors.

    This decorator allows the user to pass the keyword argument
    'should_fail' to any wrapped function.

    If should_fail is None (or is not set by user), the function executes
    normally, and can be called as

        result = func(*args, **kwargs)

    In particular, Python execution will halt if any errors are raised.

    However, if the user specifies should_fail=True or should_fail=False, then
    Python will not halt even in the presence of an error. Moreover, the
    function will return a tuple, e.g.,

        result, mistake = func(*args, should_fail=True, **kwargs)

    with mistake flagging whether or not the function failed or succeeded
    as expected.
    """

    @wraps(func)  # So wrapped_func has the same docstring, etc., as func
    def wrapped_func(*args, **kwargs):

        # Get should_fail debug flag AND remove should_fail from kwargs dict.
        # Need to use pop method so that 'should_fail' variable is not passed
        # to func() within try/except below
        should_fail = kwargs.pop('should_fail', None)

        # Otherwise, user passed something other than a bool for
        # should_fail, which isn't valid.
        check(should_fail in (True, False, None),
              'FATAL: should_fail = %s is not bool or None' % should_fail)

        # Default values for returned variables
        result = None
        mistake = None

        try:
            # Execute function
            result = func(*args, **kwargs)

            # If running functional test and expect to fail
            if should_fail is True:
                print('MISTAKE: Succeeded but should have failed.')
                mistake = True

            # If running functional test and expect to pass
            elif should_fail is False:
                print('Success, as expected.')
                mistake = False

            # If user did not pass should_fail then nothing more to do.
            else:
                pass

        except LogomakerError as e:

            # If running functional test and expect to fail
            if should_fail is True:
                print('Error, as expected: ',
                      e.__str__())
                mistake = False

            # If running functional test and expect to pass
            elif should_fail is False:
                print('MISTAKE: Failed but should have succeeded: ',
                      e.__str__())
                mistake = True

            # Otherwise, print error (but not stack trace) and halt execution
            else:
                # print('Error in', func.__name__+':', e.__str__())
                # Raise error
                raise e

        # If not in debug mode, and no error was detected above,
        if should_fail is None:

            # Just return result
            return result

        # Otherwise, if in debug mode
        else:

            # If func is a constructor,
            # extract self from args[0],
            # and set mistake attribute
            if func.__name__ == "__init__":
                args[0].mistake = mistake
                return None

            # Otherwise, return result and mistake status as attributes
            # of container class object
            else:
                return DebugResult(result, mistake)

    # Return the wrapped function to the user
    return wrapped_func


# Classes / functions imported with logomaker
from logomaker.src.Logo import Logo
from logomaker.src.Glyph import Glyph
from logomaker.src.Glyph import list_font_families
from logomaker.src.data import transform_matrix
from logomaker.src.data import iupac_to_matrix
from logomaker.src.data import alignment_to_matrix
from logomaker.src.data import saliency_to_matrix
from logomaker.src.validate import validate_matrix

# TODO: fold these two functions into transform_matrix
from logomaker.src.data import center_matrix
from logomaker.src.data import normalize_matrix

# TODO: fold these into validate_matrix
from logomaker.src.validate import validate_probability_mat
from logomaker.src.validate import validate_information_mat

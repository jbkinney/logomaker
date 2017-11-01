from __future__ import division
import ast
import re
from make_logo import make_logo
import inspect
import warnings

def make_styled_logo(style_file=None,
                     style_dict=None,
                     print_params=True,
                     print_warnings=True,
                     *args, **user_kwargs):
    """
    Description:

        Generates a logo using default parameters specified in a style file that
        can be overwritten by the user. For detailed information on all
        possible arguments, see make_logo()

    Return:

        logo (__init__.Logo): a rendered logo.

    Args:

        style_file (str): file containing default keyword arguments.

        style_dict (dict): dictionary containing style specifications.
            Overrides style_file specifications.

        print_params (bool): whether to print the specified parameters
            to stdout.

        print_warnings (bool): whether to print warnings to stderr.

        args (list): standard args list passed by user

        user_kwargs (dict): user-specified keyword arguments specifying style.
            Overrides both style_file and style_dict specifications.
    """

    # Copy kwargs explicitly specified by user
    kwargs = user_kwargs

    # If user provides a style dict
    if style_dict is not None:
        kwargs = dict(style_dict, **kwargs)

    # If user provides a parameters file
    if style_file is not None:
        file_kwargs = load_parameters(style_file, print_params, print_warnings)
        kwargs = dict(file_kwargs, **kwargs)

    # Make sure all arguments are actually valid
    names, x, xx, default_values = inspect.getargspec(make_logo)
    keys = kwargs.keys()
    for key in keys:
        if not key in names:
            del kwargs[key]
            message = "Parameter '%s' is invalid. Ignoring it." % key
            warnings.warn(message, UserWarning)

    # Make logo
    logo = make_logo(*args, **kwargs)

    # Return logo to user
    return logo


def load_parameters(file_name, print_params=True, print_warnings=True):
    """
    Description:

        Fills a dictionary with parameters parsed from specified file.

    Arg:

        file_name (str): Name of file containing parameter assignments.

        print_params (bool): whether to print the specified parameters
            to stdout.

        print_warnings (bool): whether to print warnings to stderr.

    Return:

        params_dict (dict): Dictionary containing parameter assignments
            parsed from parameters file
    """

    # Create dictionary to hold results
    params_dict = {}

    # Create regular expression for parsing parameter file lines
    pattern = re.compile(
        '^\s*(?P<param_name>[\w]+)\s*[:=]\s*(?P<param_value>.*)$'
    )

    # Quit if file_name is not specified
    if file_name is None:
        return params_dict

    # Open parameters file
    try:
        file_obj = open(file_name, 'r')
    except IOError:
        print('Error: could not open file %s for reading.' % file_name)
        raise IOError

    # Process each line of file and store resulting parameter values
    # in params_dict
    params_dict = {}
    prefix = '' # This is needed to parse multi-line files
    for line in file_obj:

        # removing leading and trailing whitespace
        line = prefix + line.strip()

        # Ignore lines that are empty or start with comment
        if (len(line) == 0) or line.startswith('#'):
            continue

        # Record current line plus a space in prefix, then continue to next
        # if line ends in a "\"
        if line[-1] == '\\':
            prefix += line[:-1] + ' '
            continue

        # Otherwise, clean prefix and continue with this parsing
        else:
            prefix = ''

        # Parse line using pattern
        match = re.match(pattern, line)

        # If line matches, record parameter name and value
        if match:
            param_name = match.group('param_name')
            param_value_str = match.group('param_value')

            # Evaluate parameter value as Python literal
            try:
                params_dict[param_name] = ast.literal_eval(param_value_str)
                if print_params:
                    print('[set] %s = %s' % (param_name, param_value_str))

            except ValueError:
                if print_warnings:
                    print(('Warning: could not set parameter "%s" because ' +
                          'could not interpret "%s" as literal.') %
                          (param_name, param_value_str))
            except SyntaxError:
                if print_warnings:
                    print(('Warning: could not set parameter "%s" because ' +
                          'of a syntax error in "%s".') %
                          (param_name, param_value_str))


        elif print_warnings:
            print('Warning: could not parse line "%s".' % line)

    return params_dict


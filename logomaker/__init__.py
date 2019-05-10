# Classes / functions imported with logomaker
from logomaker.src.Logo import Logo
from logomaker.src.Glyph import Glyph
from logomaker.src.Glyph import list_font_names
from logomaker.src.matrix import transform_matrix
from logomaker.src.matrix import sequence_to_matrix
from logomaker.src.matrix import alignment_to_matrix
from logomaker.src.matrix import saliency_to_matrix
from logomaker.src.validate import validate_matrix
from logomaker.src.colors import list_color_schemes
from logomaker.src.examples import list_example_matrices
from logomaker.src.examples import get_example_matrix
from logomaker.src.examples import list_example_datafiles
from logomaker.src.examples import open_example_datafile


# demo functions for logomaker
import os
import glob
from logomaker.src.error_handling import check, handle_errors

@handle_errors
def demo(name='fig1b'):

    """
    Performs a demonstration of the Logomaker software.

    parameters
    -----------

    name: (str)
        Must be one of {'fig1b', 'fig1c', 'fig1d', 'fig1e', 'fig1f', 'logo'}.

    returns
    -------
    None.

    """

    # build list of demo names and corresponding files
    example_dir = os.path.dirname(__file__)
    example_files = glob.glob('%s/examples/demo_*.py' % example_dir)
    examples_dict = {}
    for file_name in example_files:
        key = file_name.split('_')[-1][:-3]
        examples_dict[key] = file_name

    # check that name is valid
    check(name in examples_dict.keys(),
          'name = %s is not valid. Must be one of %s'
          % (repr(name), examples_dict.keys()))

    # open and run example file
    file_name = examples_dict[name]
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print('Running %s:\n%s\n%s\n%s' % \
              (file_name, line, content, line))
    exec(open(file_name).read())

    # Find logo in name space and, if it exists, return
    logo_names = [name for name in dir() if name[-5:] == '_logo']
    if len(logo_names) == 1:

        # returns logo if there is any
        logo = eval(logo_names[0])
        return logo

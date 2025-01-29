# Specify version
__version__ = '0.8.4'  # Updated 2025.01.29

# Standard library imports
import os
import re
import matplotlib.pyplot as plt

# Local imports
from logomaker.src.Logo import Logo
from logomaker.src.Glyph import Glyph, list_font_names
from logomaker.src.matrix import (
    transform_matrix,
    sequence_to_matrix,
    alignment_to_matrix,
    saliency_to_matrix,
    validate_matrix
)
from logomaker.src.colors import list_color_schemes
from logomaker.src.examples import (
    list_example_matrices,
    get_example_matrix,
    list_example_datafiles,
    open_example_datafile
)
from logomaker.src.error_handling import check, handle_errors, LogomakerError
from logomaker.tests.functional_tests_logomaker import run_tests

@handle_errors
def demo(name: str = 'fig1b') -> plt.Figure:
    """
    Performs a demonstration of the Logomaker software.

    Parameters
    ----------
    name: str
        Must be one of {'fig1b', 'fig1c', 'fig1d', 'fig1e', 'fig1f', 'logo'}.

    Returns
    -------
    matplotlib.figure.Figure
        The current matplotlib Figure object.

    Raises
    ------
    ValueError
        If the provided name is not in the list of valid examples.
    """

    # build list of demo names and corresponding file names
    example_dir = f'{os.path.dirname(__file__)}/examples'
    all_base_file_names = os.listdir(example_dir)
    example_file_names = [f'{example_dir}/{temp_name}'
                     for temp_name in all_base_file_names
                     if re.match(r'demo_.*\.py', temp_name)]
    examples_dict = {}
    for file_name in example_file_names:
        key = file_name.split('_')[-1][:-3]
        examples_dict[key] = file_name

    # check that name is valid
    check(name in examples_dict.keys(),
          f'name = {name} is not valid. Must be one of {examples_dict.keys()}')

    # open and run example file
    file_name = examples_dict[name]
    with open(file_name, 'r') as f:
        content = f.read()
        line = '-------------------------------------------------------------'
        print(f'Running {file_name}:\n{line}\n{content}\n{line}')
    exec(open(file_name).read())

    # return the current matplotlib Figure object
    return plt.gcf()


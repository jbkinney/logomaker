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
def demo(example='crp_energy_matrix'):

    """

    Performs a demonstration of the Logomaker software.

    parameters
    ----------

    example: (str)

        A string specifying which example matrix to draw a logo for. Must be one of
        the following.
        [
            'ars_enrichment_matrix', 'crp_counts_matrix'    , 'crp_energy_matrix'
            'nn_saliency_matrix'   , 'ss_probability_matrix', 'ww_counts_matrix',
            'ww_information_matrix'
        ]


    return
    ------

    None.

    """

    import matplotlib.pyplot as plt
    df = get_example_matrix(example)
    Logo(df,font_name='Arial Rounded MT Bold')
    plt.show()

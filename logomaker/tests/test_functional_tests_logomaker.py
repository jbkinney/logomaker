import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from logomaker.src.error_handling import LogomakerError
import pytest
import types
from .define_tests import (
    get_Logo_tests, get_Logo_style_glyphs_tests, get_Logo_fade_glyphs_in_probability_logo_tests, get_Logo_style_glyphs_below_tests, 
    get_Logo_style_single_glyph_tests, get_Logo_style_glyphs_in_sequence_tests, get_Logo_highlight_position_tests, get_Logo_highlight_position_range_tests,
    get_Logo_draw_baseline, get_Logo_style_xticks_tests, get_Logo_style_spines_tests, get_sequence_to_matrix_tests, get_alignment_to_matrix_tests, 
    get_saliency_to_matrix_tests, get_transform_matrix_tests, get_Glyph_tests, get_get_data_methods_tests, get_demo_tests,
    bool_fail_list, bool_success_list, good_crp_df, good_prob_df, random_df, color_dict_warning_df, bad_df1
)

# fixture for default Logo test kwargs
@pytest.fixture
def default_Logo_kwargs():
    return {'df': good_crp_df}

# Create a mapping of test DataFrame IDs to their names
df_name_map = {
    id(good_crp_df): "good_crp_df",
    id(good_prob_df): "good_prob_df",
    id(random_df): "random_df",
    id(color_dict_warning_df): "color_dict_warning_df"
}

def generate_id(param):
    """
    Create a custom ID based on the parameter. This function is used
    when tests are parametrized using @pytest.mark.parametrize.

    :param param: Parameter from @pytest.mark.parametrize
    :return: A string ID of the parameter
    """
    if isinstance(param, pd.DataFrame):
        # Look up the DataFrame by its ID in the mapping
        return df_name_map.get(id(param), "unknown_df")
    if isinstance(param, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return param.__name__.split('.')[-1]
    if isinstance(param, dict): # for input_kwargs
        # Generate id for each value in the dictionary
        dict_repr = {key: generate_id(value) for key, value in param.items()}
        return f"{dict_repr}"[:25]
    else:
        return f"{param}"[:25]
    


# helper method for functional test_for_mistake
@pytest.mark.skip()
def test_for_mistake(func, *args, **kw):
    """
    Run a function with the specified parameters and asserts False if there
    is an unexpected success or failure.

    Parameters
    ----------
    func: (function or class constructor)
        An executable function to which *args and **kwargs are passed.

    Returns
    -------
    None
    """
    # Run function
    obj = func(*args, **kw)

    if obj.mistake:
        assert False
       

@pytest.mark.skip()
def test_parameter_values(func,
                          var_name=None,
                          val = None,
                          should_fail=None,
                          **kwargs):
    """
    Tests if a function call with specific parameter
    value succeeds or fails as expected

    Parameters
    ----------
    func: (function)
        Executable to test. Can be function or class constructor.

    var_name: (str)
        Name of variable to test. If not specified, function is
        tested for success in the absence of any passed parameters.

    val:
        Value of specified var_name to be tested.

    should_fail: (bool)
        True if function is expected to fail, False otherwise.

    **kwargs:
        Other keyword variables to pass onto func.

    Returns
    -------

    None.

    """

    # User feed
    print("Testing %s() parameter %s ..." % (func.__name__, var_name))

    # If variable name is specified, test the value
    if var_name is not None:
        kwargs[var_name] = val
        test_for_mistake(func=func, should_fail=should_fail, **kwargs)

    # Otherwise, make sure function without parameters succeeds
    else:
        # User feedback
        print("Testing %s() without parameters." % func.__name__)

        # Test function
        test_for_mistake(func=func, should_fail=False, **kwargs)

    # close all figures that might have been generated
    plt.close('all')

# parametrize Logo tests
# define list of tuples outside for readability
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_tests(), ids=generate_id)
def test_Logo(func, var_name, val, should_fail, input_kwargs, default_Logo_kwargs):
    full_kwargs = default_Logo_kwargs.copy()
    full_kwargs.update(input_kwargs)
    test_parameter_values(func, var_name, val, should_fail, **full_kwargs)

# parametrize Logo style_glyphs tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_style_glyphs_tests(), ids=generate_id)
def test_Logo_style_glyphs(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo fade_glyphs in probability logo tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_fade_glyphs_in_probability_logo_tests(), ids=generate_id)
def test_Logo_fade_glyphs_in_probability_logo(func,var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo style_glyphs_below tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_style_glyphs_below_tests(), ids=generate_id)
def test_Logo_style_glyphs_below(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo style_single_glyph tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_style_single_glyph_tests(), ids=generate_id)
def test_Logo_style_single_glyph(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo style_glyphs_in_sequence tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_style_glyphs_in_sequence_tests(), ids=generate_id)
def test_Logo_style_glyphs_in_sequence(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo highlight_position tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_highlight_position_tests(), ids=generate_id)
def test_Logo_highlight_position(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo highlight_position_range tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_highlight_position_range_tests(), ids=generate_id)
def test_Logo_highlight_position_range(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo draw_baseline tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_draw_baseline(), ids=generate_id)
def test_Logo_draw_baseline(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo style_xticks tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_style_xticks_tests(), ids=generate_id)
def test_Logo_style_xticks(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo style_spines tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Logo_style_spines_tests(), ids=generate_id)
def test_Logo_style_spines(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo transform_matrix tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_transform_matrix_tests(), ids=generate_id)
def test_Logo_transform_matrix(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo sequence_to_matrix tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_sequence_to_matrix_tests(), ids=generate_id)
def test_Logo_sequence_to_matrix(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo alignment_to_matrix tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_alignment_to_matrix_tests(), ids=generate_id)
def test_Logo_alignment_to_matrix(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Logo saliency_to_matrix tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_saliency_to_matrix_tests(), ids=generate_id)
def test_Logo_saliency_to_matrix(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize Glyph tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_Glyph_tests(), ids=generate_id)
def test_Glyph(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize get_data_methods tests
@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_get_data_methods_tests(), ids=generate_id)
def test_get_data_methods(func, var_name, val, should_fail, input_kwargs):
    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)

# parametrize demo tests
#@pytest.mark.parametrize("func,var_name,val,should_fail,input_kwargs", get_demo_tests(), ids=generate_id)
#def test_demo(func, var_name, val, should_fail, input_kwargs):
#    test_parameter_values(func, var_name, val, should_fail, **input_kwargs)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

# Common test variables
bool_fail_list = [0, -1, 'True', 'x', 1]
bool_success_list = [False, True]

# df inputs that successfully execute when entered into Logo
good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)
good_prob_df = logomaker.get_example_matrix('ss_probability_matrix', print_description=False)
random_df = pd.DataFrame(np.random.randint(0, 3, size=(10, 4)), columns=list('ACGT'))
color_dict_warning_df = pd.DataFrame({'*': [1, 2]})

# ax input that successfully execute when entered into Logo
_, temp_ax = plt.subplots(figsize=[3, 3])

# df inputs that fail when entered into Logo
bad_df1 = 'x'

def get_Logo_tests():


    """
    Return a list of tests for the Logo class, to be used intest_functional_tests_logomaker.py.
    
    :return: list of tuples, each containing a test parameter, a value to test, a boolean indicating whether the test should fail, and a dictionary of keyword arguments:
            (param, val, should_fail, kwargs)
    """
    return [ 
        # test parameter df
        (logomaker.Logo, 'df', 0, True, {}), 
        (logomaker.Logo, 'df', -1, True, {}),
        (logomaker.Logo, 'df', 'True', True, {}),
        (logomaker.Logo, 'df', 'x', True, {}),
        (logomaker.Logo, 'df', 1, True, {}),
        (logomaker.Logo, 'df', bad_df1, True, {}),
        (logomaker.Logo, 'df', good_crp_df, False, {}),
        (logomaker.Logo, 'df', random_df, False, {}),
        (logomaker.Logo, 'df', color_dict_warning_df, False, {}),
        

        # test parameter colors
        (logomaker.Logo, 'color_scheme', 'x', True, {}),
        (logomaker.Logo, 'color_scheme', 'bad_color_name', True,{}),
        (logomaker.Logo, 'color_scheme', 3, True, {}),
        (logomaker.Logo, 'color_scheme', 'classic', False, {}),
        (logomaker.Logo, 'color_scheme', 'grays', False, {}),
        (logomaker.Logo, 'color_scheme', 'charge', False, {}),
        (logomaker.Logo, 'color_scheme', 'salmon', False, {}),
        (logomaker.Logo, 'color_scheme', 'classic', False, {'df': color_dict_warning_df}),
        (logomaker.Logo, 'color_scheme', 'grays', False, {'df': color_dict_warning_df}),
        (logomaker.Logo, 'color_scheme', 'charge', False, {'df': color_dict_warning_df}),
        (logomaker.Logo, 'color_scheme', 'salmon', False, {'df': color_dict_warning_df}),
       

        # test parameter font names
        (logomaker.Logo, 'font_name', True, True, {}),
        (logomaker.Logo, 'font_name', None, True, {}),
        (logomaker.Logo, 'font_name', good_crp_df, True, {}),
        (logomaker.Logo, 'font_name', 'DejaVu Sans', False, {}), 
        (logomaker.Logo, 'font_name', 'Arial Rounded MT Bold', False, {}),
        (logomaker.Logo, 'font_name', 'Times New Roman', False, {}),

        # test parameter stack order
        (logomaker.Logo, 'stack_order', 'incorrect_argument', True, {}), 
        (logomaker.Logo, 'stack_order', 0.0, True, {}),
        (logomaker.Logo, 'stack_order', None, True, {}),
        (logomaker.Logo, 'stack_order', 'small_on_top', False, {}), 
        (logomaker.Logo, 'stack_order', 'big_on_top', False, {}),
        (logomaker.Logo, 'stack_order', 'fixed', False, {}),

        # test parameter center_values
        (logomaker.Logo, 'center_values', 'incorrect_argument', True, {}), 
        (logomaker.Logo, 'center_values', 0.0, True, {}),
        (logomaker.Logo, 'center_values', None, True, {}),
        (logomaker.Logo, 'center_values', True, False, {}), 
        (logomaker.Logo, 'center_values', False, False, {}),

        # test parameter baseline_width
        (logomaker.Logo, 'baseline_width', 'incorrect_argument', True, {}), 
        (logomaker.Logo, 'baseline_width', -0.1, True, {}),
        (logomaker.Logo, 'baseline_width', None, True, {}),
        (logomaker.Logo, 'baseline_width', 0, False, {}), 
        (logomaker.Logo, 'baseline_width', 0.5, False, {}),  
        (logomaker.Logo, 'baseline_width', 3, False, {}),

        # test parameter flip_below
        (logomaker.Logo, 'flip_below', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'flip_below', -0.1, True, {}),
        (logomaker.Logo, 'flip_below', None, True, {}),
        (logomaker.Logo, 'flip_below', True, False, {}),
        (logomaker.Logo, 'flip_below', False, False, {}),

        # test parameter shade_below
        (logomaker.Logo, 'shade_below', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'shade_below', -0.1, True, {}),
        (logomaker.Logo, 'shade_below', 1.4, True, {}),
        (logomaker.Logo, 'shade_below', None, True, {}),
        (logomaker.Logo, 'shade_below', 0, False, {}),
        (logomaker.Logo, 'shade_below', 0.0, False, {}),
        (logomaker.Logo, 'shade_below', 0.5, False, {}),
        (logomaker.Logo, 'shade_below', 1, False, {}),
        (logomaker.Logo, 'shade_below', 1.0, False, {}),

        # test parameter fade_below
        (logomaker.Logo, 'fade_below', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'fade_below', -0.1, True, {}),
        (logomaker.Logo, 'fade_below', 1.4, True, {}),
        (logomaker.Logo, 'fade_below', None, True, {}),
        (logomaker.Logo, 'fade_below', 0, False, {}),
        (logomaker.Logo, 'fade_below', 0.0, False, {}),
        (logomaker.Logo, 'fade_below', 0.5, False, {}),
        (logomaker.Logo, 'fade_below', 1, False, {}),
        (logomaker.Logo, 'fade_below', 1.0, False, {}),

        #test parameter fade_probabilities
        (logomaker.Logo, 'fade_probabilities', 'incorrect_argument', True, {'df': good_prob_df}),
        (logomaker.Logo, 'fade_probabilities', -0.1, True, {'df': good_prob_df}),
        (logomaker.Logo, 'fade_probabilities', 1.4  , True, {'df': good_prob_df}),
        (logomaker.Logo, 'fade_probabilities', None, True, {'df': good_prob_df}),
        (logomaker.Logo, 'fade_probabilities', True, False, {'df': good_prob_df}), 
        (logomaker.Logo, 'fade_probabilities', False, False, {'df': good_prob_df}),

        # test parameter vsep
        (logomaker.Logo, 'vsep', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'vsep', -0.1, True, {}),
        (logomaker.Logo, 'vsep', None, True, {}),
        (logomaker.Logo, 'vsep', 0.0, False, {}),
        (logomaker.Logo, 'vsep', 0, False, {}),
        (logomaker.Logo, 'vsep', 0.3, False, {}),
        (logomaker.Logo, 'vsep', 10, False, {}),
        
        # test parameter show_spines
        # TODO: note that a value of True/False is still causing a logo to be drawn, eventhough draw_now = False
        (logomaker.Logo, 'show_spines', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'show_spines', -0.1, True, {}),
        (logomaker.Logo, 'show_spines', None, False, {}),
        (logomaker.Logo, 'show_spines', True, False, {}),
        (logomaker.Logo, 'show_spines', False, False, {}),

        # test parameter zorder. Need to review zorder's input check in Logo
        (logomaker.Logo, 'zorder', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'zorder', 0, False, {}),
        (logomaker.Logo, 'zorder', 1, False, {}),
        (logomaker.Logo, 'zorder', 3, False, {}),

        # test parameter figsize
        (logomaker.Logo, 'figsize', 'incorrect_argument', True, {}),
        (logomaker.Logo, 'figsize', -0.1, True, {}),
        (logomaker.Logo, 'figsize', [-1,-1], True, {}),
        (logomaker.Logo, 'figsize', [-1], True, {}),
        (logomaker.Logo, 'figsize', [0,0], True, {}),
        (logomaker.Logo, 'figsize', ['x','y'], True, {}),
        (logomaker.Logo, 'figsize', (1,2,3), True, {}),
        (logomaker.Logo, 'figsize', (10, 2.5), False, {}),
        (logomaker.Logo, 'figsize', (5,5), False, {}),

        # validate ax
        (logomaker.Logo, 'ax', 1, True, {}),
        (logomaker.Logo, 'ax', 'x', True, {}),
        (logomaker.Logo, 'ax', None, False, {}),
        (logomaker.Logo, 'ax', temp_ax, False, {}),

        ]
def get_Logo_style_glyphs_tests():
    """
    Tests for the style_glyphs method of the Logo class.
    """
    return [
        # test parameter color_scheme
        (logomaker.Logo(good_crp_df).style_glyphs, 'color_scheme', 'bad_color_scheme', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs, 'color_scheme', 1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs, 'color_scheme', True, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs, 'color_scheme', 'classic', False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs, 'color_scheme', 'gray', False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs, 'color_scheme', 'salmon', False, {})

        # TODO: how should we test kwargs for this (and other) methods?
         
    ]
def get_Logo_fade_glyphs_in_probability_logo_tests():
    """
    Tests for the fade_glyphs_in_probability_logo method of the Logo class.
    """
    return [
        # test parameter v_alpha0
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', -1.1, True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 1.1, True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 1.0, True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 'xxx', True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', True, True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 0, False, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 0.0, False, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 0.999, False, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha0', 0.5, False, {}),


        # TODO: a value of True v_alpha_1 works now, this should probably be fixed.
        # test parameter v_alpha1
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha1', 1.1, True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha1', -1, True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha1', 'xxx', True, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha1', 0.999, False, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha1', 0.5, False, {}),
        (logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, 'v_alpha1', 1.0, False, {}),
        
    ]
def get_Logo_style_glyphs_below_tests():
    """
    Tests for the style_glyphs_below method of the Logo class.
    """
    return [
        # test parameter color
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', 0, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', 'xxx', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', True, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', [0,0,-1], True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', 'red', False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', [1,1,1], False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', [0,0,0], False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', [0.1,0.2,0.3], False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'color', None, False, {}),

        # test parameter alpha
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'alpha', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'alpha', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'alpha', 1.1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'alpha', 0.1, False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'alpha', 0.5, False, {}),

        # test parameter shade
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'shade', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'shade', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'shade', 1.1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'shade', 0.1, False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'shade', 0.5, False, {}),

        # test parameter fade
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'fade', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'fade', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'fade', 1.1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'fade', 0.1, False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'fade', 0.5, False, {}),

        # test parameter flip
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'flip', 0, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'flip', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'flip', 'True', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'flip', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'flip', False, False, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_below, 'flip', True, False, {}),

    ]
def get_Logo_style_single_glyph_tests():
    """
    Tests for the style_single_glyph method of the Logo class.
    """
    return [
        # test parameter p
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', -1, True, {'c': 'A'}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', 'x', True, {'c': 'A'}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', 1.1, True, {'c': 'A'}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', 10000, True, {'c': 'A'}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', 0, False, {'c': 'A'}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', 1, False, {'c': 'A'}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'p', 10, False, {'c': 'A'}),

        # test parameter c
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 0, True, {'p': 1}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 'x', True, {'p': 1}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 1.1, True, {'p': 1}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 'A', False, {'p': 1}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 'C', False, {'p': 1}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 'G', False, {'p': 1}),
        (logomaker.Logo(good_crp_df).style_single_glyph, 'c', 'T', False, {'p': 1})

    ]

# randomly make up a sequence of the correct length
test_good_sequence = np.random.choice(['A', 'C', 'G', 'T'], size=26, p=[0.25, 0.25, 0.25, 0.25])
test_good_sequence = "".join(test_good_sequence)

# randomly make up a sequence of the incorrect length
test_bad_sequence = np.random.choice(['A', 'C', 'G', 'T'], size=10, p=[0.25, 0.25, 0.25, 0.25])
test_bad_sequence = "".join(test_bad_sequence)

def get_Logo_style_glyphs_in_sequence_tests():
    """
    Tests for the style_glyphs_in_sequence method of the Logo class.
    """
    return [
        # test parameter sequence
        (logomaker.Logo(good_crp_df).style_glyphs_in_sequence, 'sequence', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_in_sequence, 'sequence', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_in_sequence, 'sequence', 1.1, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_in_sequence, 'sequence', test_bad_sequence, True, {}),
        (logomaker.Logo(good_crp_df).style_glyphs_in_sequence, 'sequence', test_good_sequence, False, {})
    ]
def get_Logo_highlight_position_tests():
    """
    Tests for the highlight_position method of the Logo class.
    """
    return [
        # test parameter p
        (logomaker.Logo(good_crp_df).highlight_position, 'p', 'x', True, {}),
        (logomaker.Logo(good_crp_df).highlight_position, 'p', 1.5, True, {}),
        (logomaker.Logo(good_crp_df).highlight_position, 'p', 0, False, {}),
        (logomaker.Logo(good_crp_df).highlight_position, 'p', 1, False, {}),
        (logomaker.Logo(good_crp_df).highlight_position, 'p', 10, False, {})
    ]
def get_Logo_highlight_position_range_tests():
    """
    Tests for the highlight_position_range method of the Logo class.
    """
    return [
        # test parameter pmin
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmin', 'x', True, {'pmax':15}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmin', 20, True, {'pmax':15}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmin', 0, False, {'pmax':15}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmin', 1, False, {'pmax':15}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmin', 10, False, {'pmax':15}),
    
        # test parameter pmax
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmax', 'x', True, {'pmin':5}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmax', 1, True, {'pmin':5}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmax', 5.5, False, {'pmin':5}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmax', 6, False, {'pmin':5}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'pmax', 10, False, {'pmin':5}),

        # test parameter padding
        (logomaker.Logo(good_crp_df).highlight_position_range, 'padding', 'x', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'padding', -1, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'padding', -0.5, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'padding', 0, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'padding', 10, False, {'pmin':5, 'pmax':10}),

        # test parameter color
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', 'x', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', 1, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', True, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', 'wrong_color', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', 'pink', False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', 'red', False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'color', [1,1,1], False, {'pmin':5, 'pmax':10}),

        # test parameter edgecolor
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', 'x', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', 1, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', True, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', 'wrong_color', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', 'pink', False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', 'red', False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'edgecolor', [1,1,1], False, {'pmin':5, 'pmax':10}),

        # test parameter floor
        (logomaker.Logo(good_crp_df).highlight_position_range, 'floor', 'x', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'floor', 10, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'floor', -1, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'floor', 1, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'floor', None, False, {'pmin':5, 'pmax':10}),

        # test parameter ceiling
        (logomaker.Logo(good_crp_df).highlight_position_range, 'ceiling', 'x', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'ceiling', -10, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'ceiling', -1, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'ceiling', 1, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'ceiling', None, False, {'pmin':5, 'pmax':10}),
        
        # test parameter zorder. Note that a value of False passes for this parameter. This should be fixed.
        (logomaker.Logo(good_crp_df).highlight_position_range, 'zorder', 'x', True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'zorder', None, True, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'zorder', -1, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'zorder', 0.5, False, {'pmin':5, 'pmax':10}),
        (logomaker.Logo(good_crp_df).highlight_position_range, 'zorder', 1, False, {'pmin':5, 'pmax':10}),
    ]
def get_Logo_draw_baseline():
    """
    Tests for the draw_baseline method of the Logo class.
    """
    return [
        # test parameter zorder
        (logomaker.Logo(good_crp_df).draw_baseline, 'zorder', 'x', True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'zorder', None, True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'zorder', -1, False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'zorder', 0.5, False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'zorder', 1, False, {}),

        # test parameter color
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', 'x', True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', 1, True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', True, True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', 'wrong_color', True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', 'pink', False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', 'red', False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'color', [1,1,1], False, {}),

        # test parameter linewidth
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', 'x', True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', -1, True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', '1', True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', None, True, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', 0, False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', 1, False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', 1.5, False, {}),
        (logomaker.Logo(good_crp_df).draw_baseline, 'linewidth', 2, False, {}),


    ]
def get_Logo_style_xticks_tests():
    """
    Tests for the style_xticks method of the Logo class.
    """
    return [
        # test parameter anchor
        (logomaker.Logo(good_crp_df).style_xticks, 'anchor', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'anchor', None, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'anchor', 0.5, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'anchor', 1.0, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'anchor', 1, False, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'anchor', 2, False, {}),

        # test parameter spacing
        (logomaker.Logo(good_crp_df).style_xticks, 'spacing', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'spacing', None, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'spacing', 0, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'spacing', 0.5, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'spacing', 1, False, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'spacing', 2, False, {}),

        # test parameter fmt
        # TODO: fmt = 'x' seems to passing in the following. This should be fixed.
        (logomaker.Logo(good_crp_df).style_xticks, 'fmt', None, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'fmt', 0, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'fmt', '%d', False, {}),

        # test parameter rotation
        (logomaker.Logo(good_crp_df).style_xticks, 'rotation', None, True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'rotation', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'rotation', -12, False, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'rotation', 0, False, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'rotation', 1.4, False, {}),
        (logomaker.Logo(good_crp_df).style_xticks, 'rotation', 200, False, {})

        
    
        
        
    ]
def get_Logo_style_spines_tests():
    """
    Tests for the style_spines method of the Logo class.
    """
    return [
        # test parameter spines
        (logomaker.Logo(good_crp_df).style_spines, 'spines', 'x', True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', None, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', 0.5, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', 1.0, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', 'top', True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', ('top', 'bottom', 'left', 'right'), False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', ['top', 'bottom', 'left', 'right'], False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', ('top', 'bottom', 'left'), False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', ('top', 'bottom'), False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'spines', ['top'], False, {}),

        # test parameter visible
        (logomaker.Logo(good_crp_df).style_spines, 'visible', 0, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'visible', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'visible', 'True', True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'visible', 1, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'visible', False, False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'visible', True, False, {}),
        
        # test parameter color
        (logomaker.Logo(good_crp_df).style_spines, 'color', 'wrong_color', True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'color', 3, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'color', [0.5,0.5,0.5,0.5], True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'color', 'black', False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'color', 'green', False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'color', [0.4,0.5,1.0], False, {}),

        # test parameter linewidth
        (logomaker.Logo(good_crp_df).style_spines, 'linewidth', 'xxx', True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'linewidth', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'linewidth', 0.0, False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'linewidth', 0, False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'linewidth', 1, False, {}),

        # test parameter bounds
        (logomaker.Logo(good_crp_df).style_spines, 'bounds', 'xxx', True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'bounds', -1, True, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'bounds', None, False, {}),
        (logomaker.Logo(good_crp_df).style_spines, 'bounds', [0,1], False, {}),

    ]

# good matrices for testing transform_matrix
good_crp_weight_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)
good_crp_counts_df = logomaker.get_example_matrix('crp_counts_matrix', print_description=False)

def get_transform_matrix_tests():
    """
    Tests for the transform_matrix method.
    """
    return [
        # test parameter df
        (logomaker.transform_matrix, 'df', 'x', True, {'from_type': 'counts', 'to_type': 'probability'}),
        (logomaker.transform_matrix, 'df', good_crp_weight_df, True, {'from_type': 'counts', 'to_type': 'probability'}),
        (logomaker.transform_matrix, 'df', None, True, {'from_type': 'counts', 'to_type': 'probability'}),
        (logomaker.transform_matrix, 'df', good_crp_counts_df, False, {'from_type': 'counts', 'to_type': 'probability'}),

        (logomaker.transform_matrix, 'df', good_crp_counts_df, False, {'from_type': 'counts', 'to_type': 'weight'}),

        (logomaker.transform_matrix, 'df', good_crp_counts_df, False, {'from_type': 'counts', 'to_type': 'information'}),

        # test parameter center_values
        (logomaker.transform_matrix, 'center_values', 0, True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'center_values', -1, True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'center_values', 'True', True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'center_values', 'x', True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'center_values', 1, True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'center_values', False, False, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'center_values', True, False, {'df': good_crp_counts_df}),

        # test parameter normalize_values
        (logomaker.transform_matrix, 'normalize_values', 0, True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'normalize_values', -1, True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'normalize_values', 'True', True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'normalize_values', 'x', True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'normalize_values', 1, True, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'normalize_values', False, False, {'df': good_crp_counts_df}),
        (logomaker.transform_matrix, 'normalize_values', True, False, {'df': good_crp_counts_df}),

        # test parameter from_type
        (logomaker.transform_matrix, 'from_type', 1, True, {'df': good_crp_counts_df, 'to_type': 'probability'}),
        (logomaker.transform_matrix, 'from_type', 'x', True, {'df': good_crp_counts_df, 'to_type': 'probability'}),
        (logomaker.transform_matrix, 'from_type', None, True, {'df': good_crp_counts_df, 'to_type': 'probability'}),
        (logomaker.transform_matrix, 'from_type', 'counts', False, {'df': good_crp_counts_df, 'to_type': 'probability'}),

        # test parameter to_type
        (logomaker.transform_matrix, 'to_type', 1, True, {'df': good_crp_counts_df, 'from_type': 'counts'}),
        (logomaker.transform_matrix, 'to_type', 'x', True, {'df': good_crp_counts_df, 'from_type': 'counts'}),
        (logomaker.transform_matrix, 'to_type', None, True, {'df': good_crp_counts_df, 'from_type': 'counts'}),
        (logomaker.transform_matrix, 'to_type', 'probability', False, {'df': good_crp_counts_df, 'from_type': 'counts'}),
        (logomaker.transform_matrix, 'to_type', 'weight', False, {'df': good_crp_counts_df, 'from_type': 'counts'}),
        (logomaker.transform_matrix, 'to_type', 'information', False, {'df': good_crp_counts_df, 'from_type': 'counts'}),

        # test parameter background
        (logomaker.transform_matrix, 'background', 1, True, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'background', 'x', True, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'background', [-1,1,1,1], True, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'background', None, False, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'background', [0.25,0.25,0.25,0.25], False, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        
        # test parameter pseudocount
        (logomaker.transform_matrix, 'pseudocount', None, True, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'pseudocount', 'x', True, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'pseudocount', -1, True, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'pseudocount', 0, False, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'pseudocount', 1, False, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
        (logomaker.transform_matrix, 'pseudocount', 10, False, {'df': good_crp_counts_df, 'from_type': 'counts', 'to_type': 'information'}),
    ]

def get_sequence_to_matrix_tests(): 
    """
    Tests for the sequence_to_matrix method.
    """
    return [
        # test parameter seq
        (logomaker.sequence_to_matrix, 'seq', None, True, {}),
        (logomaker.sequence_to_matrix, 'seq', 3, True, {}),
        (logomaker.sequence_to_matrix, 'seq', True, True, {}),
        (logomaker.sequence_to_matrix, 'seq', 'ACGT', False, {}),
        (logomaker.sequence_to_matrix, 'seq', '!@#$', False, {}),
        (logomaker.sequence_to_matrix, 'seq', 'logomaker', False, {}),

        # test parameter cols
        (logomaker.sequence_to_matrix, 'cols', 0, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'cols', True, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'cols', None, False, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'cols', ['A','C','G','T'], False, {'seq':'ACGTACGT'}),

        # test parameter alphabet
        (logomaker.sequence_to_matrix, 'alphabet', 0, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'alphabet', True, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'alphabet', 'xxx', True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'alphabet', None, False, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'alphabet', 'dna', False, {'seq':'ACGTACGT'}),

        (logomaker.sequence_to_matrix, 'alphabet', 'rna', False, {'seq':'ACGUACGU'}),

        (logomaker.sequence_to_matrix, 'alphabet', 'protein', False, {'seq':'LMWA'}),

        # test parameter is_iupac
        (logomaker.sequence_to_matrix, 'is_iupac', 0, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'is_iupac', -1, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'is_iupac', 'True', True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'is_iupac', 'x', True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'is_iupac', 1, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'is_iupac', False, False, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'is_iupac', True, False, {'seq':'ACGTACGT'}),

        # test parameter to_type
        (logomaker.sequence_to_matrix, 'to_type', 0, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'to_type', True, True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'to_type', 'xxx', True, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'to_type', 'probability', False, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'to_type', 'weight', False, {'seq':'ACGTACGT'}),
        (logomaker.sequence_to_matrix, 'to_type', 'information', False, {'seq':'ACGTACGT'}),

        # test parameter center_weights
        (logomaker.sequence_to_matrix, 'center_weights', 0, True, {'seq':'ACGTACGT', 'to_type': 'weight'}),
        (logomaker.sequence_to_matrix, 'center_weights', -1, True, {'seq':'ACGTACGT', 'to_type': 'weight'}),
        (logomaker.sequence_to_matrix, 'center_weights', 'True', True, {'seq':'ACGTACGT', 'to_type': 'weight'}),
        (logomaker.sequence_to_matrix, 'center_weights', 'x', True, {'seq':'ACGTACGT', 'to_type': 'weight'}),
        (logomaker.sequence_to_matrix, 'center_weights', 1, True, {'seq':'ACGTACGT', 'to_type': 'weight'}),
        (logomaker.sequence_to_matrix, 'center_weights', False, False, {'seq':'ACGTACGT', 'to_type': 'weight'}),
        (logomaker.sequence_to_matrix, 'center_weights', True, False, {'seq':'ACGTACGT', 'to_type': 'weight'}),

    ]

# get sequences from file
with logomaker.open_example_datafile('crp_sites.fa', print_description=False) as f:
    raw_seqs = f.readlines()
seqs = [seq.strip() for seq in raw_seqs if ('#' not in seq) and ('>') not in seq]

def get_alignment_to_matrix_tests():
    """
    Tests for the alignment_to_matrix method.
    """
    return [
        # test parameter sequences
        (logomaker.alignment_to_matrix, 'sequences', 0, True, {}),
        (logomaker.alignment_to_matrix, 'sequences', 'x', True, {}),
        (logomaker.alignment_to_matrix, 'sequences', ['AACCT', 'AACCATA'], True, {}),
        (logomaker.alignment_to_matrix, 'sequences', seqs, False, {}),
        (logomaker.alignment_to_matrix, 'sequences', ['ACA', 'GGA'], False, {}),

        # test parameter counts
        (logomaker.alignment_to_matrix, 'counts', 0, True, {'sequences':['ACA', 'GGA']}),
        (logomaker.alignment_to_matrix, 'counts', 'x', True, {'sequences':['ACA', 'GGA']}),
        (logomaker.alignment_to_matrix, 'counts', -1, True, {'sequences':['ACA', 'GGA']}),
        (logomaker.alignment_to_matrix, 'counts', None, False, {'sequences':['ACA', 'GGA']}),
        (logomaker.alignment_to_matrix, 'counts', [3,1], False, {'sequences':['ACA', 'GGA']}),
        (logomaker.alignment_to_matrix, 'counts', np.array([3,1]), False, {'sequences':['ACA', 'GGA']}),

        # test parameter to_type
        (logomaker.alignment_to_matrix, 'to_type', 0, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'to_type', True, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'to_type', 'xxx', True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'to_type', 'counts', False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'to_type', 'probability', False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'to_type', 'weight', False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'to_type', 'information', False, {'sequences':seqs}),

        # test parameter background
        (logomaker.alignment_to_matrix, 'background', 1, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'background', 'x', True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'background', None, False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'background', [0.25,0.25,0.25,0.25], False, {'sequences':seqs}),

        # test parameter characters_to_ignore
        (logomaker.alignment_to_matrix, 'characters_to_ignore', 1, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'characters_to_ignore', 0.5, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'characters_to_ignore', True, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'characters_to_ignore', None, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'characters_to_ignore', 'A', False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'characters_to_ignore', 'C', False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'characters_to_ignore', 'G', False, {'sequences':seqs}),

        # test center_weights
        (logomaker.alignment_to_matrix, 'center_weights', 0, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'center_weights', -1, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'center_weights', 'True', True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'center_weights', 'x', True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'center_weights', 1, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'center_weights', False, False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'center_weights', True, False, {'sequences':seqs}),

        # test pseudocount
        (logomaker.alignment_to_matrix, 'pseudocount', None, True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'pseudocount', 'x', True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'pseudocount', '-1', True, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'pseudocount', 0, False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'pseudocount', 1, False, {'sequences':seqs}),
        (logomaker.alignment_to_matrix, 'pseudocount', 10, False, {'sequences':seqs}),
        
    ]

# load saliency data
with logomaker.open_example_datafile('nn_saliency_values.txt', print_description=False) as f:
    saliency_data_df = pd.read_csv(f, comment='#', sep='\t')

def get_saliency_to_matrix_tests():
    """
    Tests for the saliency_to_matrix method.
    """
    return [
        # test parameter seq
        (logomaker.saliency_to_matrix, 'seq', None, True, {'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'seq', 'x', True, {'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'seq', -1, True, {'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'seq', saliency_data_df['character'], False, {'values':saliency_data_df['value']}),
        
        # test parameter values
        (logomaker.saliency_to_matrix, 'values', None, True, {'seq':saliency_data_df['character']}),
        (logomaker.saliency_to_matrix, 'values', 'x', True, {'seq':saliency_data_df['character']}),
        (logomaker.saliency_to_matrix, 'values', -1, True, {'seq':saliency_data_df['character']}),
        (logomaker.saliency_to_matrix, 'values', saliency_data_df['value'], False, {'seq':saliency_data_df['character']}),

        # test parameter cols
        (logomaker.saliency_to_matrix, 'cols', -1, True, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'cols', 'x', True, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'cols', None, False, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'cols', ['A','C','G','T'], False, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        
        # test parameter alphabet
        (logomaker.saliency_to_matrix, 'alphabet', 0, True, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'alphabet', 'x', True, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'alphabet', 'xxx', True, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'alphabet', None, False, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),
        (logomaker.saliency_to_matrix, 'alphabet', 'dna', False, {'seq':saliency_data_df['character'], 'values':saliency_data_df['value']}),

        
    ]


fig, ax = plt.subplots(figsize=[7, 3])
# set bounding box
ax.set_xlim([0, 2])
ax.set_ylim([0, 1])


def get_Glyph_tests():
    """
    Tests for the Glyph class.
    """
    return [
        # test parameter p
        # TODO: need to fix fail_list bugs for parameter p
        (logomaker.Glyph, 'p', 0, False, {'c': 'A', 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'p', 10, False, {'c': 'A', 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'p', .5, False, {'c': 'A', 'ax': ax, 'floor':0, 'ceiling':1}),

        # test parameter c
        (logomaker.Glyph, 'c', 0, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'c', 0.1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'c', None, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'c', 'A', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'c', 'C', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1}),
        (logomaker.Glyph, 'c', 'X', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1}),

        # test parameter floor
        (logomaker.Glyph, 'floor', 'x', True, {'p':1, 'ax': ax, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'floor', 2, True, {'p':1, 'ax': ax, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'floor', None, True, {'p':1, 'ax': ax, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'floor', 0, False, {'p':1, 'ax': ax, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'floor', 0.1, False, {'p':1, 'ax': ax, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'floor', 0.5, False, {'p':1, 'ax': ax, 'ceiling':1, 'c':'A'}),

        # test parameter ceiling
        (logomaker.Glyph, 'ceiling', 'x', True, {'p':1, 'ax': ax, 'floor':0, 'c':'A'}),
        (logomaker.Glyph, 'ceiling', -1, True, {'p':1, 'ax': ax, 'floor':0, 'c':'A'}),
        (logomaker.Glyph, 'ceiling', None, True, {'p':1, 'ax': ax, 'floor':0, 'c':'A'}),
        (logomaker.Glyph, 'ceiling', 0, False, {'p':1, 'ax': ax, 'floor':0, 'c':'A'}),
        (logomaker.Glyph, 'ceiling', 0.1, False, {'p':1, 'ax': ax, 'floor':0, 'c':'A'}),
        (logomaker.Glyph, 'ceiling', 0.5, False, {'p':1, 'ax': ax, 'floor':0, 'c':'A'}),

        # test parameter ax
        (logomaker.Glyph, 'ax', 'x', True, {'p':1, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'ax', -1, True, {'p':1, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'ax', None, False, {'p':1, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'ax', ax, False, {'p':1, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter vpad
        (logomaker.Glyph, 'vpad', 'x', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'vpad', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'vpad', None, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'vpad', 1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'vpad', 0, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'vpad', 0.9, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'vpad', 0.5, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter font_name
        (logomaker.Glyph, 'font_name', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'font_name', 'DejaVu Sans', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        
        # test parameter font_weight
        (logomaker.Glyph, 'font_weight', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'font_weight', 'xxx', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'font_weight', 'bold', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'font_weight', 5, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'font_weight', 'normal', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter color
        (logomaker.Glyph, 'color', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'color', 'xxx', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'color', 'red', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'color', [1, 1, 1], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'color', [0, 0, 0], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'color', [0.1, 0.2, 0.3], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        
        # test parameter edgecolor
        (logomaker.Glyph, 'edgecolor', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgecolor', 'xxx', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgecolor', 'black', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgecolor', [1, 1, 1], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgecolor', [0, 0, 0], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgecolor', [0.1, 0.2, 0.3], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter edgewidth
        (logomaker.Glyph, 'edgewidth', 'x', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgewidth', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgewidth', None, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgewidth', 0, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgewidth', 0.9, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'edgewidth', 0.5, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter dont_stretch_more_than
        (logomaker.Glyph, 'dont_stretch_more_than', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'dont_stretch_more_than', None, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'dont_stretch_more_than', 'E', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'dont_stretch_more_than', 'A', False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter flip
        (logomaker.Glyph, 'flip', 0, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'flip', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'flip', 'True', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'flip', 'x', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'flip', 1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'flip', False, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'flip', True, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        
        # test parameter mirror
        (logomaker.Glyph, 'mirror', 0, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'mirror', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'mirror', 'True', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'mirror', 'x', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'mirror', 1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'mirror', False, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'mirror', True, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter zorder
        (logomaker.Glyph, 'zorder', 'x', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'zorder', 0, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'zorder', 0.5, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'zorder', 1, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'zorder', 5, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter alpha
        (logomaker.Glyph, 'alpha', 1.1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'alpha', -1, True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'alpha', 'xxx', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'alpha', 0.999, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'alpha', 0.5, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'alpha', 1.0, False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

        # test parameter figsize
        (logomaker.Glyph, 'figsize', 'incorrect argument', True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'figsize', ['x', 'y'], True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'figsize', (1,2,3), True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'figsize', [1, -1], True, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'figsize', (10, 2.5), False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),
        (logomaker.Glyph, 'figsize', [5, 5], False, {'p':1, 'ax': ax, 'floor':0, 'ceiling':1, 'c':'A'}),

    ]
def get_get_data_methods_tests():
    """
    Tests for the get_example_matrix and open_example_datafile methods.
    """
    return [
        # test parameter name
        (logomaker.get_example_matrix, 'name', 'wrong argument', True, {'print_description':False}),
        (logomaker.get_example_matrix, 'name', -1, True, {'print_description':False}),
        (logomaker.get_example_matrix, 'name', 'crp_energy_matrix', False, {'print_description':False}),
        (logomaker.get_example_matrix, 'name', 'ww_counts_matrix', False, {'print_description':False}),

        (logomaker.open_example_datafile, 'name', 'wrong argument', True, {'print_description':False}),
        (logomaker.open_example_datafile, 'name', -1, True, {'print_description':False}),
        (logomaker.open_example_datafile, 'name', 'nn_saliency_values.txt', False, {'print_description':False}),
        (logomaker.open_example_datafile, 'name', 'ss_sequences.txt', False, {'print_description':False}),
        
        #bool_fail_list = [0, -1, 'True', 'x', 1]

        # test parameter print_description
        (logomaker.get_example_matrix, 'print_description', 0, True, {'name':'crp_energy_matrix'}),
        (logomaker.get_example_matrix, 'print_description', -1, True, {'name':'crp_energy_matrix'}),
        (logomaker.get_example_matrix, 'print_description', 'True', True, {'name':'crp_energy_matrix'}),
        (logomaker.get_example_matrix, 'print_description', 'x', True, {'name':'crp_energy_matrix'}),
        (logomaker.get_example_matrix, 'print_description', 1, True, {'name':'crp_energy_matrix'}),
        (logomaker.get_example_matrix, 'print_description', True, False, {'name':'crp_energy_matrix'}),
        (logomaker.get_example_matrix, 'print_description', False, False, {'name':'crp_energy_matrix'}),

        (logomaker.open_example_datafile, 'print_description', 0, True, {'name':'nn_saliency_values.txt'}),
        (logomaker.open_example_datafile, 'print_description', -1, True, {'name':'nn_saliency_values.txt'}),
        (logomaker.open_example_datafile, 'print_description', 'True', True, {'name':'nn_saliency_values.txt'}),
        (logomaker.open_example_datafile, 'print_description', 'x', True, {'name':'nn_saliency_values.txt'}),
        (logomaker.open_example_datafile, 'print_description', 1, True, {'name':'nn_saliency_values.txt'}),
        (logomaker.open_example_datafile, 'print_description', True, False, {'name':'nn_saliency_values.txt'}),
        (logomaker.open_example_datafile, 'print_description', False, False, {'name':'nn_saliency_values.txt'}),
    ]
def get_demo_tests():
    """
    Tests for the demo.
    """
    return [
        (logomaker.demo, 'name', 0, True, {}),
        (logomaker.demo, 'name', True, True, {}),
        (logomaker.demo, 'name', 'xxx', True, {}),
        (logomaker.demo, 'name', 'crp', False, {}),
        (logomaker.demo, 'name', 'fig1b', False, {}),
    ]

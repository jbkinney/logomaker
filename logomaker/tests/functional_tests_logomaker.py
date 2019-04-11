#from __future__ import print_function   # so that print behaves like python 3.x not a special lambda statement

import sys
sys.path.append('../../')

import logomaker
import numpy as np
import pandas as pd

global_success_counter = 0
global_fail_counter = 0

# Common success and fail lists
bool_fail_list = [0, -1, 'True', 'x', 1]
bool_success_list = [False, True]

# helper method for functional test_for_mistake
def test_for_mistake(func, *args, **kw):
    """
    Run a function with the specified parameters and register whether
    success or failure was a mistake

    parameters
    ----------

    func: (function or class constructor)
        An executable function to which *args and **kwargs are passed.

    return
    ------

    None.
    """

    global global_fail_counter
    global global_success_counter

    # print test number
    test_num = global_fail_counter + global_success_counter
    print('Test # %d: ' % test_num, end='')
    #print('Test # %d: ' % test_num)

    # Run function
    obj = func(*args, **kw)
    # Increment appropriate counter
    if obj.mistake:
        global_fail_counter += 1
    else:
        global_success_counter += 1


def test_parameter_values(func,
                          var_name=None,
                          fail_list=[],
                          success_list=[],
                          **kwargs):
    """
    Tests predictable success & failure of different values for a
    specified parameter when passed to a specified function

    parameters
    ----------

    func: (function)
        Executable to test. Can be function or class constructor.

    var_name: (str)
        Name of variable to test. If not specified, function is
        tested for success in the absence of any passed parameters.

    fail_list: (list)
        List of values for specified variable that should fail

    success_list: (list)
        List of values for specified variable that should succeed

    **kwargs:
        Other keyword variables to pass onto func.

    return
    ------

    None.

    """

    # If variable name is specified, test each value in fail_list
    # and success_list
    if var_name is not None:

        # User feedback
        print("Testing %s() parameter %s ..." % (func.__name__, var_name))

        # Test parameter values that should fail
        for x in fail_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=True, **kwargs)

        # Test parameter values that should succeed
        for x in success_list:
            kwargs[var_name] = x
            test_for_mistake(func=func, should_fail=False, **kwargs)

        print("Tests passed: %d. Tests failed: %d.\n" %
              (global_success_counter, global_fail_counter))

    # Otherwise, make sure function without parameters succeeds
    else:

        # User feedback
        print("Testing %s() without parameters." % func.__name__)

        # Test function
        test_for_mistake(func=func, should_fail=False, **kwargs)


def test_logomaker_Logo():

    # df inputs that successfully execute when entered into Logo.
    good_rnap_df = pd.read_csv('../data/rnap_tau_final_all.41.matrix.txt', delim_whitespace=True)
    random_df = pd.DataFrame(np.random.randint(0,3, size=(10, 4)), columns=list('ACGT'))

    # df inputs that fail when entered into Logo.
    bad_df1 = 'x'

    # test parameter df
    test_parameter_values(func=logomaker.Logo, var_name='df',fail_list=[bool_fail_list,bad_df1],
                          success_list=[good_rnap_df,random_df],draw_now=False)

    # test parameter colors
    test_parameter_values(func=logomaker.Logo, var_name='color_scheme', fail_list=['x','bad_color_name',3],
                          success_list=['classic', 'grays', 'charge','salmon'],
                          df=good_rnap_df,draw_now=False)

    # test parameter font names
    test_parameter_values(func=logomaker.Logo, var_name='font_name', fail_list=['x', 'bad_font_name', good_rnap_df],
                          success_list=['DejaVu Sans', 'Arial Rounded MT Bold', 'Times New Roman'],
                          df=good_rnap_df,draw_now=False)

    # test parameter stack_order
    test_parameter_values(func=logomaker.Logo, var_name='stack_order', fail_list=['incorrect argument', 0.0, None],
                          success_list=['small_on_top', 'big_on_top', 'fixed'],
                          df=good_rnap_df,draw_now=False)

    # test parameter center_values
    test_parameter_values(func=logomaker.Logo, var_name='center_values', fail_list=['incorrect argument', 0.0, None],
                          success_list=[True, False], df=good_rnap_df,draw_now=False)

    # test parameter baseline_width
    test_parameter_values(func=logomaker.Logo, var_name='baseline_width', fail_list=['incorrect argument', -0.1, None],
                          success_list=[0,0.5,3], df=good_rnap_df,draw_now=False)

    # test parameter flip_below
    test_parameter_values(func=logomaker.Logo, var_name='flip_below', fail_list=['incorrect argument', -0.1, None],
                          success_list=[True, False], df=good_rnap_df, draw_now=False)

    # test parameter shade_below
    test_parameter_values(func=logomaker.Logo, var_name='shade_below',
                          fail_list=['incorrect argument', -0.1, 1.4, None],
                          success_list=[0, 0.0, 0.5, 1, 1.0], df=good_rnap_df, draw_now=False)

    # test parameter fade_below
    test_parameter_values(func=logomaker.Logo, var_name='fade_below',
                          fail_list=['incorrect argument', -0.1, 1.4, None],
                          success_list=[0, 0.0, 0.5, 1, 1.0], df=good_rnap_df, draw_now=False)

    # test parameter fade_probabilities
    test_parameter_values(func=logomaker.Logo, var_name='fade_probabilities',
                          fail_list=['incorrect argument', -0.1, 1.4, None],
                          success_list=[True,False], df=good_rnap_df, draw_now=False)

    # test parameter vsep
    test_parameter_values(func=logomaker.Logo, var_name='vsep',
                          fail_list=['incorrect argument', -0.1, None],
                          success_list=[0.0, 0,0.3,10], df=good_rnap_df, draw_now=False)

    # test parameter vsep
    # TODO: note that a value of True/False is still causing a logo to be drawn, eventhough draw_now = False
    test_parameter_values(func=logomaker.Logo, var_name='show_spines',
                          fail_list=['incorrect argument', -0.1],
                          success_list=[None, True, False], df=good_rnap_df, draw_now=False)

    # test parameter zorder. Need to review zorder's input check in Logo
    test_parameter_values(func=logomaker.Logo, var_name='zorder',
                          fail_list=['incorrect argument'],
                          success_list=[0, 1, 3], df=good_rnap_df, draw_now=False)

    # test parameter figsize
    test_parameter_values(func=logomaker.Logo, var_name='figsize',
                          fail_list=['incorrect argument', -0.1, [-1,-1],[-1],[0,0],['x','y'],(1,2,3)],
                          success_list=[(10, 2.5),[5,5]], df=good_rnap_df, draw_now=False)

    # validate draw_now
    test_parameter_values(func=logomaker.Logo, var_name='draw_now',
                          fail_list=['incorrect argument', -0.1],
                          success_list=[True, False], df=good_rnap_df)

    # TODO: need to implement input check for 'ax' in logo and then implement functional tests for it.


test_logomaker_Logo()

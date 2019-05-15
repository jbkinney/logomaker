from __future__ import print_function   # so that print behaves like python 3.x not a special lambda statement

import sys
sys.path.append('../../')

import logomaker

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # close all figures that might have been generated
    plt.close('all')


def test_Logo():

    # df inputs that successfully execute when entered into Logo.
    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)
    good_prob_df = logomaker.get_example_matrix('ss_probability_matrix', print_description=False)
    random_df = pd.DataFrame(np.random.randint(0, 3, size=(10, 4)), columns=list('ACGT'))

    # df inputs that fail when entered into Logo.
    bad_df1 = 'x'

    # test parameter df
    test_parameter_values(func=logomaker.Logo, var_name='df',fail_list=[bool_fail_list,bad_df1],
                          success_list=[good_crp_df,random_df])

    # test parameter colors
    test_parameter_values(func=logomaker.Logo, var_name='color_scheme', fail_list=['x','bad_color_name',3],
                          success_list=['classic', 'grays', 'charge','salmon'],
                          df=good_crp_df)

    # test parameter font names
    test_parameter_values(func=logomaker.Logo, var_name='font_name', fail_list=[True, None, good_crp_df],
                          success_list=['DejaVu Sans', 'Arial Rounded MT Bold', 'Times New Roman'],
                          df=good_crp_df)

    # test parameter stack_order
    test_parameter_values(func=logomaker.Logo, var_name='stack_order', fail_list=['incorrect argument', 0.0, None],
                          success_list=['small_on_top', 'big_on_top', 'fixed'],
                          df=good_crp_df)

    # test parameter center_values
    test_parameter_values(func=logomaker.Logo, var_name='center_values', fail_list=['incorrect argument', 0.0, None],
                          success_list=[True, False], df=good_crp_df)

    # test parameter baseline_width
    test_parameter_values(func=logomaker.Logo, var_name='baseline_width', fail_list=['incorrect argument', -0.1, None],
                          success_list=[0,0.5,3], df=good_crp_df)

    # test parameter flip_below
    test_parameter_values(func=logomaker.Logo, var_name='flip_below', fail_list=['incorrect argument', -0.1, None],
                          success_list=[True, False], df=good_crp_df)

    # test parameter shade_below
    test_parameter_values(func=logomaker.Logo, var_name='shade_below',
                          fail_list=['incorrect argument', -0.1, 1.4, None],
                          success_list=[0, 0.0, 0.5, 1, 1.0], df=good_crp_df)

    # test parameter fade_below
    test_parameter_values(func=logomaker.Logo, var_name='fade_below',
                          fail_list=['incorrect argument', -0.1, 1.4, None],
                          success_list=[0, 0.0, 0.5, 1, 1.0], df=good_crp_df)

    # test parameter fade_probabilities
    test_parameter_values(func=logomaker.Logo, var_name='fade_probabilities',
                          fail_list=['incorrect argument', -0.1, 1.4, None],
                          success_list=[True, False], df=good_prob_df)

    # test parameter vsep
    test_parameter_values(func=logomaker.Logo, var_name='vsep',
                          fail_list=['incorrect argument', -0.1, None],
                          success_list=[0.0, 0,0.3,10], df=good_crp_df)

    # test parameter vsep
    # TODO: note that a value of True/False is still causing a logo to be drawn, eventhough draw_now = False
    test_parameter_values(func=logomaker.Logo, var_name='show_spines',
                          fail_list=['incorrect argument', -0.1],
                          success_list=[None, True, False], df=good_crp_df)

    # test parameter zorder. Need to review zorder's input check in Logo
    test_parameter_values(func=logomaker.Logo, var_name='zorder',
                          fail_list=['incorrect argument'],
                          success_list=[0, 1, 3], df=good_crp_df)

    # test parameter figsize
    test_parameter_values(func=logomaker.Logo, var_name='figsize',
                          fail_list=['incorrect argument', -0.1, [-1,-1],[-1],[0,0],['x','y'],(1,2,3)],
                          success_list=[(10, 2.5),[5,5]], df=good_crp_df)

    # validate ax
    _, temp_ax = plt.subplots(figsize=[3, 3])
    test_parameter_values(func=logomaker.Logo, var_name='ax',
                          fail_list=['x',1,True],
                          success_list=[temp_ax, None], df=good_crp_df)


def test_Logo_style_glyphs():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter color_scheme
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs,var_name='color_scheme',
                          fail_list=['bad_color_scheme',1,True],
                          success_list=['classic','gray','salmon'])

    # TODO: how should we test kwargs for this (and other) methods?


def test_Logo_fade_glyphs_in_probability_logo():

    good_prob_df = logomaker.get_example_matrix('ss_probability_matrix', print_description=False)

    # test parameter v_alpha0
    test_parameter_values(func=logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, var_name='v_alpha0',
                          fail_list=[-1,1.1,1,1.0,'xxx',True], success_list=[0,0.0,0.999,0.5])

    # TODO: a value of True v_alpha_1 works now, this should probably be fixed.
    # test parameter v_alpha1
    test_parameter_values(func=logomaker.Logo(good_prob_df).fade_glyphs_in_probability_logo, var_name='v_alpha1',
                          fail_list=[1.1, -1, 'xxx'], success_list=[0.999, 0.5,1.0])


def test_Logo_style_glyphs_below():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter color
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs_below, var_name='color',
                          fail_list=[0,'xxx',True,[0,0,-1]], success_list=['red', [1,1,1], [0,0,0],[0.1,0.2,0.3], None])

    # test parameter alpha
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs_below, var_name='alpha',
                          fail_list=[-1,'x',1.1], success_list=[0,1,0.5])

    # test parameter shade
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs_below, var_name='shade',
                          fail_list=[-1,'x',1.1], success_list=[0,1,0.5])

    # test parameter fade
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs_below, var_name='fade',
                          fail_list=[-1,'x',1.1], success_list=[0,1,0.5])

    # test parameter flip
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs_below, var_name='flip',
                          fail_list=bool_fail_list, success_list=bool_success_list)



def test_Logo_style_single_glyph():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter p
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_single_glyph, var_name='p',
                          fail_list=[-1,'x',1.1,10000], success_list=[0,1,10],c='A')

    # test parameter c
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_single_glyph, var_name='c',
                          fail_list=[-1, 'x', 1.1], success_list=['A','C','G','T'], p=1)


def test_Logo_style_glyphs_in_sequence():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # randomly make up a sequence of the correct length
    test_good_sequence = np.random.choice(['A', 'C', 'G', 'T'], size=26, p=[0.25, 0.25, 0.25, 0.25])
    test_good_sequence = "".join(test_good_sequence)

    # randomly make up a sequence of the incorrect length
    test_bad_sequence = np.random.choice(['A', 'C', 'G', 'T'], size=10, p=[0.25, 0.25, 0.25, 0.25])
    test_bad_sequence = "".join(test_bad_sequence)


    # test parameter sequence
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_glyphs_in_sequence, var_name='sequence',
                          fail_list=[-1, 'x', 1.1, test_bad_sequence], success_list=[test_good_sequence])


def test_Logo_highlight_position():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter p
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position, var_name='p',
                          fail_list=['x',1.5], success_list=[0, 1, 10])

def test_Logo_highlight_position_range():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter pmin
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='pmin',
                          fail_list=['x', 20], success_list=[0, 1, 10], pmax=15)

    # test parameter pmax
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='pmax',
                          fail_list=['x', 1], success_list=[5.5, 6, 10], pmin=5)

    # test parameter padding
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='padding',
                          fail_list=['x', -1], success_list=[-0.5, 0, 10], pmin=5,pmax=10)

    # test parameter color
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='color',
                          fail_list=['x', 1,True,'wrong_color'], success_list=['pink','red',[1,1,1]], pmin=5, pmax=10)

    # test parameter edgecolor
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='edgecolor',
                          fail_list=['x', 1, True, 'wrong_color'], success_list=[None,'pink', 'red', [1, 1, 1]], pmin=5,
                          pmax=10)

    # test parameter floor
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='floor',
                          fail_list=['x', 10], success_list=[-1,1,None],
                          pmin=5,
                          pmax=10)

    # test parameter ceiling
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='ceiling',
                          fail_list=['x', -10], success_list=[-1, 1, None],
                          pmin=5,
                          pmax=10)

    # test parameter zorder. Note that a value of False passes for this parameter. This should be fixed.
    test_parameter_values(func=logomaker.Logo(good_crp_df).highlight_position_range, var_name='zorder',
                          fail_list=['x', None], success_list=[-1, 0.5, 1],
                          pmin=5,
                          pmax=10)


def test_Logo_draw_baseline():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter zorder
    test_parameter_values(func=logomaker.Logo(good_crp_df).draw_baseline, var_name='zorder',
                          fail_list=['x', None], success_list=[-1, 0.5, 1])

    # test parameter color
    test_parameter_values(func=logomaker.Logo(good_crp_df).draw_baseline, var_name='color',
                          fail_list=['x', 1, True, 'wrong_color'], success_list=['pink', 'red', [1, 1, 1]])

    # test parameter linewidth
    test_parameter_values(func=logomaker.Logo(good_crp_df).draw_baseline, var_name='linewidth',
                          fail_list=['x', -1, '1', None], success_list=[0,1,1.5,2])


def test_Logo_style_xticks():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter anchor
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_xticks, var_name='anchor',
                          fail_list=['x', None, 0.5, 1.0], success_list=[-1, 1, 2])

    # test parameter spacing
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_xticks, var_name='spacing',
                          fail_list=['x', None, 0, 0.5], success_list=[1, 2])

    # test parameter fmt
    # TODO: fmt = 'x' seems to passing in the following. This should be fixed.
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_xticks, var_name='fmt',
                          fail_list=[None, 0], success_list=['%d'])

    # test parameter rotation
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_xticks, var_name='rotation',
                          fail_list=[None, 'x'], success_list=[-12,0,1.4,200])

def test_Logo_style_spines():

    good_crp_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)

    # test parameter anchor
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_spines, var_name='spines',
                          fail_list=['x', None, 0.5, 1.0,'top'],
                          success_list=[('top', 'bottom', 'left', 'right'),
                                        ['top', 'bottom', 'left', 'right'],
                                        ('top', 'bottom', 'left'),
                                        ('top', 'bottom'),
                                        ['top']])

    # test parameter visible
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_spines, var_name='visible',
                          fail_list=bool_fail_list, success_list=bool_success_list)

    # test parameter color
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_spines, var_name='color',
                          fail_list=['wrong_color',3,[0.5,0.5,0.5,0.5]], success_list=['black','green',[0.4,0.5,1.0]])

    # test parameter linewidth
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_spines, var_name='linewidth',
                          fail_list=['xxx',-1], success_list=[0.0,0,1])

    # test parameter bounds
    test_parameter_values(func=logomaker.Logo(good_crp_df).style_spines, var_name='bounds',
                          fail_list=['xxx', -1], success_list=[None,[0,1]])

def test_transform_matrix():

    good_crp_weight_df = logomaker.get_example_matrix('crp_energy_matrix', print_description=False)
    good_crp_counts_df = logomaker.get_example_matrix('crp_counts_matrix', print_description=False)

    # test parameter df
    test_parameter_values(func=logomaker.transform_matrix, var_name='df',
                          fail_list=['x',good_crp_weight_df,None], success_list=[good_crp_counts_df],
                          from_type='counts', to_type='probability')

    test_parameter_values(func=logomaker.transform_matrix, var_name='df',
                          fail_list=[], success_list=[good_crp_counts_df],
                          from_type='counts', to_type='weight')

    test_parameter_values(func=logomaker.transform_matrix, var_name='df',
                          fail_list=[], success_list=[good_crp_counts_df],
                          from_type='counts', to_type='information')

    # test parameter center_values
    test_parameter_values(func=logomaker.transform_matrix, var_name='center_values',
                          fail_list=bool_fail_list, success_list=bool_success_list,
                          df= good_crp_counts_df)

    # test parameter normalize_values
    test_parameter_values(func=logomaker.transform_matrix, var_name='normalize_values',
                          fail_list=bool_fail_list, success_list=bool_success_list,
                          df=good_crp_counts_df)

    # test parameter from_type
    test_parameter_values(func=logomaker.transform_matrix, var_name='from_type',
                          fail_list=[1,'x',None], success_list=['counts'],
                          df=good_crp_counts_df,to_type='probability')

    # test parameter to_type
    test_parameter_values(func=logomaker.transform_matrix, var_name='to_type',
                          fail_list=[1, 'x', None], success_list=['probability','weight','information'],
                          df=good_crp_counts_df, from_type='counts')

    # test parameter background
    test_parameter_values(func=logomaker.transform_matrix, var_name='background',
                          fail_list=[1, 'x', [-1,1,1,1]], success_list=[None, [0.25,0.25,0.25,0.25]],
                          df=good_crp_counts_df, from_type='counts', to_type='information')

    # test parameter pseudocount
    test_parameter_values(func=logomaker.transform_matrix, var_name='pseudocount',
                          fail_list=[None, 'x', -1], success_list=[0,1,10],
                          df=good_crp_counts_df, from_type='counts', to_type='probability')


def test_sequence_to_matrix():

    # test parameter seq
    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='seq',
                          fail_list=[None, 3, True], success_list=['ACGT', '!@#$', 'logomaker'])

    # test parameter cols
    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='cols',
                          fail_list=[0, True], success_list=[None, ['A','C','G','T']],seq='ACGTACGT')

    # test parameter alphabet
    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='alphabet',
                          fail_list=[0, True, 'xxx'], success_list=[None,'dna'], seq='ACGTACGT')

    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='alphabet',
                          fail_list=[], success_list=['rna'], seq='ACGUACGU')

    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='alphabet',
                          fail_list=[], success_list=['protein'], seq='LMWA')

    # test parameter is_iupac
    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='is_iupac',
                          fail_list=bool_fail_list, success_list=bool_success_list, seq='ACGTACGT')

    # test parameter to_type
    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='to_type',
                          fail_list=[0, True, 'xxx'], success_list=['probability','weight','information'],
                          seq='ACGTACGT')

    # test parameter center_weights
    test_parameter_values(func=logomaker.sequence_to_matrix, var_name='center_weights',
                           fail_list=bool_fail_list, success_list=bool_success_list,
                           seq='ACGTACGT',to_type='weight')


def test_alignment_to_matrix():

    # get sequences from file
    with logomaker.open_example_datafile('crp_sites.fa', print_description=False) as f:
        raw_seqs = f.readlines()
    seqs = [seq.strip() for seq in raw_seqs if ('#' not in seq) and ('>') not in seq]

    # test parameter sequences
    test_parameter_values(func=logomaker.alignment_to_matrix,var_name='sequences',
                          fail_list = [0,'x',['AACCT','AACGATA']], success_list = [seqs,['ACA','GGA']])

    # test parameter counts
    # TODO: need to find a working example for counts other than None
    test_parameter_values(func=logomaker.alignment_to_matrix, var_name='counts',
                          fail_list=[0, 'x', -1], success_list=[None],sequences=['ACA', 'GGA'])

    # test parameter to_type
    test_parameter_values(func=logomaker.alignment_to_matrix, var_name='to_type',
                          fail_list=[0, True, 'xxx'], success_list=['counts','probability', 'weight', 'information'],
                          sequences=seqs)

    # test parameter background
    test_parameter_values(func=logomaker.alignment_to_matrix, var_name='background',
                          fail_list=[1, 'x'], success_list=[None, [0.25,0.25,0.25,0.25]],
                          sequences=seqs)

    # test parameter characters_to_ignore
    test_parameter_values(func=logomaker.alignment_to_matrix, var_name='characters_to_ignore',
                          fail_list=[1, 0.5,True,None], success_list=['A','C','G'],
                          sequences=seqs)

    # test parameter center_weights
    test_parameter_values(func=logomaker.alignment_to_matrix, var_name='center_weights',
                          fail_list=bool_fail_list, success_list=bool_success_list,
                          sequences=seqs)

    # test parameter pseudocount
    test_parameter_values(func=logomaker.alignment_to_matrix, var_name='pseudocount',
                          fail_list=[None, 'x', -1], success_list=[0, 1, 10],
                          sequences=seqs)


def test_saliency_to_matrix():

    # load saliency data
    with logomaker.open_example_datafile('nn_saliency_values.txt', print_description=False) as f:
        saliency_data_df = pd.read_csv(f, comment='#', sep='\t')

    # test parameter seq
    test_parameter_values(func=logomaker.saliency_to_matrix, var_name='seq',
                          fail_list=[None, 'x', -1], success_list=[saliency_data_df['character']],
                          values=saliency_data_df['value'])

    # test parameter values
    test_parameter_values(func=logomaker.saliency_to_matrix, var_name='values',
                          fail_list=[None, 'x', -1], success_list=[saliency_data_df['value']],
                          seq=saliency_data_df['character'])

    # test parameter cols
    test_parameter_values(func=logomaker.saliency_to_matrix, var_name='cols',
                          fail_list=[-1,'x'], success_list=[None,['A','C','G','T']],
                          seq=saliency_data_df['character'],values=saliency_data_df['value'])

    # test parameter alphabet
    test_parameter_values(func=logomaker.saliency_to_matrix, var_name='alphabet',
                          fail_list=[0, True, 'xxx'], success_list=[None,'dna'],
                          seq=saliency_data_df['character'], values=saliency_data_df['value'])


def test_Glyph():

    fig, ax = plt.subplots(figsize=[7, 3])

    # set bounding box
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])

    # test parameter p
    # TODO: need to fix fail_list bugs for parameter p
    test_parameter_values(func=logomaker.Glyph,var_name='p',
                          fail_list=[],success_list=[0,1,10, 0.5],
                          c='A', ax=ax, floor=0, ceiling=1)

    # test parameter c
    test_parameter_values(func=logomaker.Glyph, var_name='c',
                          fail_list=[0,0.1,None], success_list=['A','C','x'],
                          p=1, ax=ax, floor=0, ceiling=1)

    # test parameter floor
    test_parameter_values(func=logomaker.Glyph, var_name='floor',
                          fail_list=['x',2, None], success_list=[0,0.1,0.5],
                          p=1, ax=ax, ceiling=1, c='A')

    # test parameter ceiling
    test_parameter_values(func=logomaker.Glyph, var_name='ceiling',
                          fail_list=['x', -1, None], success_list=[0, 0.1, 0.5],
                          p=1, ax=ax, floor=0, c='A')

    # test parameter ax
    test_parameter_values(func=logomaker.Glyph, var_name='ax',
                          fail_list=['x', -1], success_list=[None,ax],
                          p=1, ceiling=1, floor=0, c='A')

    # test parameter width
    test_parameter_values(func=logomaker.Glyph, var_name='width',
                          fail_list=['x', -1], success_list=[0.1,1,10],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter vpad
    test_parameter_values(func=logomaker.Glyph, var_name='vpad',
                          fail_list=['x', -1, None, 1], success_list=[0, 0.9, 0.5],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter font_name
    test_parameter_values(func=logomaker.Glyph, var_name='font_name',
                          fail_list=[-1], success_list=['DejaVu Sans'],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter font_weight
    test_parameter_values(func=logomaker.Glyph, var_name='font_weight',
                          fail_list=[-1,'xxx'], success_list=['bold',5, 'normal'],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter color
    test_parameter_values(func=logomaker.Glyph, var_name='color',
                          fail_list=[-1, 'xxx'], success_list=['red', [1,1,1], [0,0,0],[0.1,0.2,0.3]],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter edgecolor
    test_parameter_values(func=logomaker.Glyph, var_name='edgecolor',
                          fail_list=[-1, 'xxx'], success_list=['black', [1, 1, 1], [0, 0, 0], [0.1, 0.2, 0.3]],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter edgewidth
    test_parameter_values(func=logomaker.Glyph, var_name='edgewidth',
                          fail_list=['x', -1, None], success_list=[0, 0.9, 0.5],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter dont_stretch_more_than
    test_parameter_values(func=logomaker.Glyph, var_name='dont_stretch_more_than',
                          fail_list=[-1, None], success_list=['E','A'],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter flip
    test_parameter_values(func=logomaker.Glyph, var_name='flip',
                          fail_list=bool_fail_list, success_list=bool_success_list,
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter mirror
    test_parameter_values(func=logomaker.Glyph, var_name='mirror',
                          fail_list=bool_fail_list, success_list=bool_success_list,
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter zorder
    test_parameter_values(func=logomaker.Glyph, var_name='zorder',
                          fail_list=['x'], success_list=[0,0.5,1,5],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter alpha
    test_parameter_values(func=logomaker.Glyph, var_name='alpha',
                          fail_list=[1.1, -1, 'xxx'], success_list=[0.999, 0.5, 1.0],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)

    # test parameter figsize
    test_parameter_values(func=logomaker.Glyph, var_name='figsize',
                          fail_list=['incorrect argument', ['x', 'y'], (1, 2, 3),[1,-1]],
                          success_list=[(10, 2.5), [5, 5]],
                          p=1, ceiling=1, floor=0, c='A', ax=ax)


def test_logomaker_get_data_methods():

    # testing parameter name in get_example_matrix
    test_parameter_values(func=logomaker.get_example_matrix, var_name='name',
                          fail_list = ['wrong argument', -1], success_list=['crp_energy_matrix','ww_counts_matrix'], print_description=False)

    # testing parameter print_description in get_example_matrix
    test_parameter_values(func=logomaker.get_example_matrix, var_name='print_description',
                          fail_list=bool_fail_list, success_list=bool_success_list, name='crp_energy_matrix')

    # testing parameter name in open_example_datafile
    test_parameter_values(func=logomaker.open_example_datafile, var_name='name',
                          fail_list=['wrong argument', -1], success_list=['nn_saliency_values.txt', 'ss_sequences.txt'], print_description=False)

    # testing parameter print_description in open_example_datafile
    test_parameter_values(func=logomaker.open_example_datafile, var_name='print_description',
                          fail_list=bool_fail_list, success_list=bool_success_list,
                          name='nn_saliency_values.txt')

def test_demo():

    test_parameter_values(func=logomaker.demo,var_name='name',
                          fail_list=[0, True, 'xxx'], success_list=['crp', 'fig1b'])

    plt.close('all')


def run_tests():
    """
    Run all Logomaker functional tests. There are 547 tests as of 14 May 2019.

    parameters
    ----------
    None.

    return
    ------
    None.
    """

    # run tests for the Logo class and it's helper methods
    test_Logo()
    test_Logo_style_glyphs()
    test_Logo_fade_glyphs_in_probability_logo()
    test_Logo_style_glyphs_below()
    test_Logo_style_single_glyph()
    test_Logo_style_glyphs_in_sequence()
    test_Logo_highlight_position()
    test_Logo_highlight_position_range()
    test_Logo_draw_baseline()
    test_Logo_style_xticks()
    test_Logo_style_spines()

    # run tests for the methods in the matrix module
    test_transform_matrix()
    test_sequence_to_matrix()
    test_alignment_to_matrix()
    test_saliency_to_matrix()

    # run tests for the Glyph class
    test_Glyph()

    #test_demo()
    test_logomaker_get_data_methods()

if __name__ == '__main__':
    run_tests()

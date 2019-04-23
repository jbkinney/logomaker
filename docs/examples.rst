Examples
========

This section illustrates a variety of logos drawn using logomaker and focuses on customization and styling.
We begin by importing useful packages::

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import logomaker as lm


:math:`\Delta \Delta G` CRP Weight Logo
-----------------------------------------

We begin by loading a crp energy matrix model::

    # load crp dataframe
    crp_df = pd.read_csv('matrices/crp_energy_matrix.txt', delim_whitespace=True, index_col=0)
    crp_df.head()

+-----+---------+--------+--------+---------+
| pos | A       | C      | G      | T       |
+=====+=========+========+========+=========+
| 0   | -0.2975 | 0.2525 | 0.1525 | -0.1075 |
+-----+---------+--------+--------+---------+
| 1   | -0.4700 | 0.4500 | 0.1800 | -0.1600 |
+-----+---------+--------+--------+---------+
| 2   | -0.4475 | 0.5125 | 0.2725 | -0.3375 |
+-----+---------+--------+--------+---------+
| 3   | -0.3675 | 0.4625 | 0.4825 | -0.5775 |
+-----+---------+--------+--------+---------+
| 4   | -0.0975 | 0.2325 | 0.0925 | -0.2275 |
+-----+---------+--------+--------+---------+

This energy matrix was determined by Kinney *et. al.* in [#sortseq2010]_. The following illustration of
the CRP energy logo uses shading and fading of characters below the x-axis This emphasizes characters
with positive values. The shade and fade features are set by using the keyword arguments **shade_below**
and **fade_below** in the constructor for Logo. Additionally, styling options for spines, ticks,
and lables is also demonstrated::

    ### Style CRP panel

    logo = lm.Logo(-crp_df,
                   shade_below=.5,
                   fade_below=.5,
                   font_name='Arial Rounded MT Bold')

    logo.style_spines(visible=False)
    logo.style_spines(spines=['left','bottom'], visible=True)
    logo.ax.set_ylabel("energy ($k_B T$)", labelpad=-1)
    logo.style_xticks(rotation=90, fmt='%d', anchor=0)
    logo.ax.xaxis.set_ticks_position('none')
    logo.ax.set_ylim([-2,2])
    logo.ax.xaxis.set_tick_params(pad=-1)

.. image:: _static/examples_images/1B.png

5' Splice Sites in the Human Genome
-----------------------------------

We now illustrate a probability generated from 202,764 5'ss sequences in the human transcriptome.
The data are obtained from [#wong2018]_.

::

    ss_df = pd.read_csv('matrices/ss_prob_matrix.txt', delim_whitespace=True, index_col=0)
    ss_df.head()


+-----+----------+----------+----------+----------+
| pos | A        | C        | G        | U        |
+=====+==========+==========+==========+==========+
| 0   | 0.325785 | 0.359893 | 0.188348 | 0.125974 |
+-----+----------+----------+----------+----------+
| 1   | 0.630292 | 0.109679 | 0.120055 | 0.139974 |
+-----+----------+----------+----------+----------+
| 2   | 0.101195 | 0.027272 | 0.799193 | 0.072340 |
+-----+----------+----------+----------+----------+
| 3   | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
+-----+----------+----------+----------+----------+
| 4   | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
+-----+----------+----------+----------+----------+

::

    logo = lm.Logo(ss_df,
                   width=.8,
                   vpad=.05,
                   fade_probabilities=True,
                   stack_order='small_on_top',
                   color_scheme='dodgerblue',
                   font_name='Rosewood Std')

    logo.ax.set_xticks(range(len(ss_df)))
    logo.ax.set_xticklabels('%+d'%x for x in [-3,-2,-1,1,2,3,4,5,6])
    logo.style_spines(spines=['left', 'right'], visible=False)
    logo.ax.set_yticks([0,.5,1])
    logo.ax.axvline(2.5, color='k', linewidth=1, linestyle=':')
    logo.ax.set_ylabel('probability')

.. image:: _static/examples_images/1C.png

The dashed line indicates intron/exon boundaries. This example shows the use of the keyword argument
**fade_probabilities**; when True, the characters in each stack are assigned an alpha value equal to
their height. Stacking order of characters is also set by using the keyword argument **stack_order**:
if stack_order =  'small_on_top', glyphs are stacked away from x-axis in order of decreasing absolute value.
**vpad** allows whitespace to be set above and below each character.

References
----------

.. [#sortseq2010] Kinney JB, Murugan A, Callan CG, Cox EC. 2010. `Using deep sequencing to characterize the biophysical mechanism of a transcriptional regulatory sequence`. Proc Natl Acad Sci USA 107:9158-9163 :download:`PDF <sortseq2010.pdf>`.

.. [#wong2018] Wong MS, Kinney JB, Krainer AR. `Quantitative activity profile and context dependence of all 434 human 5' splice sites`. Mol Cell. 2018;71:1012-26 e3.
.. _examples:

Examples
========

As described in :ref:`quickstart`, the five logos shown in Figure 1 of Tareen and Kinney (2019) [#Tareen2019]_ can be generated using the function ``logomaker.demo``. Here we describe each of these logos, as well as the snippets of code used to generate them. All snippets shown below are designed for use within an iPython Jupyter Notebook, and assume that the following header cell has already been run. ::

    # standard imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # displays logos inline within the notebook;
    # remove if using a python interpreter instead
    %matplotlib inline

    # logomaker import
    import logomaker

CRP Energy Logo
---------------

The following code creates an energy logo for the *E. coli* transcription factor CRP. The energy matrix illustrated by this logo was reported by Kinney et. al. (2010) [#sortseq2010]_ based on the analysis of a massively parallel reporter assay. This energy matrix is included with Logomaker as example data, and is loaded here by calling ``logomaker.get_example`` with the argument ``'crp_energy_matrix'``. A Logo object named ``crp_logo`` is then created using the ``shade_below``, ``fade_below``, and ``font_name`` styling arguments. Subsequent styling is then performed using the Logo object methods ``style_spines`` and ``style_xticks``. Additional styling is also performed using methods of ``crp_logo.ax``, the matplotlib Axes object on which the logo is drawn. ::

    # load crp energy matrix
    crp_df = -logomaker.get_example_matrix('crp_energy_matrix',
                                            print_description=False)

    # create Logo object
    crp_logo = logomaker.Logo(crp_df,
                              shade_below=.5,
                              fade_below=.5,
                              font_name='Arial Rounded MT Bold')

    # styling using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # styling using Axes methods
    crp_logo.ax.set_ylabel("$-\Delta \Delta G$ (kcal/mol)", labelpad=-1)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)

.. image:: _static/examples_images/crp_energy_logo.png

5' Splice Sites Logo
--------------------


We now illustrate a probability logo computed from all annotated 5' splices sites in the human genome.
The data are obtained from [#frankish2019]_. The dashed line indicates intron/exon boundaries.
This example shows the use of the keyword argument **fade_probabilities**; when set to True, the characters in each
stack are assigned an alpha (representing transparency) value equal to their height. Stacking order of
characters is also set by using the keyword argument **stack_order**: if stack_order =  'small_on_top', glyphs
are stacked away from x-axis in order of decreasing absolute value. **vpad** allows whitespace to be set
above and below each character.::

    # load ss probability matrix
    ss_df = logomaker.get_example_matrix('ss_probability_matrix',
                                  print_description=False)

    # create and style logo
    logo = logomaker.Logo(ss_df,
                   width=.8,
                   vpad=.05,
                   fade_probabilities=True,
                   stack_order='small_on_top',
                   color_scheme='dodgerblue',
                   font_name='Rosewood Std')
    logo.ax.set_xticks(range(len(ss_df)))
    logo.ax.set_xticklabels('%+d'%x for x in [-3, -2, -1, 1, 2, 3, 4, 5, 6])
    logo.style_spines(spines=['left', 'right'], visible=False)
    logo.ax.set_yticks([0, .5, 1])
    logo.ax.axvline(2.5, color='k', linewidth=1, linestyle=':')
    logo.ax.set_ylabel('probability')

.. image:: _static/examples_images/ss_probability_logo.png

Protein Sequence Logo: WW domain
--------------------------------

We now show a logo drawn from the WW domain alignment [#WWdomain]_, and highlight the eponymous
positions of this alignment. To do the highlights, we use the Logomaker method *highlight_position*. Note that
the color scheme is part of a number of default color dictionaries Logomaker has. The list of available color schemes
can be viewed by calling `logomaker.list_color_schemes()`. The user can choose named colors in matplotlib and also
pass in custom color dictionaries::

    # load ww information matrix
    ww_df = logomaker.get_example_matrix('ww_information_matrix',
                                  print_description=False)

    # create and style logo
    logo = logomaker.Logo(ww_df,
                   font_name='Stencil Std',
                   color_scheme='NajafabadiEtAl2017',
                   vpad=.1,
                   width=.8)
    logo.ax.set_ylabel('information (bits)')
    logo.style_xticks(anchor=0, spacing=5, rotation=45)
    logo.highlight_position(p=4, color='gold', alpha=.5)
    logo.highlight_position(p=26, color='gold', alpha=.5)
    logo.ax.set_xlim([-1, len(ww_df)])

.. image:: _static/examples_images/ww_information_logo.png

Autonomously Replicating Sequence (ARS) Logo
--------------------------------------------

We demonstrate an enrichment logo representing the effects mutations have on replication efficiency within the ARS1
replication origin of S. cerevisiae. These data (unpublished) were collected by Justin B. Kinney from a mutARS-seq
experiment analogous to the one reported by [#Liachko2013]_. We use the function *highlight_position_range* to
highlight a range of positions indicating the A (lightcyan), the B1 (honeydew), B2 (lavenderblush) elements for the ARS.::


    # load ars matrix
    ars_df = logomaker.get_example_matrix('ars_enrichment_matrix',
                                  print_description=False)

    # load ars wt sequence
    with logomaker.open_example_datafile('ars_wt_sequence.txt',
                                  print_description=False) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines if '#' not in l]
        ars_seq = ''.join(lines)

    # trim ars matrix and sequence
    start = 10
    stop = 100
    ars_df = ars_df.iloc[start:stop, :]
    ars_df.reset_index(inplace=True, drop=True)
    ars_seq = ars_seq[start:stop]

    # create and style logo
    logo = logomaker.Logo(ars_df,
                   color_scheme='dimgray',
                   font_name='Luxi Mono')
    logo.style_glyphs_in_sequence(sequence=ars_seq, color='darkorange')
    logo.style_spines(visible=False)
    logo.ax.set_ylim([-4, 4])
    logo.ax.set_ylabel('$\log_2$ enrichment', labelpad=0)
    logo.ax.set_yticks([-4, -2, 0, 2, 4])
    logo.ax.set_xticks([])
    logo.highlight_position_range(pmin=7, pmax=22, color='lightcyan')
    logo.highlight_position_range(pmin=33, pmax=40, color='honeydew')
    logo.highlight_position_range(pmin=64, pmax=81, color='lavenderblush')

.. image:: _static/examples_images/ars_enrichment_logo.png

Saliency Logo
-------------

Saliency maps of deep neural networks accentuate important nucleotides. We adapt a saliency logo from [#Jaganathan]_
representing the importance of nucleotides in the vicinity of U2SUR exon 9, as predicted by a deep neural network
model of splice site selection. This example demonstrates how Logomaker is able to leverage functionality
from `matplotlib <https://matplotlib.org/>`_, thus allowing the user to customize their logos however much they want
(reproduced with author permission)::

    # load saliency matrix
    saliency_df = logomaker.get_example_matrix('nn_saliency_matrix',
                                        print_description=False)

    # create and style saliency logo
    logo = logomaker.Logo(saliency_df)
    ax = logo.ax
    logo.style_spines(visible=False)
    logo.style_spines(spines=['left'], visible=True, bounds=[0, .75])
    ax.set_xlim([20, 115])
    ax.set_yticks([0, .75])
    ax.set_yticklabels(['0', '0.75'])
    ax.set_xticks([])
    ax.set_ylabel('        saliency', labelpad=-1)

    # draw gene
    exon_start = 55-.5
    exon_stop = 90+.5
    y = -.2
    ax.set_ylim([-.3, .75])
    ax.axhline(y, color='k', linewidth=1)
    xs = np.arange(-3, len(saliency_df),10)
    ys = y*np.ones(len(xs))
    ax.plot(xs, ys, marker='4', linewidth=0, markersize=5, color='k')
    ax.plot([exon_start, exon_stop],
            [y, y], color='k', linewidth=10, solid_capstyle='butt')

.. image:: _static/examples_images/nn_saliency_logo.png

References
~~~~~~~~~~

.. [#Tareen2019] Tareen A, Kinney JB (2019) `Logomaker: beautiful sequence logos in Python <https://biorxiv.org>`_. bioRxiv doi:XXXX/XXXX.

.. [#sortseq2010] Kinney JB, Murugan A, Callan CG, Cox EC. 2010. `Using deep sequencing to characterize the biophysical mechanism of a transcriptional regulatory sequence`. Proc Natl Acad Sci USA 107:9158-9163 :download:`PDF <sortseq2010.pdf>`.

.. [#frankish2019] Frankish, A. et al. (2019). `GENCODE reference annotation for the human and mouse genomes.` Nucl Acids Res, 47(D1), D766–D773.

.. [#WWdomain] Fowler, D. M. et al. `High-resolution mapping of protein sequence-function relationships.` Nature Methods 7, 741–746 (2010).

.. [#Liachko2013] Liachko, I. et al. (2013). `High-resolution mapping, characterization, and optimization of autonomously replicating sequences in yeast.` Genome Res, 23(4), 698-704.

.. [#Jaganathan] Jaganathan, K. et al. (2019). `Predicting Splicing from Primary Sequence with Deep Learning.` Cell, 176(3), 535-548.e24.
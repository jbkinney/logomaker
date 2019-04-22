Tutorial
========

This tutorial provides a walk through of the logomaker functionality. Code snippets are provided for
easy reproduction.

Load Sequence Alignment
-----------------------

The input to Logomaker's Logo class is a pandas data frame in which columns represent
characters, rows represent positions, and values represent character heights. Multiple sequence alignments
are commonly found in fasta format.

::

    >0	caiFp	-41.5
    ATAAGCAGGATTTAGCTCACACTTAT
    >1	caiTp	-41.5
    AAAAATGTGATACCAATCACAGAATA
    >2	fixAp	-126.5
    ATATTGGTGATCCATAAAACAATATT

We can remove the non-sequence lines to obtain just the raw sequences::

    with open('sequences.fasta','r') as f:
        seqs = [l.strip() for l in f.readlines() if '>' not in l and len(l.strip())>0]

    ATAAGCAGGATTTAGCTCACACTTAT
    AAAAATGTGATACCAATCACAGAATA
    ATATTGGTGATCCATAAAACAATATT

Logomaker provides the method `alignment_to_matrix` which generates a counts dataframe ready to be input to
logomaker.

::

    counts_mat = logomaker.alignment_to_matrix(seqs)
    counts_mat.head()

returns:

+-----+-------+-------+------+------+
| pos | A     | C     | G    | T    |
+=====+=======+=======+======+======+
| 0   | 133.0 | 65.0  | 72.0 | 88.0 |
+-----+-------+-------+------+------+
| 1   | 147.0 | 46.0  | 58.0 | 107.0|
+-----+-------+-------+------+------+
| 2   | 166.0 | 26.0  | 38.0 | 128.0|
+-----+-------+-------+------+------+
| 3   | 164.0 | 28.0  | 43.0 | 123.0|
+-----+-------+-------+------+------+
| 4   | 133.0 | 45.0  | 47.0 | 133.0|
+-----+-------+-------+------+------+

Entering the counts matrix into The Logo class draws a counts logo::

    logomaker.Logo(counts_mat)

.. image:: _static/tutorial_images/counts_logo.png

Matrix Definitions
------------------

A matrix is defined by a set of textual characters, a set of numerical positions, and a numerical
quantity for every character-position pair. In what follows, we use the symbol :math:`i` to represent possible
positions, and the symbol :math:`c` (or :math:`c'`) to represent possible characters.

Within Python, each matrix is represented as a pandas data frame in which rows are indexed by positions
and columns are named using the character each represents. Logomaker can also read and write matrices as
text files. Any set of numerical positions can be used, as can any non-whitespace characters. Logomaker is
agnostic to the set of characters used.

Logos
-----

Any matrix can be represented as a logo in a straight-forward manner. Given a matrix,
a corresponding logo is drawn by stacking  the unique characters on top of one another
at each specified position. Each character at each position is drawn with a height given
by the value of the corresponding matrix element.

Characters with positive heights are stacked on top of one another starting from a baseline value of 0,
whereas characters with heights less than zero are stacked below one another starting from the baseline.
Logomaker provides the option of flipping characters with negative height upside down and/or darkening
the color with which such characters are drawn.

Built-in matrix and logo types
------------------------------

Although Logomaker will draw logos corresponding to any user-specified matrix, additional support
is provided for matrices of five specific types: counts matrix, probability matrix, enrichment matrix,
saliency matrix, and information matrix. Each matrix type directly or indirectly represents the marginal
statistics of a sequence alignment, and Logomaker can generate any one of these types of matrices from a
sequence alignment supplied by the user. Methods to interconvert matrices of these types are also provided.
Moreover, each of these five matrix types comes with its own logo style. These matrices and their corresponding
logos are described in detail below.

Counts matrix
-------------

A counts matrix represent the number of occurrences of each character at each position within a sequence
alignment (although the user can choose to exclude certain characters, e.g., '-' character representing gaps).
Specifically, a counts matrix has entries :math:`n_{ic}` that represent the number of occurrences of character
:math:`c` at position :math:`i`. These :math:`n_{ic}` values are all required to be greater or equal to zero. Counts logos are
assigned character heights corresponding to these :math:`n_{ci}` values. The y axis of such logos is labeled 'counts'
and extends from 0 to :math:`N`, where :math:`N` is the number of sequences in the alignment. Note that, Because certain
characters might be excluded when computing :math:`n_{ic}` from an alignment, it is possible to have
:math:`\sum_c n_{ic} < N` at some positions.

Probability matrix
------------------

A probability matrix represents the probability of observing each possible character at each possible position
within a certain type of sequence. Probability matrix elements are denoted by :math:`p_{ic}` and can be estimated
from a counts matrix via

:math:`p_{ic} = \frac{n_{ic} + \lambda}{\sum_{c'} n_{ic'} + C \lambda}`

where :math:`C` is the number of possible characters and :math:`\lambda` is a user-defined pseudocount.
A probability logo has heights given by these :math:`p_{ci}` values. The y axis extends from 0 to 1
and is labeled 'probability'.

Enrichment or Weight matrix
---------------------------

An enrichment matrix represent the relative likelihood of observing each character at each position
relative to some user-specified "background" model. Such matrices are sometimes referred to as position weight
matrices (PWMs) or position-specific scoring matrices (PSSMs). The elements :math:`w_{ic}` of an
enrichment matrix can be computed from a probability matrix (elements :math:`p_{ic}`) and a
background matrix (also a probability matrix but denoted :math:`b_{ic}`) using the formula

:math:`w_{ic} = \log_2 \frac{p_{ic}}{b_{ic}}`

This equation can be inverted to give :math:`p_{ic}`:

:math:`p_{ic} = \frac{b_{ic} 2^{w_{ic}}}{ \sum_{c'} b_{ic'} 2^{w_{ic'}} }`

where the denominator is included to explicitly enforce the the requirement that :math:`\sum_c p_{ic} = 1` at
every :math:`i`. Note that :math:`b_{ic}` will often not depend on $i$, but it does vary with :math:`i` in some cases, such as
computation of enrichment scores in deep mutational scanning experiments. Enrichment logos have heights given
by the :math:`w_{ci}` values, which can be either positive or negative. The y-axis is labeled ':math:`\log_2` enrichment'
by default.

Information matrix
------------------

Information logos were described in the original 1990 paper of Schneider and Stephens cite{Schneider},
and remain the most popular type of sequence logo. The entries :math:`I_{ic}`in the corresponding information matrices
are given by

:math:`I_{ci} = p_{ci} I_i,~~~I_i = \sum_c p_{ci} \log_2 \frac{p_{ci}}{b_{ci}}`

The position-dependent (but not character dependent) quantity :math:`I_i` is called the "information content"
of site :math:`i`, and the sum of these quantities, :math:`I = \sum_{i} I_i`, is the information content
of the entire matrix. These information values :math`I_{ic}`  are nonnegative and are said to be in units of
'bits' due to the use of :math:`\log_2` in Eq. ref{eq:prob_to_info}. A corresponding information logo is drawn
using these :math:`I_{ic}` values as character heights, as well as a y-axis labeled  'information (bits)'.

.. :math:`g_{ic} = \tilde{g}_{ic} - \frac{1}{C} \sum_{c'} \tilde{g}_{ic'} ,~~~\tilde{g}_{ic} = -\frac{1}{\alpha} \log \frac{p_{ic}}{b_{ic}}`

.. :math:`p_{ci} = \frac{b_{ci} \exp [ - \alpha g_{ci} ] }{\sum_{c'} b_{c'i} \exp[ - \alpha g_{c'i} ] }`


Make an counts logo
~~~~~~~~~~~~~~~~~~~
::

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    %matplotlib inline
    plt.ion()

    import logomaker

    # Load CRP binding site sequences
    with open('../data/crp_sites.fasta','r') as f:
        seqs = [l.strip() for l in f.readlines() if '>' not in l and len(l.strip())>0]

    # Preview sequences
    print('There are %d sequences, all of length %d'%(len(seqs), len(seqs[0])))
    seqs[:5]

There are 358 sequences, all of length 26:

|
    'ATAAGCAGGATTTAGCTCACACTTAT'
|
    'AAAAATGTGATACCAATCACAGAATA'
|
    'ATATTGGTGATCCATAAAACAATATT'
|
    'ATATTGGTGAGGAACTTAACAATATT'
|
    'GATTATTTGCACGGCGTCACACTTTG'


::

    # Alignment -> Counts matrix
    counts_df = logomaker.alignment_to_matrix(seqs)
    logo = logomaker.Logo(counts_df)


.. image:: _static/tutorial_images/counts_logo.png


Transform to a probability logo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    counts_df = logomaker.alignment_to_matrix(seqs, to_type='probability')

    logo = logomaker.Logo(counts_df,
                          color_scheme='purple',
                          fade_probabilities=True,
                          show_spines=False,
                          font_name='Impact')


.. image:: _static/tutorial_images/probability_logo.png


Make and information logo
~~~~~~~~~~~~~~~~~~~~~~~~~
::

    # Counts matrix -> Information matrix
    info_mat = logomaker.transform_matrix(counts_mat,
                                         background=background,
                                         from_type='counts',
                                         to_type='information')
    logomaker.Logo(info_mat)


.. image:: _static/tutorial_images/info_mat.png

Make an enrichment logo
~~~~~~~~~~~~~~~~~~~~~~~~
::

    # Convert seuqenes to weight matrix
    weight_df = logomaker.alignment_to_matrix(seqs, to_type='weight', center_weights=True)

    # preview weight matrix
    weight_df.head()

+-----+-----------+-----------+----------+----------+
| pos |    A      |    C      |     G    |     T    |
+=====+===========+===========+==========+==========+
| 0   |  0.201587 | 0.067196  | 0.067196 | 0.067196 |
+-----+-----------+-----------+----------+----------+
| 1   |  0.201587 | 0.067196  | 0.067196 | 0.067196 |
+-----+-----------+-----------+----------+----------+
| 2   | -0.10637  | -0.167351 | 0.13686  | 0.13686  |
+-----+-----------+-----------+----------+----------+
| 3   |  0.287282 | 0.041222  | -0.2039  | 0.44996  |
+-----+-----------+-----------+----------+----------+
| 4   | -0.056109 | -0.871858 | 0.344537 | 0.583429 |
+-----+-----------+-----------+----------+----------+


::

    fig, ax = plt.subplots(figsize=[6.5,1.5])

    # Create counts matrix
    logo = logomaker.Logo(weight_df,
                          ax=ax,
                          center_values=False,
                          fade_below=.7,
                          shade_below=.5,
                          font_name='Arial Rounded MT Bold')

    # Style axes
    logo.style_spines(visible=False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Tight layout
    plt.tight_layout()

    # Save as pdf
    out_file = out_prefix+'.pdf'
    fig.savefig(out_file)
    print('Done! Output written to %s.'%out_file)

.. image:: _static/tutorial_images/Example_CRP.png



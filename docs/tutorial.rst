.. _tutorial:

Tutorial
========

This tutorial provides a walk-through of Logomaker via a series of `Jupyter notebooks <https://jupyter.org/>`_.
Each notebook focuses on a different aspect of the Logomaker functionality. To run the notebooks, simply
download the notebooks to your local machine and run them in a local Jupyter server. These notebooks
assume that the user has `pip` installed Logomaker already; if not, please download the notebooks
to a directory where the Logomaker repository is cloned.

Lesson 1: Basic logo creation and styling
-----------------------------------------

Shows how to load data into Logomaker, draw a logo, and perform some basic logo styling.
`Access notebook on GitHub <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/1_simple_example_basic_styling.ipynb>`_.

Lesson 2: Logos from alignments
-------------------------------

Provides methods to convert multi-sequence alignments into dataframes that can subsequently be rendered as logos.
`Access notebook on GitHub <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/2_logos_from_alignment.ipynb>`_.


Lesson 3: Transform between logos of different types
----------------------------------------------------

Logomaker supports multiple different logo types and allows the user to transform
from one type of logo to another type. This lesson walks the user through
this functionality and gives mathematical descriptions of the built-in matrix types that Logomaker supports.
`Access notebook on GitHub <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/3_transform_between_logos_of_different_types.ipynb>`_.

Lesson 4: Saliency logos
------------------------

Saliency logos are used to represent important positions within a sequence as predicted by deep neural networks. This notebook
shows how to use Logomaker to draw saliency logos.
`Access notebook on GitHub <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/4_saliency_logos.ipynb>`_.

Lesson 5: Advanced styling
--------------------------

Describes the advanced logo styling functionality that Logomaker offers.
`Access notebook on GitHub <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/5_advanced_styling.ipynb>`_.

Lesson 6: Glyph objects
-----------------------

Shows how to render and customize individual characters, which in Logomaker are called ``Glyphs``.
`Access notebook on GitHub <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/6_glyph_objects.ipynb>`_.
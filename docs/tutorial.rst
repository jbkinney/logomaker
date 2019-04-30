Tutorial
========

This tutorial provides a walk through of Logomaker via a series of Jupyter notebooks. Each notebook
focuses on a different aspect of the Logomaker functionality. To run the notebooks, simply download the notebooks
to your local machine and run them in a local Jupyter server. These notebooks will assume that the user has `pip`
installed Logomaker already; in case you haven't download the notebooks to a directory where you have cloned the
Logomaker github repository.

Simple Example and Basic Styling
--------------------------------

The `Simple Example and Basic Styling <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/1_simple_example_basic_styling.ipynb>`_
tutorial notebook shows the user how to load data into Logomaker and draw a logo. This tutorial also introduces
some basic styling techniques.

Logos from alignment
---------------------

Logomaker provides methods to convert multiple sequence alignments to valid dataframes that can subsequently
be drawn. This functionality is covered in the
`Logos from alignment <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/2_logos_from_alignment.ipynb>`_
tutorial.

Transform between logos of different types
------------------------------------------

Logomaker supports multiple different logo types and allows the user to transform
from one type of logo multiple others. The
`Transform between logos of different types <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/3_transform_between_logos_of_different_types.ipynb>`_
tutorial walks the user through this functionality and gives mathematical descriptions of the built-in matrices Logomaker supports.

Saliency Logos
--------------

Saliency logos are used to represent important nucleotides as predicted by deep neural networks. The tutorial notebook
`saliency logos <https://github.com/jbkinney/logomaker/blob/master/logomaker/tutorials/4_saliency_logos.ipynb>`_
shows the user how to use Logomaker to draw saliency logos. This notebook contains two examples, (i) a saliency logo based on randomly
chosen saliency values and (ii) a saliency logo adapted from a publised paper.

Advanced Styling
----------------

Glyph
-----




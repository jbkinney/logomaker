import numpy as np
import pandas as pd
from six import string_types



class Matrix:
    """
    Container class for matrix models
    """

    def __init__(self, df):

        # Copy data frame
        df = df.copy().fillna(0.0).astype(float)

        # Check that columns are strings of at least length 1
        for c in df.columns:
            assert isinstance(c, string_types)
            assert len(c) >= 1

        # Keep only last non-whitespace character as column name
        df.columns = [col.strip()[-1] for col in df.columns]

        # Sort columns alphabetically
        df = df.reindex(sorted(df.columns), axis=1)

        # Record characters
        self.cs = list(df.columns)

        # Cast indices as integers
        df.index = df.index.astype(int)

        # Name index
        df.index.rename('pos', inplace=True)

        # Record positions
        self.ps = df.index.values

        # Record dataframe
        self.df = df

        # Determine type of characters
        # CONTINUE HERE


import pandas as pd
import os
from logomaker.src.error_handling import check, handle_errors

# load directory of file
matrix_dir = os.path.dirname(os.path.abspath(__file__)) \
             + '/../examples/matrices'

# List of supported distributions by name
VALID_MATRICES = ['.'.join(name.split('.')[:-1]) for name in
                  os.listdir(matrix_dir) if '.txt' in name]

# load directory of file
data_dir = os.path.dirname(os.path.abspath(__file__)) \
           + '/../../data'

# List of supported distributions by name
VALID_DATAFILES = [name for name in
                   os.listdir(data_dir) if
                   len(name.split('.')) >= 2 and
                   len(name.split('.')[0]) > 0]

@handle_errors
def list_example_matrices():
    """
    Return list of available matrices.
    """
    return VALID_MATRICES


@handle_errors
def list_example_datafiles():
    """
    Return list of available data files.
    """
    return VALID_DATAFILES


@handle_errors
def get_example_matrix(name=None, print_description=True):
    """
    Returns an example matrix from which a logo can be made.

    parameters
    ----------

    name: (None or str)
        Name of example matrix.

    print_description: (bool)
        If true, a description of the example matrix will be printed

    returns
    -------

    df: (data frame)
        A data frame containing an example matrix.
    """

    # check that dataset is valid
    check(name in list_example_matrices(),
          'Matrix "%s" not recognized. Please choose from: \n%s'
          % (name, '\n'.join([repr(x) for x in VALID_MATRICES])))

    # set file dataset
    file_name = '%s/%s.txt' % (matrix_dir, name)
    assert os.path.isfile(file_name), 'File %s does not exist!'%file_name

    # if user wants a description of the example matrx
    if print_description:
        print('Description of example matrix "%s":' % name)
        with open(file_name, 'r') as f:
            lines = f.readlines()
            lines = [l for l in lines if len(l)>0 and l[0] == '#']
            description = "".join(lines)
            print(description)

    # load data frame
    df = pd.read_csv(file_name, sep='\t', index_col=0, comment='#')

    # return data frame
    return df


@handle_errors
def open_example_datafile(name=None, print_description=True):
    """
    Returns a file handle to an example dataset

    parameters
    ----------

    name: (None or str)
        Name of example matrix.

    print_description: (bool)
        If true, a description of the example matrix will be printed

    returns
    -------

    f: (file handle)
        A handle to the requested file
    """

    # check that dataset is valid
    check(name in list_example_datafiles(),
          'Matrix "%s" not recognized. Please choose from: \n%s'
          % (name, '\n'.join([repr(x) for x in VALID_DATAFILES])))

    # set file dataset
    file_name = '%s/%s' % (data_dir, name)
    assert os.path.isfile(file_name), 'File %s does not exist!' % file_name

    # if user wants a description of the data file, provide it
    if print_description:
        print('Description of example matrix "%s":' % name)
        with open(file_name, 'r') as f:
            lines = f.readlines()
            lines = [l for l in lines if len(l)>0 and l[0] == '#']
            description = "".join(lines)
            print(description)

    # return file handle
    return open(file_name, 'r')

import re
import pandas as pd
import pdb

def load_meme(file_name,
              motif_name=None,
              motif_num=None,
              get_alphabet=False,
              get_background=False,
              get_names=False,
              get_dict=False,
              get_list=False,
              get_filecontents=False):
    file_contents = open(file_name).read()

    # Fix returns
    file_contents = re.sub('\r\n', '\n', file_contents)
    file_contents = re.sub('\n\r', '\n', file_contents)
    file_contents = re.sub('\r', '\n', file_contents)

    # Change tab to 4 spaces
    file_contents = re.sub('\t', '    ', file_contents)

    if get_filecontents:
        return file_contents

    # Try to parse standard alphabet
    meme_alphabet = re.compile('ALPHABET\s*=\s*(?P<alphabet>\S+)',
                               flags=re.MULTILINE)
    m = re.search(meme_alphabet, file_contents)
    if m:
        columns = list(m.group('alphabet'))
    else:
        # Otherwise, try to parse custom alphabet
        meme_alphabet = re.compile(
            'ALPHABET.*\n(?P<alphabet>(?:.*\n)+)END ALPHABET', re.MULTILINE)
        m = re.search(meme_alphabet, file_contents)
        if m:
            text = m.group('alphabet')
            columns = [l.strip()[0] for l in text.split('\n') if
                       len(l.strip()) > 0]
        else:
            print('Could not determine alphabet. Returning None.')
            return None
    if len(columns) == 0:
        print('Alphabet has zero length. Returning None.')
        return None

    # Parse background
    background_pattern = re.compile(
        "Background letter frequencies.*\n"
        + "(?P<background>(?:\S\s+[\d\.]+\s+)+)",
        flags=re.MULTILINE)
    m = re.search(background_pattern, file_contents)
    if m:
        line = m.group('background')
        atoms = line.strip().split()
        assert len(atoms) % 2 == 0, \
            'Error: background line %s has odd number of entries.' % line
        keys = atoms[::2]
        vals = [float(x) for x in atoms[1::2]]
        background_dict = dict(zip(keys, vals))

        # If keys don't match alphabet, REDEFINE ALPHABET
        if set(keys) != set(columns):
            print('Warning: MEME alphabet does not match background. '
                  + 'Using background specification to set alphabet instead.')
            columns = list(keys)

    else:
        background_dict = None

    # Parse motif names
    name_pattern = re.compile("^MOTIF +(?P<name>.*)$", flags=re.MULTILINE)
    name_matches = re.findall(name_pattern, file_contents)
    if not name_matches:
        print('Could not find any motif names in file. Returning None.')
        return None

    # If user only wants names
    if get_names:
        return name_matches

    # Parse matrices
    matrix_pattern = re.compile(
        "^letter-probability matrix:\s+"
        + "alength=\s*(?P<num_cols>\d+)\s+"
        + "w=\s*(?P<num_rows>\d+).*$\n"
        + "(?P<matrix>[\d\.\s]+)+",
        flags=re.MULTILINE)
    matrix_matches = re.finditer(matrix_pattern, file_contents)
    if not matrix_matches:
        print('Could not find any motifs in file. Returning None.')
        return None

    # Iterate through motifs
    matrix_dict = {}
    matrix_list = []
    for match_num, m in enumerate(matrix_matches):

        # Get name
        if len(name_matches) < match_num:
            print("Error: number of names does not match number of matrices."
                  + "Returning None")
            return None
        name = name_matches[match_num]

        # Extract matrix information
        matrix_str = m.group('matrix')
        num_cols = int(m.group('num_cols'))
        num_rows = int(m.group('num_rows'))

        # Check number of columns
        assert len(columns) == num_cols

        # Set indices
        indices = range(num_rows)

        # Fill in matrix
        matrix = pd.DataFrame(index=indices, columns=columns)
        matrix_lines = [line.strip() for line in matrix_str.split('\n') if
                        len(line.strip()) > 0]
        for i, line in enumerate(matrix_lines):
            entries = [float(x) for x in line.split()]
            assert len(entries) == num_cols, \
                'Error: %d entries does not match %d columns' % \
                (len(entries), num_cols)
            matrix.loc[i, :] = entries
        matrix.loc[:, :] = matrix.values.astype(float)

        # Check number of rows
        assert len(matrix) == num_rows

        # Record name
        matrix.name = name

        # Record matrix
        matrix_dict[name] = matrix
        matrix_list.append(matrix)

    ### Figure out what to return

    # get_alphabet
    if get_alphabet:
        return columns

    # get_background
    if get_background:
        if background_dict is not None:
            return background_dict
        else:
            print('Could not find background line in file. Returning None.')
            return None

    # get_motif_dict
    elif get_dict:
        if len(matrix_dict) > 0:
            return matrix_dict
        else:
            print('Could not parse any motifs. Returning None.')

    # get_motif_list
    elif get_list:
        if len(matrix_list) > 0:
            return matrix_list
        else:
            print('Could not parse any motifs. Returning None.')

    # only return the requested motif by name
    elif motif_name is not None:
        if not motif_name in matrix_dict:
            print('Could not find motif of name %s. Returning None.' % motif_name)
            return None
        else:
            return matrix_dict[motif_name]

    # only return the requested motif by number
    elif motif_num is not None:
        if len(matrix_list) < motif_num:
            print('Could not find motif number %d. Returning None.' % motif_num)
            return None
        else:
            return matrix_list[motif_num - 1]

    # Otherwise, return the first motif and give a warning
    else:
        if len(matrix_list) > 1:
            print('Warning: Returning only the first of %d motifs.' %
                  len(matrix_list))
        return matrix_list[0]
        
# do imports
import matplotlib.pyplot as plt
import logomaker as logomaker

# get data frame of all color schemes
all_df = logomaker.list_color_schemes()

# set the two types of character sets
char_sets = ['ACGTU', 'ACDEFGHIKLMNPQRSTVWY']
colspans = [1, 3]
num_cols = sum(colspans)

# compute the number of rows
num_rows_per_set = []
for char_set in char_sets:
    num_rows_per_set.append((all_df['characters'] == char_set).sum())
num_rows = max(num_rows_per_set)

# create figure
height_per_row = .8
width_per_col = 1.5
fig = plt.figure(figsize=[width_per_col * num_cols, height_per_row * num_rows])

# for each character set
for j, char_set in enumerate(char_sets):

    # get color schemes for that character set only
    df = all_df[all_df['characters'] == char_set].copy()
    df.sort_values(by='color_scheme', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # for each color scheme
    for row_num, row in df.iterrows():
        # set axes
        col_num = sum(colspans[:j])
        col_span = colspans[j]
        ax = plt.subplot2grid((num_rows, num_cols), (row_num, col_num),
                              colspan=col_span)

        # get color scheme
        color_scheme = row['color_scheme']

        # make matrix for character set
        mat_df = logomaker.sequence_to_matrix(char_set)

        # make and style logo
        logomaker.Logo(mat_df,
                       ax=ax,
                       color_scheme=color_scheme,
                       show_spines=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(repr(color_scheme))

# style and show figure
fig.tight_layout()
fig.show()

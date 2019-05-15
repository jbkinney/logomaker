# do imports
import matplotlib.pyplot as plt
import logomaker as logomaker

# load ARS enrichment matrix
ars_df = logomaker.get_example_matrix('ars_enrichment_matrix',
                                      print_description=False)

# load wild-type ARS1 sequence
with logomaker.open_example_datafile('ars_wt_sequence.txt',
                                     print_description=False) as f:
    lines = f.readlines()
    lines = [l.strip() for l in lines if '#' not in l]
    ars_seq = ''.join(lines)

# trim matrix and sequence
start = 10
stop = 100
ars_df = ars_df.iloc[start:stop, :]
ars_df.reset_index(inplace=True, drop=True)
ars_seq = ars_seq[start:stop]

# create Logo object
ars_logo = logomaker.Logo(ars_df,
                          color_scheme='dimgray',
                          font_name='Luxi Mono')

# color wild-type ARS1 sequence within logo
ars_logo.style_glyphs_in_sequence(sequence=ars_seq, color='darkorange')

# highlight functional regions of ARS1
ars_logo.highlight_position_range(pmin=7, pmax=22, color='lightcyan')
ars_logo.highlight_position_range(pmin=33, pmax=40, color='honeydew')
ars_logo.highlight_position_range(pmin=64, pmax=81, color='lavenderblush')

# additional styling using Logo methods
ars_logo.style_spines(visible=False)

# style using Axes methods
ars_logo.ax.set_ylim([-4, 4])
ars_logo.ax.set_ylabel('$\log_2$ enrichment', labelpad=0)
ars_logo.ax.set_yticks([-4, -2, 0, 2, 4])
ars_logo.ax.set_xticks([])

# show plot
ars_logo.fig.show()

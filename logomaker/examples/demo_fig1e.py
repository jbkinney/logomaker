# do imports
import matplotlib.pyplot as plt
import logomaker as lm

# load ars matrix
ars_df = lm.get_example_matrix('ars_enrichment_matrix',
                              print_description=False)

# load ars wt sequence
with lm.open_example_datafile('ars_wt_sequence.txt',
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
logo = lm.Logo(ars_df,
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

# show plot
plt.show()

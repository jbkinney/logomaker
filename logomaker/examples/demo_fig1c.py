# do imports
import matplotlib.pyplot as plt
import logomaker as lm

# load ss probability matrix
ss_df = lm.get_example_matrix('ss_probability_matrix',
                              print_description=False)

# create and style logo
logo = lm.Logo(ss_df,
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

# show plot
plt.show()

# do imports
import matplotlib.pyplot as plt
import logomaker as lm

# load crp energy matrix
crp_df = -lm.get_example_matrix('crp_energy_matrix',
                                print_description=False)

# create and style logo
logo = lm.Logo(crp_df,
               shade_below=.5,
               fade_below=.5,
               font_name='Arial Rounded MT Bold')
logo.style_spines(visible=False)
logo.style_spines(spines=['left', 'bottom'], visible=True)
logo.ax.set_ylabel("$-\Delta \Delta G$ (kcal/mol)", labelpad=-1)
logo.style_xticks(rotation=90, fmt='%d', anchor=0)
logo.ax.xaxis.set_ticks_position('none')
logo.ax.xaxis.set_tick_params(pad=-1)

# show plot
plt.show()

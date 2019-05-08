# do imports
import matplotlib.pyplot as plt
import logomaker as lm

# load ww information matrix
ww_df = lm.get_example_matrix('ww_information_matrix',
                              print_description=False)

# create and style logo
logo = lm.Logo(ww_df,
               font_name='Stencil Std',
               color_scheme='NajafabadiEtAl2017',
               vpad=.1,
               width=.8)
logo.ax.set_ylabel('information (bits)')
logo.style_xticks(anchor=0, spacing=5, rotation=45)
logo.highlight_position(p=4, color='gold', alpha=.5)
logo.highlight_position(p=26, color='gold', alpha=.5)
logo.ax.set_xlim([-1, len(ww_df)])

# show plot
plt.show()

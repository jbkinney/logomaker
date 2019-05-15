# do imports
import matplotlib.pyplot as plt
import logomaker as logomaker

# load ww information matrix
ww_df = logomaker.get_example_matrix('ww_information_matrix',
                                     print_description=False)

# create Logo object
ww_logo = logomaker.Logo(ww_df,
                         font_name='Stencil Std',
                         color_scheme='NajafabadiEtAl2017',
                         vpad=.1,
                         width=.8)

# style using Logo methods
ww_logo.style_xticks(anchor=0, spacing=5, rotation=45)
ww_logo.highlight_position(p=4, color='gold', alpha=.5)
ww_logo.highlight_position(p=26, color='gold', alpha=.5)

# style using Axes methods
ww_logo.ax.set_ylabel('information (bits)')
ww_logo.ax.set_xlim([-1, len(ww_df)])

# show plot
ww_logo.fig.show()

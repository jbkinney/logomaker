# do imports
import matplotlib.pyplot as plt
import logomaker as logomaker

# make Figure and Axes objects
fig, ax = plt.subplots(1,1,figsize=[4,2])

# load logo matrix
logo_df = logomaker.get_example_matrix('logomaker_logo_matrix',
                                       print_description=False)

# create color scheme
color_scheme = {
    'L' : [0, .5, 0],
    'O' : [1, 0, 0],
    'G' : [1, .65, 0],
    'maker': 'gray'
}

# create Logo object
logo_logo = logomaker.Logo(logo_df, 
                           ax=ax,
                           color_scheme=color_scheme, 
                           baseline_width=0,
                           font_name='Arial', 
                           show_spines=False,
                           vsep=.005,
                           width=.95)

# color the 'O' at the end of the logo a different color
logo_logo.style_single_glyph(c='O', p=3, color=[0, 0, 1])

# change the font of 'maker' and flip characters upright.
logo_logo.style_glyphs_below(font_name='OCR A Std', flip=False, width=1.0)

# remove tick marks
ax.set_xticks([])
ax.set_yticks([])

# tighten layout
logo_logo.fig.tight_layout()

# show plot
logo_logo.fig.show()
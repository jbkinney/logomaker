# do imports
import matplotlib.pyplot as plt
import numpy as np
import logomaker as lm

# load saliency matrix
saliency_df = lm.get_example_matrix('nn_saliency_matrix',
                                    print_description=False)

# create and style saliency logo
logo = lm.Logo(saliency_df)
ax = logo.ax
logo.style_spines(visible=False)
logo.style_spines(spines=['left'], visible=True, bounds=[0, .75])
ax.set_xlim([20, 115])
ax.set_yticks([0, .75])
ax.set_yticklabels(['0', '0.75'])
ax.set_xticks([])
ax.set_ylabel('        saliency', labelpad=-1)

# draw gene
exon_start = 55-.5
exon_stop = 90+.5
y = -.2
ax.set_ylim([-.3, .75])
ax.axhline(y, color='k', linewidth=1)
xs = np.arange(-3, len(saliency_df),10)
ys = y*np.ones(len(xs))
ax.plot(xs, ys, marker='4', linewidth=0, markersize=5, color='k')
ax.plot([exon_start, exon_stop],
        [y, y], color='k', linewidth=10, solid_capstyle='butt')

# show plot
plt.show()

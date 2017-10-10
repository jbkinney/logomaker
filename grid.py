import matplotlib as mpl
import matplotlib.cm as cm


def compute_chargrid_characters(df, cmap, font_name, wtseq=None, width=.8, height=.8):
    poss = df.index.copy()
    rownums = range(len(poss))

    chars = df.columns.copy()
    colnums = range(len(chars))

    # Generate colormap
    vals = df.values.ravel()
    vmin = min(vals)
    vmax = max(vals)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scalar_to_color_func = cm.ScalarMappable(norm=norm, cmap=cmap)

    # List of fill with characters
    char_list = []

    # Create character for each position
    for colnum, char in enumerate(chars):
        for rownum, pos in enumerate(poss):
            x = rownum - .5
            y = colnum - .5
            w = width
            h = height
            alpha = 1

            edgecolor = 'none'
            if wtseq and wtseq[rownum] == char:
                edgecolor = 'k'

            # Get color
            val = df.loc[pos, char]
            color = scalar_to_color_func.to_rgba(val)

            # Create and store character
            char_obj = anylogo.Character(c=char, x=x, y=y, w=w, h=h, color=color,
                                         font_name=font_name, edgecolor=edgecolor)
            char_list.append(char_obj)

    # Get box
    xlb = min([c.box.xlb for c in char_list])
    xub = max([c.box.xub for c in char_list])
    ylb = min([c.box.ylb for c in char_list])
    yub = max([c.box.yub for c in char_list])
    box = anylogo.Box(xlb, xub, ylb, yub)

    return char_list, box


poss = prob_df.index.copy()

# Draw chargrid
fig = plt.figure(figsize=[10, 2])
ax = fig.add_subplot(1, 1, 1)

char_list, box = compute_chargrid_characters(prob_df, cmap='Purples', wtseq=wtseq, font_name=anylogo.DEFAULT_FONT)
for c in char_list:
    c.draw(ax)
ax.set_xlim([box.xlb, box.xub])
ax.set_ylim([box.ylb, box.yub])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(length=0)
ax.set_xticks(poss)
ax.set_xticklabels(poss, rotation=90)
print ''
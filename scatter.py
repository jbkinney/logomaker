color_scheme = 'classic'
color_dict = anylogo.color.get_color_dict(color_scheme, 'ACGT')


def compute_charscatter_characters(df, wtseq, color_dict, font_name, width=.8, aspect=1):
    poss = df.index.copy()
    rownums = range(len(poss))
    xmin = min(poss) - .5
    xmax = max(poss) + .5

    chars = df.columns.copy()
    colnums = range(len(chars))

    # Generate colormap
    vals = df.values.ravel()
    vmin = min(vals)
    vmax = max(vals)

    # List of fill with characters
    char_list = []

    # Create character for each position
    for colnum, char in enumerate(chars):
        for rownum, pos in enumerate(poss):
            x = rownum - .5
            w = width
            h = aspect * (vmax - vmin) * width / (xmax - xmin)
            alpha = 1

            # Get color
            y = df.loc[pos, char] - .5 * h
            color = color_dict[char]

            edgecolor = None
            if wtseq:
                if wtseq[rownum] == char:
                    edgecolor = 'k'
                else:
                    edgecolor = 'none'

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

char_list, box = compute_charscatter_characters(prob_df, wtseq=wtseq, color_dict=color_dict,
                                                font_name=anylogo.DEFAULT_FONT, aspect=6)
for c in char_list:
    c.draw(ax)
ax.set_xlim([box.xlb, box.xub])
ax.set_ylim([box.ylb, box.yub])
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(length=0)
ax.set_xticks(poss)
ax.set_xticklabels(poss, rotation=90)
print ''

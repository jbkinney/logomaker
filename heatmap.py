# Heat map visualization
fig = plt.figure(figsize=[10, 2])
ax = fig.add_subplot(1, 1, 1)
cax = ax.imshow(prob_df.T, cmap='Purples')
chars = prob_df.columns
colnums = range(len(chars))
poss = prob_df.index
rownums = range(len(poss))
ax.set_xticks(rownums)
ax.set_xticklabels(poss)
ax.set_yticks(colnums)
ax.set_yticklabels(chars)

# Plot circles at wt sequence
wtseq = 'ATTAATGTGAGTTAGCTCACTCATTA'
char_to_colnum_dict = dict(zip(chars, colnums))
for rownum in rownums:
    colnum = char_to_colnum_dict[wtseq[rownum]]
    dot = plt.Circle((rownum, colnum), 0.2, facecolor='w', edgecolor='k')
    ax.add_artist(dot)

cbar = fig.colorbar(cax)
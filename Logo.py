from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import character

SMALL = 1E-6

# Logo base class
class Logo:
    def __init__(self,
                 matrix,
                 font_properties,
                 character_style,
                 highlight_sequence,
                 highlight_style,
                 placement_style,
                 axes_style):

        # Set font properties
        self.font_properties = font_properties

        # Set matrix
        self.df = matrix.copy()

        # Get list of characters
        self.chars = np.array([str(c) for c in self.df.columns])

        # Get list of positions
        self.poss = self.df.index.copy()

        # Set positions
        self.L = len(self.df)

        # Set character styles
        self.character_style = character_style.copy()
        self.highlight_style = highlight_style.copy()
        self.placement_style = placement_style.copy()

        # Set axes style
        self.axes_style = axes_style.copy()

        # Set wild type sequence
        self.highlight_sequence = highlight_sequence
        if self.highlight_sequence is not None:
            assert isinstance(self.highlight_sequence, basestring), \
                'Error: highlight_sequence is not a string.'
            assert len(self.highlight_sequence ) == len(self.poss), \
                'Error: highlight_sequence has a different length than matrix.'
            assert set(list(str(self.highlight_sequence))) == set(self.chars),\
                'Error: highlight_sequence %s contains invalid characters'\
                %self.highlight_sequence
            self.use_highlight = True
        else:
            self.use_highlight = False

        # Compute characters and box
        self.compute_characters()


    def compute_characters(self):

        # Get largest value for computing transparency
        max_alpha_val = self.placement_style['max_alpha_val']
        if max_alpha_val is None:
            max_alpha_val = abs(self.df.values).max()

        # Compute hstretch values for all characters
        width = self.placement_style['width']
        hpad = self.placement_style['hpad']
        vpad = self.placement_style['vpad']
        hstretch_dict, vstretch_dict = \
            character.get_stretch_vals(self.chars,
                                    width=width,
                                    height=1,
                                    font_properties=self.font_properties,
                                    hpad=hpad,
                                    vpad=vpad)
        # Set max_hstretch
        uniform_stretch = self.placement_style['uniform_stretch']
        max_stretched_character = \
            self.placement_style['max_stretched_character']
        if uniform_stretch:
            max_hstretch = min(hstretch_dict.values())
        elif max_stretched_character:
            max_hstretch = hstretch_dict[max_stretched_character]
        else:
            max_hstretch = np.Inf

        char_list = []
        for i, pos in enumerate(self.poss):

            vals = self.df.iloc[i, :].values
            ymin = vals[vals < 0].sum()

            # Reorder columns
            stack_order = self.placement_style['stack_order']
            if stack_order == 'big_on_top':
                indices = np.argsort(vals)
            elif stack_order == 'small_on_top':
                tmp_indices = np.argsort(vals)
                pos_tmp_indices = tmp_indices[vals[tmp_indices] >= 0]
                neg_tmp_indices = tmp_indices[vals[tmp_indices] < 0]
                indices = np.array(list(neg_tmp_indices[::-1]) +
                                   list(pos_tmp_indices[::-1]))
            elif stack_order == 'fixed':
                indices = range(len(vals))
            else:
                assert False, 'Error: unrecognized stack_order value %s.'%\
                              stack_order
            ordered_chars = self.chars[indices]

            # This is the same for every character
            x = pos - width/2.0
            w = width

            # Initialize y
            y = ymin

            for n, char in enumerate(ordered_chars):

                # Get value
                val = self.df.loc[pos, char]

                # Get height
                h = abs(val)
                if h < SMALL:
                    continue

                # Get facecolor, edgecolor, and edgewidth
                if self.use_highlight and (char == self.highlight_sequence[i]):
                    facecolor = self.highlight_style['facecolors'][char].copy()
                    edgecolor = self.highlight_style['edgecolors'][char].copy()
                    boxcolor = self.highlight_style['boxcolors'][char].copy()
                    boxedgecolor = self.highlight_style['boxedgecolors'][char]\
                        .copy()
                    edgewidth = self.highlight_style['edgewidth']
                    boxedgewidth = self.highlight_style['boxedgewidth']
                else:
                    facecolor = self.character_style['facecolors'][char].copy()
                    edgecolor = self.character_style['edgecolors'][char].copy()
                    boxcolor = self.character_style['boxcolors'][char].copy()
                    boxedgecolor = self.character_style['boxedgecolors'][char]\
                        .copy()
                    edgewidth = self.character_style['edgewidth']
                    boxedgewidth = self.character_style['boxedgewidth']

                # Get flip and shade character accordingly
                if val <= 0.0:
                    flip = self.placement_style['below_flip']
                    shade = self.placement_style['below_shade']
                    alpha = self.placement_style['below_alpha']
                    facecolor *= np.array([shade, shade, shade, alpha])
                    edgecolor *= np.array([shade, shade, shade, alpha])
                else:
                    flip = False

                # Set alpha
                if self.placement_style['use_transparency']:
                    alpha = h / max_alpha_val
                    if alpha > 1:
                        alpha = 1.0
                    elif alpha <= 0:
                        alpha = 0.0
                    facecolor[3] *= alpha
                    edgecolor[3] *= alpha

                # Create Character object
                char = character.Character(
                    c=char, xmin=x, ymin=y, width=w, height=h, flip=flip,
                    font_properties=self.font_properties,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    boxcolor=boxcolor,
                    boxedgecolor=boxedgecolor,
                    boxedgewidth=boxedgewidth,
                    hpad=hpad,
                    vpad=vpad,
                    max_hstretch=max_hstretch)

                # Store Character object
                char_list.append(char)

                # Increment y
                y += h

        # Get box
        xmin = min([c.bbox.xmin for c in char_list])
        xmax = max([c.bbox.xmax for c in char_list])
        ymin = min([c.bbox.ymin for c in char_list])
        ymax = max([c.bbox.ymax for c in char_list])
        bbox = Bbox([[xmin, ymin], [xmax, ymax]])

        # Set char_list and box
        self.char_list = char_list
        self.bbox = bbox


    def draw(self, ax=None):

        # If no axes is provided, default to current axes
        if ax is None:
            ax = plt.gca()

        # Draw characters
        for char in self.char_list:
            char.draw(ax)

        # Draw floor line
        ax.axhline(0, color='k', linewidth=self.axes_style['baseline_width'])

        # Draw gridlines
        if self.axes_style['show_gridlines']:
            ax.grid(**self.axes_style['gridline_dict'])

        # Set limits
        ax.set_xlim(self.axes_style['xlim'])
        ax.set_ylim(self.axes_style['ylim'])

        # Draw x-axis annotation
        if self.axes_style['xticks'] is not None:
            ax.set_xticks(self.axes_style['xticks'])

        if self.axes_style['yticks'] is not None:
            ax.set_yticks(self.axes_style['yticks'])

        if self.axes_style['xlabel'] is not None:
            ax.set_xlabel(self.axes_style['xlabel'],
                    font_properties=self.axes_style['label_fontproperties'])

        if self.axes_style['ylabel'] is not None:
            ax.set_ylabel(self.axes_style['ylabel'],
                    font_properties=self.axes_style['label_fontproperties'])

        if self.axes_style['title'] is not None:
            ax.set_title(self.axes_style['title'],
                    font_properties=self.axes_style['title_fontproperties'])

        if self.axes_style['xtick_length'] is not None:
            ax.xaxis.set_tick_params(length=self.axes_style['xtick_length'])

        if self.axes_style['ytick_length'] is not None:
            ax.yaxis.set_tick_params(length=self.axes_style['ytick_length'])

        if self.axes_style['left_spine'] is not None:
            ax.spines['left'].set_visible(self.axes_style['left_spine'])

        if self.axes_style['right_spine'] is not None:
            ax.spines['right'].set_visible(self.axes_style['right_spine'])

        if self.axes_style['top_spine'] is not None:
            ax.spines['top'].set_visible(self.axes_style['top_spine'])

        if self.axes_style['bottom_spine'] is not None:
            ax.spines['bottom'].set_visible(self.axes_style['bottom_spine'])

        # Initial rendering needed to get axes tick labels
        if self.axes_style['use_tightlayout']:
            plt.tight_layout()
        plt.draw()

        if self.axes_style['xticklabels'] is not None:
            xticklabels = self.axes_style['xticklabels']
        elif self.axes_style['xtick_format'] is not None:
            xticklabels = [self.axes_style['xtick_format'] % x
                           for x in ax.get_xticks()]
        else:
            xticklabels = ax.get_xticklabels()

        ax.set_xticklabels(xticklabels,
                        rotation=self.axes_style['xtick_rotation'],
                        font_properties=self.axes_style['tick_fontproperties'])

        if self.axes_style['yticklabels'] is not None:
            yticklabels = self.axes_style['yticklabels']
        elif self.axes_style['ytick_format'] is not None:
            yticklabels = [self.axes_style['ytick_format'] % y
                           for y in ax.get_yticks()]
        else:
            yticklabels = ax.get_yticklabels()

        ax.set_yticklabels(yticklabels,
                        rotation=self.axes_style['ytick_rotation'],
                        font_properties=self.axes_style['tick_fontproperties'])

        # Do final drawing
        if self.axes_style['use_tightlayout']:
            plt.tight_layout()
        plt.draw()
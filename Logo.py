from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import character
from matplotlib.text import Text
from matplotlib.lines import Line2D
import pdb

SMALL = 1E-6

# Logo base class
class Logo:
    def __init__(self,
                 matrix,
                 font_properties,
                 character_style,
                 highlight_sequence,
                 highlight_style,
                 fullheight,
                 fullheight_style,
                 placement_style,
                 axes_style,
                 scalebar_style):

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
        self.fullheight_style = fullheight_style.copy()
        self.scalebar_style = scalebar_style.copy()

        # Set axes style
        self.axes_style = axes_style.copy()

        # Set wild type sequence
        self.highlight_sequence = highlight_sequence
        if self.highlight_sequence is not None:
            assert isinstance(self.highlight_sequence, basestring), \
                'Error: highlight_sequence is not a string.'
            assert len(self.highlight_sequence ) == len(self.poss), \
                'Error: highlight_sequence has a different length than matrix.'
            self.use_highlight = True
        else:
            self.use_highlight = False

        # Set fullheight
        self.fullheight = fullheight
        if self.fullheight is not None:
            self.use_fullheight = True
        else:
            self.use_fullheight = False

        # Compute characters and box
        self.compute_characters()

    def compute_characters(self):

        # Get largest value for computing transparency
        max_alpha_val = self.placement_style['max_alpha_val']
        if max_alpha_val is None:
            try:
                max_alpha_val = abs(self.df.values).max()
            except:
                pdb.set_trace()

        # Compute ymin and ymax of logo.
        # Set NaN values to zero when doing so
        values = self.df.fillna(0).values.astype(float)
        pos_mask = (values > 0).astype(float)
        neg_mask = (values < 0).astype(float)
        ymax = (values * pos_mask).sum(axis=1).max()
        ymin = (values * neg_mask).sum(axis=1).min()
        yspan = ymax-ymin

        # Set ysep
        ysep = yspan * self.placement_style['vsep']

        # Recalibrate if not showing empty boxes
        if self.placement_style['remove_flattened_characters']:
            pos_mask = (values > ysep).astype(float)
            neg_mask = (values < -ysep).astype(float)
            ymax = (values * pos_mask).sum(axis=1).max()
            ymin = (values * neg_mask).sum(axis=1).min()
            yspan = ymax-ymin

        # If ylim was manually specified, override all that
        if self.axes_style['ylim'] is not None:
            ylim = self.axes_style['ylim']
            ymin = ylim[0]
            ymax = ylim[1]
            yspan = ymax-ymin
            ysep = yspan * self.placement_style['vsep']

        # Compute hstretch values for all characters
        width = self.placement_style['width']
        hpad = self.placement_style['hpad']
        vpad = self.placement_style['vpad']
        hstretch_dict, vstretch_dict = \
            character.get_stretch_vals(self.chars,
                                    width=width,
                                    height=1,
                                    font_properties=self.font_properties,
                                    hpad=0,
                                    vpad=0)

        xmin = min(self.poss) - width / 2.0
        xmax = max(self.poss) + width / 2.0

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

            # Remove non-finite values and characters
            finite_indices = np.isfinite(vals)
            chars = self.chars[finite_indices]
            vals = vals[finite_indices]

            if not self.placement_style['remove_flattened_characters']:
                col_ymin = vals[vals < 0].sum()
            else:
                col_ymin = vals[vals < -ysep].sum()

            # Reorder columns
            stack_order = self.placement_style['stack_order']
            if stack_order == 'big_on_top':
                ordered_indices = np.argsort(vals)
            elif stack_order == 'small_on_top':
                tmp_indices = np.argsort(vals)
                pos_tmp_indices = tmp_indices[vals[tmp_indices] >= 0]
                neg_tmp_indices = tmp_indices[vals[tmp_indices] < 0]
                ordered_indices = np.array(list(neg_tmp_indices[::-1]) +
                                   list(pos_tmp_indices[::-1]))
            elif stack_order == 'fixed_going_up':
                ordered_indices = range(len(vals))
            elif stack_order == 'fixed_going_down':
                ordered_indices = range(len(vals))[::-1]
            else:
                assert False, 'Error: unrecognized stack_order value %s.'%\
                              stack_order

            # Reorder characters and values as appropriate
            ordered_chars = chars[ordered_indices]
            vals = vals[ordered_indices]

            # This is the same for every character
            x = pos - width/2.0
            w = width

            # Initialize y
            y = col_ymin

            for n, char in enumerate(ordered_chars):

                # Get value
                val = vals[n]

                # Get height
                h = abs(val)

                # Make sure character has finite height
                if h - ysep < SMALL:
                    if not self.placement_style['remove_flattened_characters']:
                        y += h
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
                    zorder = self.highlight_style['zorder']
                else:
                    facecolor = self.character_style['facecolors'][char].copy()
                    edgecolor = self.character_style['edgecolors'][char].copy()
                    boxcolor = self.character_style['boxcolors'][char].copy()
                    boxedgecolor = self.character_style['boxedgecolors'][char]\
                        .copy()
                    edgewidth = self.character_style['edgewidth']
                    boxedgewidth = self.character_style['boxedgewidth']
                    zorder = self.character_style['zorder']

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
                char_obj = character.Character(
                    c=char,
                    xmin=x,
                    ymin=y+ysep/2.0,
                    width=w,
                    height=h-ysep,
                    flip=flip,
                    font_properties=self.font_properties,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    boxcolor=boxcolor,
                    boxedgecolor=boxedgecolor,
                    boxedgewidth=boxedgewidth,
                    hpad=hpad,
                    vpad=vpad,
                    max_hstretch=max_hstretch,
                    zorder=zorder)

                # Store Character object
                char_list.append(char_obj)

                # Increment y
                y += h

        # Set ylims for full-height characters
        if self.axes_style['ylim'] is not None:
            ylim = self.axes_style['ylim']
            ymin = ylim[0]
            ymax = ylim[1]
            yspan = ymax-ymin

        # Process fixed characters
        if self.use_fullheight:
            for pos, char in self.fullheight.items():

                # Set patch style
                facecolor = self.fullheight_style['facecolors'][char].copy()
                edgecolor = self.fullheight_style['edgecolors'][char].copy()
                boxcolor = self.fullheight_style['boxcolors'][char].copy()
                boxedgecolor = self.fullheight_style['boxedgecolors'][char] \
                    .copy()
                edgewidth = self.fullheight_style['edgewidth']
                boxedgewidth = self.fullheight_style['boxedgewidth']
                zorder = self.fullheight_style['zorder']

                # Set ysep
                ysep = yspan * self.fullheight_style['vsep']
                width = self.fullheight_style['width']

                # This is the same for every character
                y = ymin + ysep/2.0
                h = yspan - ysep
                if h < SMALL:
                    continue

                x = pos - width/2.0
                w = width

                # Create Character object

                char_obj = character.Character(
                    c=char,
                    xmin=x,
                    ymin=y,
                    width=w,
                    height=h,
                    flip=False,
                    font_properties=self.font_properties,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=edgewidth,
                    boxcolor=boxcolor,
                    boxedgecolor=boxedgecolor,
                    boxedgewidth=boxedgewidth,
                    hpad=hpad,
                    vpad=vpad,
                    max_hstretch=max_hstretch,
                    zorder=zorder)

                # Store Character object
                char_list.append(char_obj)

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

        # Set limits
        ax.set_xlim(self.axes_style['xlim'])
        ax.set_ylim(self.axes_style['ylim'])

        # Draw baseline
        if self.axes_style['show_baseline']:
            ax.axhline(0, **self.axes_style['baseline_dict'])

        # Style gridlines
        ax.grid(**self.axes_style['gridline_dict'])

        # Draw vlines
        for x in self.axes_style['vline_positions']:
            ax.axvline(x, **self.axes_style['vline_dict'])

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

        # Set xticklabels
        xticks = np.array(ax.get_xticks()).astype(int)
        if not self.axes_style['show_position_zero']:
            indices = xticks >= 0
            xticks[indices] = xticks[indices] + 1

        if self.axes_style['xticklabels'] is not None:
            xticklabels = self.axes_style['xticklabels']
        elif self.axes_style['xtick_format'] is not None:
            xticklabels = [self.axes_style['xtick_format'] % x for x in xticks]
        else:
            xticklabels = xticks
        ax.set_xticklabels(xticklabels,
                           rotation=self.axes_style['xtick_rotation'],
                           font_properties=
                           self.axes_style['tick_fontproperties'])

        # Set yticklabels
        if self.axes_style['yticklabels'] is not None:
            yticklabels = self.axes_style['yticklabels']
        elif self.axes_style['ytick_format'] is not None:
            yticklabels = [self.axes_style['ytick_format'] % y
                           for y in ax.get_yticks()]
        else:
            yticklabels = ax.get_yticks()
        ax.set_yticklabels(yticklabels,
                           rotation=self.axes_style['ytick_rotation'],
                           font_properties=
                                self.axes_style['tick_fontproperties'])

        # Draw scalebar and remove left and right spines, yticks
        if self.scalebar_style['visible']:
            linestyle = self.scalebar_style['line_kwargs']
            textstyle = self.scalebar_style['text_kwargs']

            # Shrink spine to the desired length
            spine = ax.spines['left']
            spine.set_bounds(linestyle['ymin'],
                             linestyle['ymax'])

            # Shift spine out
            spine.set_position(('data', linestyle['xloc']))

            # Make spine think and set styling
            spine.set_linewidth(2)
            spine.set_color(linestyle['color'])

            # Label spine
            ax.set_yticks([textstyle['y']])
            fontdict = {
                'verticalalignment': textstyle['verticalalignment'],
                'horizontalalignment': textstyle['horizontalalignment']
            }
            ax.set_yticklabels([textstyle['text']],
                               rotation=textstyle['rotation'],
                               fontdict=fontdict)
            ax.yaxis.set_tick_params(length=0)
            ax.grid('off')
            ax.spines['right'].set_visible(False)

        # Do final drawing
        if self.axes_style['use_tightlayout']:
            plt.tight_layout()
        plt.draw()

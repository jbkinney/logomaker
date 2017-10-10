# Information logo clas
class InformationLogo(Logo):
    def __init__(self, ax, prob_df, bg_df, color_dict, ylabel=None,
                 use_transparency=False,
                 font_name=DEFAULT_FONT):

        # Set axis
        self.ax = ax

        # Set axis parameters
        self.logo_style = logo_style
        self.floor_line_width = .5

        # Set data
        df = prob_df.copy()
        info_vec = prob_df.values * np.log2(prob_df.values / bg_df.values)
        df.loc[:, :] = prob_df.values * info_vec.sum(axis=1)[:, np.newaxis]
        assert all(np.ravel(df.values) >= 0)
        self.df = df.copy()
        self.poss = df.index.copy()
        self.chars = df.columns.copy()

        # Get colors
        color_dict = color.get_color_dict(color_scheme=color_scheme, chars=self.chars)

        # Compute characters
        self.stack_order='big_on_top'
        self.color_dict = color_dict
        self.font_name = font_name
        self.use_transparency = use_transparency
        self.neg_shade = 1
        self.neg_flip = False
        self.max_alpha_val = None
        self.use_transparency = False

        self.compute_characters()

        assert np.isclose(self.box.ylb, 0), \
            'Error: box.ylb=%f is not zero.' % self.box.ylb

        self.signed_heights_df = df
        self.floor_line_width = 0
        self.font_name = font_name
        self.prob_df = prob_df.copy()

        self.bg_df = bg_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1])) + 1
        )
        self.xticklabels = ['%d' % x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [0, max(2, box.yub)]
        self.yticks = range(
            int(np.ceil(self.ylim[0])),
            int(np.floor(self.ylim[1])) + 1
        )
        self.yticklabels = ['%d' % y for y in self.yticks]
        self.ylabel = 'information\n(bits)' if (ylabel is None) else ylabel

        # Register that logo has been set
        Logo.__init__(self, logo_set=True)

# Probability logo clas
class ProbabilityLogo(Logo):
    def __init__(self, prob_df, color_dict, ylabel=None,
                 use_transparency=True,
                 font_name=DEFAULT_FONT, floor_line_width=.5):
        df = prob_df.copy()
        assert all(np.ravel(df.values) >= 0)

        char_list, box = compute_logo_characters(
            df=df,
            stack_order='small_on_top',
            color_dict=color_dict,
            font_name=font_name,
            max_alpha_val=1.0,
            use_transparency=use_transparency)

        assert np.isclose(box.ylb, 0), \
            'Error: box.ylb=%f is not zero.' % box.ylb
        assert np.isclose(box.yub, 1), \
            'Error: box.yub=%f is not one.' % box.yub

        self.signed_heights_df = df
        self.floor_line_width = 0  # floor_line_width
        self.font_name = font_name
        self.prob_df = prob_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1])) + 1
        )
        self.xticklabels = ['%d' % x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [0, 1]
        self.yticks = [0, .5, 1]
        self.yticklabels = ['%.1f' % y for y in self.yticks]
        self.ylabel = 'probability' if (ylabel is None) else ylabel

        # Register that logo has been set
        Logo.__init__(self, logo_set=True)

# Effect logo clas
class EffectLogo(Logo):
    def __init__(self, effect_df, color_dict, ylabel=None,
                 use_transparency=True,
                 font_name=DEFAULT_FONT, floor_line_width=.5):
        df = effect_df.copy()
        char_list, box = compute_logo_characters(
            df=df,
            stack_order='big_on_top',
            color_dict=color_dict,
            font_name=font_name,
            use_transparency=use_transparency,
            neg_shade=.5,
            neg_flip=True
        )

        self.signed_heights_df = df
        self.floor_line_width = floor_line_width
        self.font_name = font_name
        self.effect_df = effect_df.copy()
        self.box = box
        self.char_list = char_list

        self.xlim = [box.xlb, box.xub]
        self.xticks = range(
            int(np.ceil(self.xlim[0])),
            int(np.floor(self.xlim[1])) + 1
        )
        self.xticklabels = ['%d' % x for x in self.xticks]
        self.xlabel = 'position'

        self.ylim = [box.ylb, box.yub]
        self.yticks = range(
            int(np.ceil(self.ylim[0])),
            int(np.floor(self.ylim[1])) + 1
        )
        self.yticklabels = ['%d' % y for y in self.yticks]
        self.ylabel = 'effect' if (ylabel is None) else ylabel

        # Register that logo has been set
        Logo.__init__(self, logo_set=True)


def compute_logo_characters(df, \
                            stack_order, color_dict, \
                            font_name=DEFAULT_FONT, \
                            neg_shade=1, neg_flip=False, \
                            max_alpha_val=None,
                            use_transparency=False):
    poss = df.index.copy()
    chars = df.columns.copy()

    char_list = []

    # Get largest value for computing transparency
    if max_alpha_val is None:
        max_alpha_val = abs(df.values).max()

    for i, pos in enumerate(poss):

        vals = df.loc[pos, :].values
        ymin = (vals * (vals < 0)).sum()
        ymax = (vals * (vals > 0)).sum()

        # Reorder columns
        if stack_order == 'big_on_top':
            indices = np.argsort(vals)
        elif stack_order == 'small_on_top':
            indices = np.argsort(vals)[::-1]
        ordered_chars = chars[indices]

        # This is the same for every character
        x = pos - .5
        w = 1.0

        # Initialize y
        y = ymin

        for n, char in enumerate(ordered_chars):

            # Get value
            val = df.loc[pos, char]

            # Get height
            h = abs(val)
            if h < SMALL:
                continue

            # Get color
            color = color_dict[char]

            # Get flip, alpha, and shade
            if val >= 0.0:
                alpha = 1.0
                flip = False
                shade = 1.0
            else:
                alpha = neg_shade
                flip = neg_flip
                shade = neg_shade

            if use_transparency:
                alpha *= h / max_alpha_val
                if alpha > 1:
                    alpha = 1.0

            assert alpha <= 1.0, \
                'Error: alpha=%f must be in [0,1]' % alpha

            # Create and store character
            char = Character(
                c=char, x=x, y=y, w=w, h=h,
                alpha=alpha, color=color, flip=flip,
                shade=shade, font_name=font_name, edgecolor='none')
            char_list.append(char)

            # Increment y
            y += h

    # Get box
    xlb = min([c.box.xlb for c in char_list])
    xub = max([c.box.xub for c in char_list])
    ylb = min([c.box.ylb for c in char_list])
    yub = max([c.box.yub for c in char_list])
    box = Box(xlb, xub, ylb, yub)

    return char_list, box



def draw(ax,
         prob_df=None,
         effect_df=None,
         find_beta=False,
         info_per_pos=1.0,
         use_transparency=False,
         background=None,
         ylabel=None,
         logo_type='probability',
         logo_style='classic',
         color_scheme='classic',
         shuffle_colors=False,
         font_name=DEFAULT_FONT):
    # Convert effect_df to prob_df
    if (prob_df is None) and not (effect_df is None):
        bg_df = data.set_bg_df(background, effect_df)
        target_info = info_per_pos * len(effect_df)
        effect_df = data.normalize_effect_df(effect_df)
        if find_beta:
            beta = data.get_beta_for_effect_df(effect_df, bg_df, target_info)
        else:
            beta = 1
        prob_df = data.effect_df_to_prob_df(effect_df, bg_df, beta)

    # Convert prob_df to effect_df (does use bg_df)
    elif (effect_df is None) and not (prob_df is None):
        bg_df = data.set_bg_df(background, prob_df)
        prob_df = data.normalize_prob_df(prob_df)
        effect_df = prob_df.copy()
        values = np.log(prob_df.values / bg_df.values)
        effect_df.loc[:, :] = values - values.mean(axis=1)[:, np.newaxis]
    else:
        assert False, \
            'Error: exactly one of effect_df or prob_df must not be None.'

    # Get color dict from specified color scheme
    chars = prob_df.columns
    color_dict = color.get_color_dict(color_scheme=color_scheme, chars=chars, shuffle_colors=shuffle_colors)

    # Create logos based on logo_type
    if logo_type == 'information':
        logo = InformationLogo(ax, prob_df, bg_df, color_dict,
                               logo
                               use_transparency=use_transparency,
                               ylabel=ylabel,
                               font_name=font_name)

    # elif logo_type == 'probability':
    #     logo = ProbabilityLogo(prob_df, ax, color_dict,
    #                            use_transparency=use_transparency,
    #                            ylabel=ylabel,
    #                            font_name=font_name)
    #
    # elif logo_type == 'effect':
    #     logo = EffectLogo(effect_df, ax, color_dict,
    #                       use_transparency=use_transparency,
    #                       ylabel=ylabel,
    #                       font_name=font_name)

    else:
        assert False, 'Error! Unrecognized logo_type %s' % logo_type

    # Draw logo
    logo.draw()
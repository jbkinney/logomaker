# Set constants
SMALL = 1E-6

class Box:
    def __init__(self, xlb, xub, ylb, yub):
        self.xlb = xlb
        self.xub = xub
        self.ylb = ylb
        self.yub = yub
        self.x = xlb
        self.y = ylb
        self.w = xub - xlb
        self.h = yub - ylb
        self.bounds = (xlb, xub, ylb, yub)

def restrict_dict(in_dict, keys_to_keep):
    return dict([(k, v) for k, v in in_dict.iteritems() if k in keys_to_keep])
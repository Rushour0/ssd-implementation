import torch

def decimate(tensor, m):
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

def colorstr(string, color='green'):
    colors = dict(
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38,
        brightgrey=37,
        brightred=91,
        brightgreen=92,
        brightyellow=93,
        brightblue=94,
        black=30,
        
    )
    return '\033[1;%dm%s\033[0m' % (colors[color], string)
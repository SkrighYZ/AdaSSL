
import torch.nn as nn

def initialize_mlp(mlp, act='relu'):
    # Initialize mlp weights for numerical stability
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain(act))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

def _append_layer(layers, in_dim, out_dim, act, norm):
    act_dict = {'relu': nn.ReLU, 'mish': nn.Mish, 'leaky_relu': nn.LeakyReLU} 

    if norm is None:
        layers.append(nn.Linear(in_dim, out_dim, bias=True))
        layers.append(act_dict[act](inplace=True))
    elif norm == 'bn':
        layers.append(nn.Linear(in_dim, out_dim, bias=False))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(act_dict[act](inplace=True))
    elif norm == 'gn':
        layers.append(nn.Linear(in_dim, out_dim, bias=False))
        layers.append(nn.GroupNorm(32, out_dim))
        layers.append(act_dict[act](inplace=True))


def build_mlp(sizes, act='relu', norm='bn', bias=False):
    
    layers = []

    for i in range(len(sizes) - 2):
        _append_layer(layers, sizes[i], sizes[i + 1], act, norm)
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=bias))

    return nn.Sequential(*layers)



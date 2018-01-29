"""
Write different functionalities not directly available in torch here
"""

import torch
import torch.nn as nn

######### Activations ##########

def get_activations(activations=None):
    if activations == "swish":
        return Swish()
    else:
        # default activation
        return nn.Hardtanh(0, 20, inplace=True)

class Swish(torch.nn.Module):
    """ Implemetation of Swish Activation function:
    y = x * signmoid(x)
    """

    def forward(self, input):
        return (input * torch.sigmoid(input))

    def __repr__(self):
        return self.__class__.__name__ + '()'


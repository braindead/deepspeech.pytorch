"""
Write different functionalities not directly available in torch here
"""

import torch

######### Activations ##########

class Swish(torch.nn.Module):
    """ Implemetation of Swish Activation function:
    y = x * signmoid(x)
    """

    def forward(self, input):
        return (input * torch.sigmoid(input))

    def __repr__(self):
        return self.__class__.__name__ + '()'


import math

import torch
import torch.nn as nn
from torch.nn import init
from src.models.modules.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply


try:
    from pytorch_block_sparse import BlockSparseMatrix
    from pytorch_block_sparse.block_sparse_linear import BlockSparseLinearFunction
except ImportError:
    BlockSparseMatrix = None
    BlockSparseLinearFunction = None


class ButterflyBlockSparseLinear(nn.Module):
    """
    Arguments
    ---------
        sparsity_config: optional: this parameter determins sparsity pattern configuration; it is based on SparsityConfig class.
    """
    def __init__(self, in_features, out_features, blocks=8, bias=True):
        """
        Currently it only supports squared matrix and sqrt(in_features) should be divisible by 32 (e.g. 1024, 4096).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        l_blocks = self.in_features // blocks
        r_blocks = self.out_features //blocks

        # initialize butterfly left and right factors
        if in_features < out_features:
            self.factorL = nn.Parameter(torch.empty(blocks, l_blocks, l_blocks))
            self.factorR = nn.Parameter(torch.empty(blocks, r_blocks, l_blocks))
        else:
            self.factorL = nn.Parameter(torch.empty(blocks, r_blocks, l_blocks))
            self.factorR = nn.Parameter(torch.empty(blocks, r_blocks, r_blocks))
        self.saving = (torch.numel(self.factorR)+torch.numel(self.factorL))/(in_features*out_features)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan = self.factorR.shape[-1]
        gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.factorR.uniform_(-bound, bound)
        fan = self.factorL.shape[-1]
        gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.factorL.uniform_(-bound, bound)


        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        b, d1, d2 = input.shape
        input_squeeze = input.reshape(-1, input.shape[-1])
        # print(input.shape, self.factorL.shape, self.factorR.shape)
        input_squeeze = blockdiag_butterfly_multiply(input_squeeze, self.factorL, self.factorR)
        input_squeeze = input_squeeze.reshape(b, d1, -1)
        return (input_squeeze + self.bias) if self.bias is not None else input_squeeze

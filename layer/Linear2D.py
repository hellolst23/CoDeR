# -*- coding: utf-8 -*-
"""
# @File    : Linear2D.py

"""

from layer import *
from util.utils import timefn
class Linear2D(nn.Module):
    """
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *,H_{z}, H_{in})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *,H_{z}, H_{out})` where all but the last dimension
          are the same shape as the input
    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(5,10, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([5,10,30])
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    #@timefn
    def forward(self, input):
        """
        Args:
            input: torch.Tensor, batchSize  *  (*) * N * in_features
        Returns:
            output: torch.Tensor, batchSize *  (*) * N * out_features
        """
        if len(input.shape) == 4:  # batchSize  *  (*) * N * in_features
            output = torch.matmul(input, self.weight.unsqueeze(0).unsqueeze(0))  # batchSize  *  (*) * N * out_features
            if self.bias is not None:
                output = output + self.bias.unsqueeze(0)  # batchSize  *  (*) * N * out_features
        return output


    def extra_repr(self):
        return 'z_features={}, in_features={}, out_features={}, bias={}'.format(
            self.z_features,self.in_features, self.out_features, self.bias is not None
        )

if __name__ == "__main__":
    m = Linear2D(10,20)
    a = torch.ones(6,2,15,10)
    b = m(a)
    print(b.shape)
    for name, para in m.named_parameters():
        print(name, para.shape, para)

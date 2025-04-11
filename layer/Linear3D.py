from layer import *
class Linear3D(nn.Module):
    def __init__(self,z_features, in_features, out_features, bias=True):
        super(Linear3D, self).__init__()
        self.z_features = z_features
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(z_features,in_features,out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(z_features,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = torch.matmul(input, self.weight.unsqueeze(0))
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)
        return output

    def extra_repr(self):
        return 'z_features={}, in_features={}, out_features={}, bias={}'.format(
            self.z_features,self.in_features, self.out_features, self.bias is not None
        )

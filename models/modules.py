from torch import nn
import torch
from torch.nn import functional as f

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=128, activation='softplus'):
        super().__init__()
        self.fc_1 = nn.Linear(in_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        if activation=='softplus':
            self.activation = f.softplus
        elif activation=='cos':
            self.activation = torch.cos
        elif activation=='tanh':
            self.activation = f.tanh
        elif activation=='relu':
            self.activation = f.relu
        elif activation=='gelu':
            self.activation = f.gelu
        elif activation=='leakyrelu':
            self.activation = lambda x: f.leaky_relu(x, negative_slope=0.2)
        else:
            raise NotImplementedError(activation + " not implemented")

    def forward(self, x):
        
        out = self.activation(self.fc_1(x))
        out = self.activation(self.fc_2(out))
        return self.fc_out(out)
        
        
class NeuralTransMat(nn.Module):
    def __init__(self, in_dim, num_states, hid_dim=128, activation='softplus'):
        super().__init__()
        self.state_transitions = nn.ModuleList([MLP(in_dim,num_states, hid_dim, activation) for i in range(num_states)])
        self.num_states = num_states

    def forward(self, x):
        out = torch.cat([self.state_transitions[i](x).unsqueeze(-2) for i in range(self.num_states)], dim=-2)
        # output is unnormalised p(z_t | z_t-1)
        # now we normalise
        return f.softmax(out, dim=-1)

class LocalLinear(nn.Module):
    """
    LocalLinear layer. Reproduces nn.Linear, but only locally across ``num_dims``.
    Code adapted from https://github.com/xunzheng/notears/blob/master/notears/nonlinear.py.
    """
    def __init__(self, num_dims, in_dim, out_dim):
        super().__init__()
        self.num_dims = num_dims
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.Tensor(num_dims,
                                                in_dim,
                                                out_dim))
        self.bias = nn.Parameter(torch.Tensor(num_dims, out_dim))

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        k = 1.0 / self.in_dim
        bound = torch.sqrt(torch.tensor(k))
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x -> (B, D, num_feat)
        out = torch.matmul(x[:,:,None,:], self.weight[None,:,:,:]).squeeze(2)
        out += self.bias[None,:,:]
        return out


class CausalMLP(nn.Module):
    def __init__(self, in_dim, hid_dim=128, activation='cos', num_lags=1):
        super().__init__()
        # Inspiration taken from https://github.com/xunzheng/notears/blob/master/notears/nonlinear.py
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_lags = num_lags
        # first layer -> NoTEARS adjacency matrix
        self.fc1_pos = nn.Linear(in_dim*num_lags, in_dim*hid_dim)
        self.fc1_neg = nn.Linear(in_dim*num_lags, in_dim*hid_dim)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()

        # Locally connected layers
        self.fc_2 = LocalLinear(in_dim, hid_dim, hid_dim)
        self.fc_out = LocalLinear(in_dim, hid_dim, 1)
        if activation=='softplus':
            self.activation = f.softplus
        elif activation=='cos':
            self.activation=torch.cos
        elif activation=='tanh':
            self.activation = f.tanh
        elif activation=='relu':
            self.activation = f.relu
        else:
            raise NotImplementedError(activation + " not implemented")

    def _bounds(self):
        d = self.in_dim
        bounds = []
        for _ in range(d*self.num_lags):
            for _ in range(self.hid_dim):
                for _ in range(d):
                    bound = (0, None)
                bounds.append(bound)
        return bounds
    
    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(torch.abs(self.fc1_pos.weight - self.fc1_neg.weight))
        return reg

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        reg += torch.sum(self.fc_2.weight ** 2)
        reg += torch.sum(self.fc_out.weight ** 2)
        return reg
    
    @torch.no_grad()
    def fc1_to_adj(self):  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.in_dim
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W  # [i, j]
        return W
    
    def forward(self, x):
        shape_in = x.shape
        x.reshape(-1, shape_in[-1])
        # x -> (B, D)
        x = self.fc1_pos(x) - self.fc1_neg(x) # -> (B, D*N)
        x = x.reshape(-1, self.in_dim, self.hid_dim) # -> (B, D, N)
        out = self.activation(x)
        out = self.activation(self.fc_2(out))
        out = self.fc_out(out).squeeze(-1)
        return out.reshape(shape_in)
    
class ResidualBlock(nn.Module):
    "Each residual block should up-sample an image x2"
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.input_channels = input_channels
        if input_channels != 64:
            self.match_conv = nn.Conv2d(in_channels=input_channels,
                        out_channels=64,
                        kernel_size=1,
                        stride=1,
                        padding=0)
    

    def forward(self, x):
        x = f.upsample_bilinear(x, scale_factor=2)
        res = self.match_conv(x) if self.input_channels!=64 else x
        x = f.leakyrelu(self.conv1(x), 0.2)
        x = self.conv2(x)
        return x + res
    
class CNNResidualDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_dim=3, n_layers=2):
        super(CNNResidualDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.first_mlp = MLP(latent_dim, hidden_dim*4*4, activation='leakyrelu')
        self.first_block = ResidualBlock(input_channels=hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(input_channels=hidden_dim)
        for i in range(n_layers)])
        self.out_conv = nn.Conv2d(in_channels=hidden_dim,
                                  out_channels=out_dim,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.first_mlp(x).reshape((b, -1, 4, 4))
        x = self.first_block(x)
        for residual_layer in self.residual_blocks:
            x = residual_layer(x)
        x = self.out_conv(x)
        return x

class CNNFastDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, n_layers=2):
        super(CNNFastDecoder, self).__init__()
        self.in_dec = MLP(input_dim, hidden_size*8*8, activation='leakyrelu')
        self.hidden_conv_1 = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        for _ in range(n_layers)])
        self.hidden_conv_2 = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        for _ in range(n_layers)])
        self.out_conv = nn.Conv2d(in_channels=hidden_size,
                      out_channels=output_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.in_dec(x).reshape((b, -1, 8, 8))
        for hidden_layer_1, hidden_layer_2 in zip(self.hidden_conv_1, self.hidden_conv_2):
        #for hidden_layer_1 in self.hidden_conv_1:
            x = f.leaky_relu(hidden_layer_1(x), 0.2)
            x = f.pad(x, (0,1,0,1))
            x = f.leaky_relu(hidden_layer_2(x), 0.2)
        x = self.out_conv(x)
        return x

class CNNFastEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, hidden_size, n_layers=2):
        super(CNNFastEncoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=hidden_size,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.hidden_conv_1 = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        for _ in range(n_layers)])
        self.hidden_conv_2 = nn.ModuleList([
            nn.Conv2d(in_channels=hidden_size,
                      out_channels=hidden_size,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        for _ in range(n_layers)])

        self.out = MLP(hidden_size*4*4, output_dim, hidden_size, 'leakyrelu')

    def forward(self, x):
        x = f.relu(self.in_conv(x))
        for hidden_layer_1, hidden_layer_2 in zip(self.hidden_conv_1, self.hidden_conv_2):
        #for hidden_layer_1 in self.hidden_conv_1:
            x = f.leaky_relu(hidden_layer_1(x), 0.2)
            x = f.leaky_relu(hidden_layer_2(x), 0.2)
        x = x.flatten(-3, -1)
        return self.out(x)


if __name__ == "__main__":

    x = torch.randn(10,20,2)
    model = CausalMLP(2,16)
    print(model(x).shape)
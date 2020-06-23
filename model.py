import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

class HF(nn.Module):
    def __init__(self):
        super(HF, self).__init__()

    def forward(self, i, v, s, H):
        '''
        :param v: batch_size (B) x latent_size (L)
        :param s: batch_size (B) x latent_size (L)
        :return: s_new = s - 2 * v v_T / norm(v,2) * s
        '''
        K = v.shape[1]
        vvT = torch.bmm( v.unsqueeze(2), v.unsqueeze(1) )  # v . v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
        norm_sq = torch.sum( v * v, 1 ) # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.unsqueeze(-1).unsqueeze(-1).expand( norm_sq.size(0), K, K) # expand sizes : B x L x L
        H[str(i)] = torch.eye(K, K).to(dev) - 2 * vvT / norm_sq # (B, L, L)
        s_new = torch.bmm(H[str(i)], s.reshape(s.shape[0], -1).unsqueeze(2)).squeeze(2)
        return s_new


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, bias=True, groups=1):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)


    def forward(self, x):
        return F.conv2d(x, self.weight,
                        self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, alpha=1., num_flows=2):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))

        self.max_alpha = alpha
        log_alpha = (torch.ones(in_features) * alpha).log()
        self.log_alpha_ = nn.Parameter(log_alpha)
        self.mu_ = nn.Parameter(torch.ones(in_features))


        # Householder flow
        self.num_flows = num_flows
        self.HF = HF()
        self.v_layers = nn.ModuleList()
        # T >0
        for i in range(0, self.num_flows):
            self.v_layers.append(nn.Linear(in_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def q_s_Flow(self, s, x, H):
        v = {}
        # Householder Flow:
        if self.num_flows > 0:
            v['0'] = x
            for i in range(0, self.num_flows):
                v[str(i + 1)] = self.v_layers[i](v[str(i)])
                v[str(i + 1)] = F.leaky_relu(v[str(i + 1)])
                s[str(i + 1)] = self.HF(i + 1, v[str(i + 1)], s[str(i)], H)
            return s[str(self.num_flows)]
        return s['0']

    def forward(self, x):
        if self.num_flows == 0:
            self.log_alpha_.data = torch.clamp(self.log_alpha_.data, max=math.log(self.max_alpha))
        alpha = self.log_alpha_.exp()

        s = {}
        H = {}
        # print(alpha.shape, x.shape)
        s['0'] = torch.sqrt(alpha) * torch.randn(x.size()).to(dev) + 1

        if self.num_flows > 0:
            s_K = self.q_s_Flow(s, x, H)
            self.U = H[str(self.num_flows)]
            for i in reversed(range(1, self.num_flows)):
                self.U = torch.bmm(self.U, H[str(i)])
            X_noised = x * s_K
        else:
            X_noised = x * s['0']

        activation = F.linear(X_noised, self.W)
        return activation + self.bias

    def kl_reg(self):
        if self.num_flows == 0:
            c1 = 1.16145124
            c2 = -1.50204118
            c3 = 0.58629921
            alpha = self.log_alpha_.exp()
            negative_kl = 0.5 * self.log_alpha_ + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3
            kl = -negative_kl
            # print('alpha, kl fc: ', alpha.mean().data, kl.mean().data)
            return kl.sum()

        if self.num_flows > 0:
            alpha = self.log_alpha_.exp()
            # print(alpha.max())
            M, K = self.U.shape[:2]
            kl = torch.log((torch.sum(self.U, dim=-1) ** 2 + torch.sum(alpha * self.U**2, dim=-1)) / alpha.unsqueeze(0).expand(M, K))
            kl = torch.sum(kl, dim=-1)
            kl = kl * self.out_features
            return kl.sum() / 2

# Define a simple 4 layer Network
class NetFC(nn.Module):
    def __init__(self, drop_prob=0.5, num_flows=2):
        super(NetFC, self).__init__()
        hidden_size = 1060
        self.fc1 = Linear(28 * 28, hidden_size, alpha=0.05 / (1 - 0.05),  num_flows=num_flows)
        self.fc2 = Linear(hidden_size, hidden_size, alpha=drop_prob / (1 - drop_prob),  num_flows=num_flows)
        self.fc3 = Linear(hidden_size, hidden_size, alpha=drop_prob / (1 - drop_prob))
        self.fc4 = Linear(hidden_size, 10, alpha=drop_prob / (1 - drop_prob))
        # self.threshold = threshold

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class NetCNN(nn.Module):
    def __init__(self, drop_prob=0.5, num_flows=2):
        super(NetCNN, self).__init__()
        self.conv1 = Conv2d(3, 32, stride=2)
        self.conv2 = Conv2d(32, 64, stride=2)
        self.l1 = Linear(64 * 7 * 7, 128, alpha=drop_prob / (1 - drop_prob),  num_flows=num_flows)
        self.l2 = Linear(128, 10, alpha=drop_prob / (1 - drop_prob),  num_flows=num_flows)
        self._init_weights()

    def forward(self, input):
        out = F.relu(self.conv1(input.to(dev)))
        # out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.l1(out))
        return self.l2(out)

    def _init_weights(self):
        for layer in self.children():
            if hasattr(layer, 'weight'): nn.init.xavier_uniform(layer.weight, gain=nn.init.calculate_gain('relu'))


class Learner(nn.Module):
    def __init__(self, net, num_batches, num_samples):
        super(Learner, self).__init__()
        self.num_batches = num_batches
        self.num_samples = num_samples
        self.net = net

    def forward(self, input, target, kl_weight=0.1):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
        kl = kl / self.num_samples
        cross_entropy = F.cross_entropy(input, target)
        elbo = - cross_entropy - kl
        return cross_entropy + kl_weight * kl, elbo



import torch
import torch.nn as nn
import torch.nn.functional as F


class DSA(nn.Module):
    def __init__(self,
                 in_channels=None,
                 insert_place=None,
                 U_segment=4,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 alpha=2,
                 split_ratio=1.0,
                 with_identity=False,
                 **kwargs):
        super(DSA, self).__init__()
        self.in_channels = in_channels
        self.U_segment = U_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.insert_place = insert_place
        self.split_ratio = split_ratio
        self.with_identity = with_identity
        print('kernel_size: {}.'.format(kernel_size))
        print('insert_place: {}'.format(insert_place))
        print('alpha: {}'.format(alpha))
        print('split_ratio: {}'.format(split_ratio))
        print('with_identity: {}'.format(with_identity))

        self.generate = nn.Sequential(
            nn.Linear(U_segment, U_segment * alpha, bias=False),
            nn.BatchNorm1d(U_segment * alpha),
            nn.ReLU(inplace=True),
            nn.Linear(U_segment * alpha, kernel_size, bias=False),
            nn.Softmax(-1))

        # split channel
        num_shift_channel = int(in_channels * split_ratio)
        self.num_shift_channel = num_shift_channel
        assert self.num_shift_channel != 0
        self.split_sizes = [num_shift_channel, in_channels - num_shift_channel]

    def forward(self, x):
        x = list(x.split(self.split_sizes, dim=1))
        x[0] = self.split_channel_forward(x[0])
        x = torch.cat(x, dim=1)
        return x

    def split_channel_forward(self, x):
        if self.insert_place == 'bottom' or self.with_identity:
            identity = x
        nu, c, t, h, w = x.size()
        u = self.U_segment
        n_batch = nu // self.U_segment
        new_x = x.view(n_batch, u, c, t, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
        out = F.adaptive_avg_pool3d(new_x.view(n_batch * c, u, t, h, w), (1, 1, 1))
        out = out.view(-1, u)
        conv_kernel = self.generate(out.view(-1, u)).view(n_batch * c, 1, -1, 1, 1)
        out = F.conv3d(new_x.view(1, n_batch * c, u, t, h * w),
                           conv_kernel,
                       bias=None,
                       stride=(self.stride, 1, 1),
                       padding=(self.padding, 0, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, u, t, h, w)
        out = out.permute(0, 2, 1, 3, 4, 5).contiguous().view(nu, c, t, h, w)

        if self.insert_place == 'bottom' or self.with_identity:
            out = out+identity

        return out


class DSABasicblock(nn.Module):
    def __init__(self,
                 block,
                 insert_place='res',
                 **kwargs
                 ):
        super(DSABasicblock, self).__init__()
        self.block = block
        assert insert_place in ['res', 'res_1', 'res_2', 'res_3','bottom']
        self.insert_place = insert_place

        if insert_place == 'res' or insert_place == 'res_1':
            in_channels = block.conv1.in_channels
        elif insert_place == 'res_2':
            in_channels = block.conv1.out_channels
        elif insert_place == 'res_3':
            in_channels = block.conv2.out_channels
        elif insert_place == 'bottom':
            in_channels = block.conv2.out_channels
        else:
            raise KeyError

        self.dsa = DSA(in_channels=in_channels,
                         insert_place=insert_place,
                         **kwargs)


    def forward(self, x):
        identity = x
        if self.insert_place=='res' or self.insert_place=='res_1':
            out = self.dsa(x)

        out = self.block.conv1(out) if self.insert_place=='res' or self.insert_place=='res_1' else self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        if self.insert_place=='res_2':
            out = self.dsa(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.insert_place=='res_3':
            out = self.dsa(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        if self.insert_place=='bottom':
            out = self.dsa(out)
        return out

    def init_weights(self):
        pass



class DSABottleneck(nn.Module):
    def __init__(self,
                 block,
                 insert_place='res',
                 **kwargs
                 ):
        super(DSABottleneck, self).__init__()
        self.block = block
        assert insert_place in ['res', 'res_1', 'res_2', 'res_3', 'res_4', 'bottom']
        self.insert_place = insert_place

        if insert_place == 'res' or insert_place == 'res_1':
            in_channels = block.conv1.in_channels
        elif insert_place == 'res_2':
            in_channels = block.conv1.out_channels
        elif insert_place == 'res_3':
            in_channels = block.conv2.out_channels
        elif insert_place == 'res_4':
            in_channels = block.conv3.out_channels
        elif insert_place == 'bottom':
            in_channels = block.conv3.out_channels
        else:
            raise KeyError

        self.dsa = DSA(in_channels=in_channels,
                         insert_place=insert_place,
                       **kwargs)

    def forward(self, x):
        identity = x
        if self.insert_place=='res' or self.insert_place=='res_1':
            out = self.dsa(x)

        out = self.block.conv1(out) if self.insert_place=='res' or self.insert_place=='res_1' else self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        if self.insert_place=='res_2':
            out = self.dsa(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        if self.insert_place=='res_3':
            out = self.dsa(out)

        out = self.block.conv3(out)
        out = self.block.bn3(out)

        if self.insert_place=='res_4':
            out = self.dsa(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        if self.insert_place=='bottom':
            out = self.dsa(out)
        return out

    def init_weights(self):
        pass


def Make_DSA(block, depth=None, **kwargs):
    assert depth in [18, 50]
    if depth == 18:
        return DSABasicblock(block, **kwargs)
    elif depth == 50:
        return DSABottleneck(block, **kwargs)
    else:
        NotImplementedError

if __name__ == '__main__':
    input = torch.rand(16, 512, 4, 7, 7)
    dsa = DSA(in_channels=512,
                insert_place='bottom',
                split_ratio=1/4,
                with_identity=False)
    out = dsa(input)
    print(out.shape)
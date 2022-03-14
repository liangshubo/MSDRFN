import  torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self,in_channel,out_channel,ker_size,
                 size_unchange = True,act = True):
        super(BasicConv, self).__init__()

        if size_unchange:
            padding = ker_size // 2
        else:padding = 0

        conv = []
        conv.append(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                              kernel_size=ker_size,stride=1,padding=padding))
        if act:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)
    def forward(self,x):
        return self.conv(x)


class Reconstruct(nn.Module):
    def __init__(self,scale, n_feat,out_channel):
        super(Reconstruct, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                #m.append(BasicConv(n_feat, 4 * n_feat, 3,act=False))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            #m.append(BasicConv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        self.m = nn.Sequential(*m)
        self.conv_end = BasicConv(n_feat//(scale**2),out_channel,1,act=True)
        self.scale = scale

    def forward(self,x):

        out = self.m(x)
        out = self.conv_end(out)
        return out


# MSFE Mutil-Scale feature extraction

class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel,ker_size,
                 size_unchange = True,act = True):
        super(DoubleConv, self).__init__()
        self.conv1 = BasicConv(in_channel,out_channel,ker_size
                               ,size_unchange,act=False)
        self.conv2 = BasicConv(out_channel,out_channel,ker_size
                               ,size_unchange,act=True)
    def forward(self,x):
        return self.conv2(self.conv1(x))

class Inception(nn.Module):

    def __init__(self,in_channel,out_channel):
        super(Inception, self).__init__()
        self.branch1 = DoubleConv(in_channel,out_channel, 3)
        self.branch2 = DoubleConv(in_channel,out_channel, 5)
        self.branch3 = DoubleConv(in_channel, out_channel,7)
    def forward(self,x):

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return [out1,out2,out3]


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class MSFEB(nn.Module):
    def __init__(self,in_cannel,conv_out_channel,reduction = 4):

        super(MSFEB, self).__init__()
        self.Inception1 = Inception(in_cannel,conv_out_channel)
        self.Inception2 = Inception(conv_out_channel*3+in_cannel,conv_out_channel)
        self.SEnet = CALayer(in_cannel,reduction)
        self.conv_end = nn.Conv2d(conv_out_channel*3,in_cannel,1)

    def MSFE(self,x):
        res = x
        out11,out12,out13 = self.Inception1(x)
        input2 = torch.cat([out11,out12,out13,res],dim=1)
        out21,out22,out23 = self.Inception2(input2)
        input3 = torch.cat([out21,out22,out23],dim=1)
        output = self.conv_end(input3)
        return output

    def forward(self,x):
        res = x
        feature = self.MSFE(x)
        feature = res + feature
        output = self.SEnet(feature)
        return output


class MSRFN(nn.Module):
    def __init__(self,in_channel,n_feats,msconv_channel,scale = 4,n_block = 4):
        super(MSRFN, self).__init__()
        self.SFE = nn.Conv2d(in_channel,n_feats,kernel_size=1)
        body = []
        body = [ MSFEB(n_feats,msconv_channel ) for _ in range( n_block)]
        res = []
        res = [Reconstruct(scale,n_feats,in_channel) for _ in range(n_block)]
        res.append(Reconstruct(scale,n_feats*n_block,in_channel))

        self.res = nn.Sequential(*res)
        self.body = nn.Sequential(*body)
        self.scale =scale

    def gloab_feature_exact(self,x):
        res = []
        for layer in self.body:
            x = layer(x)
            res.append(x)
        res.append(torch.cat(res,dim=1))
        return res

    def gloab_feature_fusion(self,feature_list):
        for i in feature_list:
            print(i.shape)
        out = 0
        for i in range(len(feature_list)):
            out += self.res[i](feature_list[i])
        return out
    def forward(self,x):
        bicubic = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
        feature1 = self.SFE(x)
        res = self.gloab_feature_exact(feature1)
        out = self.gloab_feature_fusion(res)
        out = out + bicubic

        return out
def make_model(args):
    return MSRFN(in_channel=3,n_feats=64,msconv_channel=32,scale=4)
    	

if __name__ == '__main__':

    x = torch.randn([1, 3, 64, 64])
    model = MSRFN(in_channel=3,n_feats=64,msconv_channel=32,scale=4)
    print(model)
    out = model(x)
    print(out.shape)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))






# 测试
# 开发时间：2023/2/27 20:16
import math
import torch
import numbers
from einops import rearrange

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
# from blindSR.utils import LayerNorm, GRN
# from models.basicblock import  BSConvURB,CCALayer
# from models.CNN_SR import PSAModule
# from models.SwinT import SwinT
# from models.basicblock import duoInput_Attention,ResidualPA,PALayer,CCALayer
# fzf: from swinUnetSR.restormer_swin import ParaHybridStage_back,ParaHybridStage,PPM
# from SRModel.LKFormer import ParaHybridStage_back,ParaHybridStage,PPM
from timm.models.layers import to_2tuple, trunc_normal_

# inplans, planes: 输入通道，输出通道
# 不会改变输入的H和W
##### ParaHybrid block
class ParaHybridBlock_back(nn.Module):
    def __init__(self, dim, head=4,ffn_expansion_factor=4, distillation_rate=0.25, LayerNorm_type='WithBias',bias=False):
        super(ParaHybridBlock_back, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        # self.attn = Attention_back(dim,num_heads=head)
        # fzf   LKA_back_new_attn:LKRA
        self.attn = LKA_back_new_attn(dim)  #LKA_back_new_attn inceptionAttn_back
        # self.CA = CAB_back(dim)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # fzf   inceptionAttn_back:GPFN
        self.ffn = inceptionAttn_back(dim) # inceptionAttn_back
        # self.ffn = LKA_back_new_attn_no(dim)

    def forward(self, x):
        x1 = self.norm1(x)
        x = x + self.attn(x1) # + self.CA(x1)

        x = x + self.ffn(self.norm2(x))

        return x

class inceptionAttn_back(nn.Module):
    def __init__(self,dim=64):
        super(inceptionAttn_back, self).__init__()
        hiddenC = int(2*dim)
        self.conv1_1 = nn.Conv2d(dim, hiddenC, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        # self.attn = MSPA_back(hiddenC) ##默认
        self.attn = MSPA_new_back(hiddenC) ### used
        self.conv1_2 = nn.Conv2d(hiddenC, dim, kernel_size=1, bias=True)
    def forward(self,x):
        # shorcut = x.clone()
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.attn(x)
        x = self.conv1_2(x)
        return x

class MSPA_new_back(nn.Module):
    def __init__(self, dim, head=4,distillation_rate=0.25, bias=False):
        super(MSPA_new_back, self).__init__()
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        hiddenF = int(dim) # 可以尝试扩大通道数

        self.conv1_3 = nn.Conv2d(dim, hiddenF, kernel_size=1, bias=bias)
        self.Conv3 = nn.Conv2d(hiddenF, hiddenF, kernel_size=3, padding=1, stride=1,
                                 dilation=1, groups=hiddenF)

        # self.act = nn.SiLU(inplace=True)

        # self.conv1_out = nn.Conv2d(int(5*hiddenF), dim, kernel_size=1, bias=bias)
        self.conv1_out = nn.Conv2d(int(1*hiddenF), dim, kernel_size=1, bias=bias)
        # self.CA = CCALayer(dim) int(4*hiddenF)

    def forward(self, x):
        # x1 = x.clone()
        x1 = self.conv1(x)

        c1 = self.conv1_3(x)
        c1 = self.Conv3(c1)
        # attn1 = torch.cat([ c2, c3], 1) ## c1, c2, c3, c4
        attn1 = self.conv1_out(c1) # attn1


        out1 = attn1*x1

        return out1

class LKA_back_new_attn(nn.Module):
    def __init__(self, dim):
        super(LKA_back_new_attn,self).__init__()
        hidden = int(2*dim)
        padding2 = (11 // 2, 1 // 2)
        padding1 = (1 // 2, 11 // 2)
        self.conv1_0 = nn.Conv2d(dim, dim, kernel_size=1)

        self.conv1_1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.SiLU()

        self.conv_spatial = nn.Conv2d(hidden, hidden, 7, stride=1, padding=3, groups=hidden)

        self.conv1_4 = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.Conv11 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=(1, 11), padding=padding1, stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(11, 1), padding=padding2, stride=1,
                                              dilation=1, groups=hidden))
        self.Conv21 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),nn.Conv2d(hidden, hidden, kernel_size=(1, 21), padding=(0, int(21 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(21, 1), padding=(int(21 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.Conv31 = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1),
                                    nn.Conv2d(hidden, hidden, kernel_size=(1, 31), padding=(0, int(31 // 2)), stride=1,
                                              dilation=1, groups=hidden),
                                    nn.Conv2d(hidden, hidden, kernel_size=(31, 1), padding=(int(31 // 2), 0), stride=1,
                                              dilation=1, groups=hidden))

        self.conv1_5 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = self.conv1_0(x)

        attn = self.conv1_1(x)
        attn = self.act(attn)
        attn = self.conv_spatial(attn)


        u3 = attn.clone()
        attn = self.conv1_4(attn)
        attn = self.Conv11(attn)
        attn = self.act(attn + u3)

        u4 = attn.clone()
        attn = self.Conv21(attn)
        attn = self.act(attn + u4)

        u5 = attn.clone()
        attn = self.Conv31(attn)
        attn = self.act(attn + u5)

        # u6 = attn.clone()
        # attn = self.Conv41(attn)
        # attn = self.act(attn + u6)

        attn = self.conv1_5(attn)

        out1 = u * attn
        out1 = self.proj_1(out1)
        return out1
    
#### stage block back
class ParaHybridStage_back(nn.Module):
    def __init__(self, dim, depth=4):
        super(ParaHybridStage_back, self).__init__()

        # self.basicBlock = nn.Sequential(*[
        #     ParaHybridBlock(dim=dim,head=heads[i]) for i in range(depth)])
        # fzf   ParaHybridBlock_back:TL
        self.B1 = ParaHybridBlock_back(dim=dim)

        self.B2 = ParaHybridBlock_back(dim=dim)

        self.B3 = ParaHybridBlock_back(dim=dim)

        self.B4 = ParaHybridBlock_back(dim=dim)

        self.B5 = ParaHybridBlock_back(dim=dim)

        self.B6 = ParaHybridBlock_back(dim=dim)
        #
        # self.B7 = ParaHybridBlock_back(dim=dim)
        #
        # self.B8 = ParaHybridBlock_back(dim=dim)
        ### new
        # self.Conv1_out = nn.Conv2d(int(6*dim), dim, kernel_size=1)

        self.lastConv = nn.Conv2d(dim,dim,3,1,1)
        ###new
        # self.inPA = PALayer(dim)

    def forward(self, x):

        x1 = self.B1(x)

        # tem1 = self.Conv1_1(tem1)
        x2 = self.B2(x1)

        x3 = self.B3(x2)
        #
        x4 = self.B4(x3)

        x5 = self.B5(x4)

        x6 = self.B6(x5)
        out1 = self.lastConv(x6)+x
        return out1

#### pool
class PPM(nn.Module):
    def __init__(self, in_dim,  bins=(1, 2, 3, 6),LayerNorm_type='WithBias'):
        super(PPM, self).__init__()
        reduction_dim = int(in_dim / len(bins))
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                LayerNorm(reduction_dim, LayerNorm_type),
                nn.SiLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
    
class ResidualPA(nn.Module):
    def __init__(self, dim, hidden_features, bias=True):
        super(ResidualPA, self).__init__()

        # hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=7, stride=1, padding=3,
                                groups=hidden_features * 2, bias=bias)

        self.conv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)
        self.act = torch.nn.SiLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,bias=bias)
        self.sigmoid = nn.Sigmoid()

        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # input = x.clone()
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.act(self.conv1(x1))
        x1 = self.sigmoid(self.conv3(x1))
        x = x1 * x2
        # x = self.project_out(x)
        return x
    
# 计算多输入间的权重矩阵
class duoInput_Attention(nn.Module):
    def __init__(self, dim,bias=False):
        super(duoInput_Attention, self).__init__()
        # self.num_heads = num_heads
        # self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_q = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, (1,7), 1, padding=(0,3)),nn.Conv2d(dim, dim, (7,1), 1, padding=(3,0)))

        self.conv_k = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, (1,7), 1, padding=(0,3)),nn.Conv2d(dim, dim, (7,1), 1, padding=(3,0)))

        self.conv_v = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1),nn.Conv2d(dim, dim, (1,7), 1, padding=(0,3)),nn.Conv2d(dim, dim, (7,1), 1, padding=(3,0)))

        self.act = nn.GELU()

        self.project_out1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x1,x2,x3):
        # b, c, h, w = x.shape

        q = self.act(self.conv_q(x1))
        k = self.act(self.conv_k(x2))
        v = self.act(self.conv_v(x3))

        q1 = q.clone()
        k1 = k.clone()
        v1 = v.clone()

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)

        attn1 = (q @ k.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out1_1 = (attn1 @ q1)
        attn1 = (q @ v.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out1_2 = (attn1 @ q1)
        out1 = out1_1 + out1_2

        attn2 = (k @ q.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2_1 = (attn2 @ k1)
        attn2 = (k @ v.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out2_2 = (attn2 @ k1)
        out2 = out2_1 + out2_2

        attn3 = (v @ q.transpose(-2, -1))
        attn3 = attn3.softmax(dim=-1)
        out3_1 = (attn3 @ v1)
        attn3 = (v @ k.transpose(-2, -1))
        attn3 = attn3.softmax(dim=-1)
        out3_2 = (attn3 @ v1)
        out3 = out3_1 + out3_2

        out1 = self.project_out1(out1) + x1
        out2 = self.project_out2(out2) + x2
        out3 = self.project_out3(out3) + x3

        return out1,out2,out3
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class GRN(nn.Module):
    """
    Generalized Residual Normalization (GRN).
    GRN normalizes spatial dimensions (H, W) of the input for each channel,
    and scales the normalized features with a learnable parameter alpha.
    
    Args:
        dim (int): Number of input channels (C).
        eps (float): A small value to avoid division by zero. Default: 1e-6.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))  # Learnable scaling parameter
        self.beta = nn.Parameter(torch.zeros(dim))  # Learnable bias parameter

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (N, H, W, C) or (N, C, H, W).
        
        Returns:
            Tensor: GRN-normalized tensor with the same shape as input.
        """
        # Step 1: Compute the L2 norm across spatial dimensions (H, W)
        if x.ndim == 4:  # Assume input is (N, C, H, W)
            spatial_dims = (2, 3)  # Normalize along H and W
        else:
            raise ValueError("Input tensor must be 4D (N, C, H, W).")

        # Compute the L2 norm for each channel
        l2_norm = torch.sqrt(torch.sum(x ** 2, dim=spatial_dims, keepdim=True) + self.eps)

        # Step 2: Normalize the input
        x_normalized = x / l2_norm

        # Step 3: Re-scale and shift using learnable parameters
        # Alpha and Beta are applied per channel
        x_out = self.alpha.unsqueeze(-1).unsqueeze(-1) * x_normalized + self.beta.unsqueeze(-1).unsqueeze(-1)
        
        return x_out  

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
    
class PSAModule(nn.Module):
    # dim=96 conv_groups=[1, 4, 8, 12], dim=64 conv_groups=[1, 4, 8, 16]
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out

class SwinT(nn.Module):
    def __init__(
            # self, conv, n_feats, kernel_size,
            # bias=True, bn=False, act=nn.ReLU(True)):
            self,  embed_dim=64, heads=8):

        super(SwinT, self).__init__()
        m = []
        depth = 2
        num_heads = heads
        window_size = 8
        resolution = 64
        mlp_ratio = 2.0
        m.append(BasicLayer(dim=embed_dim,
                            depth=depth,
                            resolution=resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            norm_layer=nn.LayerNorm))
        self.transformer_body = nn.Sequential(*m)

    def forward(self, x):
        res = self.transformer_body(x)
        return res
class BasicLayer(nn.Module):
    def __init__(self, dim, resolution, embed_dim=50, depth=2, num_heads=8, window_size=8,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, norm_layer=None):

        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.window_size = window_size
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, resolution=resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.patch_embed = PatchEmbed(
            embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, h, w

    def forward(self, x):
        x, h, w = self.check_image_size(x)
        _, _, H, W = x.size()
        x_size = (H, W)
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, x_size)
        x = self.patch_unembed(x, x_size)
        if h != H or w != W:
            x = x[:, :, 0:h, 0:w].contiguous()
        return x

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
            # self.norm2 = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops   
class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, pretrained_window_size=0,drop=0.,attn_drop=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.resolution = to_2tuple(resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)

        # self.attn = WindowAttention_v2(
        #     dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        #     pretrained_window_size=to_2tuple(pretrained_window_size))

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        # x = x + self.mlp(x)
        return x
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True,
                 qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class LargeKernelConv(nn.Module):
    def __init__(self, dim, kernel_size, small_kernel):
        super(LargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # self.Decom = Decom
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding1 = (kernel_size // 2, small_kernel//2)
        padding2 = (small_kernel//2, kernel_size // 2)
        self.pw1 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.pw2 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.pw3 = torch.nn.Conv2d(
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.Lconv1 = nn.Conv2d(2*dim, 2*dim, kernel_size=(kernel_size,small_kernel), padding=padding1,stride=1, dilation=1,groups=dim)
        self.Lconv2 = nn.Conv2d(2*dim, 2*dim, kernel_size=(small_kernel,kernel_size), padding=padding2,stride=1, dilation=1,groups=dim)
        self.Sconv = nn.Conv2d(2*dim, 2*dim, kernel_size=(small_kernel,small_kernel), padding=small_kernel//2,stride=1, dilation=1,groups=dim)

        # self.act1 = nn.GELU()  # SiLU
        # self.act2 = nn.GELU()  # SiLU
        # self.act3 = nn.GELU()  # SiLU

    def forward(self, x):
        x1 = self.pw1(x)
        x1 = self.Lconv1(x1)
        # x1 = x+x1
        # x1 = self.act1(x1)

        x2 = self.pw2(x)
        x2 = self.Lconv2(x2)


        x3 = self.pw3(x)
        x3 = self.Sconv(x3)

        return x1 + x2 + x3

# 大核卷积块
class LBlock(nn.Module):
    r""" SLaK Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, large_kernel=7,small_kernel=7):
        super().__init__()

        self.large_kernel = LargeKernelConv(dim=dim,kernel_size=large_kernel,small_kernel=small_kernel)

        self.norm = LayerNorm(2*dim, eps=1e-6)
        self.pwconv1 = nn.Linear(2*dim, 8 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(8 * dim, 4*dim)
        self.act2 = nn.GELU()
        self.pwconv3 = nn.Linear(4 * dim, 1 * dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.large_kernel(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.act2(x)
        x = self.pwconv3(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class R_LBlock(nn.Module):
    def __init__(self, dim, depth=4, large_kernel=7,small_kernel=7,useSwin=False):
        super().__init__()
        # build blocks
        # self.blocks = nn.ModuleList([
        #     LBlock(dim=dim,large_kernel=large_kernel,small_kernel=small_kernel)
        #     for i in range(depth)])
        if useSwin:
            self.blocks = nn.Sequential(*[SwinT(dim) for j in range(depth)])
        else:
            self.blocks = nn.Sequential(
                *[LBlock(dim=dim,large_kernel=large_kernel,small_kernel=small_kernel)
                    for i in range(depth)])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        # for blk in self.blocks:
        #     x = blk(x)
        x1 = self.blocks(x)
        x = self.conv(x1) + x
        return x

class HybridBlock(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 可以考虑添加一个Conv1的卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.swinT = SwinT(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x) + x
        x = self.swinT(x) + x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# 改成先全局再局部
class all_local_Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 可以考虑添加一个Conv1的卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.swinT = SwinT(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # x = self.dwconv(x) + x
        x = self.swinT(x) + x

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 可以考虑添加一个Conv1的卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class FBlock(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, useCA=False,useShffule=False,last=False,depth=[2,4,2],drop_path=0.):
        super(FBlock,self).__init__()

        self.useCA = useCA
        self.useShffule = useShffule
        self.islast = last

        if self.islast:
            self.conv_last1 = nn.Conv2d(dim, 3, 3, 1, 1)
            self.conv_last2 = nn.Conv2d(dim, 3, 3, 1, 1)
            self.conv_last3 = nn.Conv2d(dim, 3, 3, 1, 1)
        # self.norm_layer1 = nn.Sequential(
        #     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
        #     nn.Conv2d(dim, dim, kernel_size=1),
        # )
        self.norm_layer1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.block1 = nn.Sequential(*[Block(dim,drop_path) for j in range(depth[0])])
        # *[FBlock(dim=embed_dim, useShffule=True) for j in range(depths[i])]
        # 下采样 2*dim
        self.downConv1 = nn.Sequential(
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 2*dim, kernel_size=2, stride=2),
        )
        ########pixel shuffle
        # m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        # m.append(nn.PixelShuffle(scale))
        if useShffule and not last:
            self.transC1 = nn.Conv2d(2*dim, 4 * dim, kernel_size=1)
            self.pixelShullfe1 = nn.PixelShuffle(2)
        else:
            self.transC1 = nn.Conv2d(2*dim, dim, kernel_size=1)

        self.block2 = nn.Sequential(*[Block(2*dim,drop_path) for j in range(depth[1])])
        # 下采样 4*dim
        self.downConv2 = nn.Sequential(
            LayerNorm(2*dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(2*dim, 4*dim, kernel_size=2, stride=2),
        )

        if useShffule and not last:
            self.transC2 = nn.Conv2d(4*dim, 4 * 2*dim, kernel_size=1)
            self.pixelShullfe2 = nn.PixelShuffle(2)
        else:
            self.transC2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # self.norm_layer3 = nn.Sequential(
        #     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
        #     nn.Conv2d(dim, dim, kernel_size=1),
        # )
        self.block3 = nn.Sequential(*[HybridBlock(4*dim,drop_path) for j in range(depth[2])])

        # self.conv1 = nn.Conv2d(dim*7, dim, kernel_size=1)

        if useCA:
            self.CA = PSAModule(dim,dim)


    def forward(self, x):
        x1 = self.norm_layer1(x)
        x1 = self.block1(x1)
        x1_down = self.downConv1(x1) # 2 dim

        # x2 = self.norm_layer2(x1_down)
        x2 = self.block2(x1_down)
        x2_down = self.downConv2(x2) # 4 dim

        # x3 = self.norm_layer3(x2_down)
        x3 = self.block3(x2_down)

        # 融合前三个block的输出
        # F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        ############## Sequentially Add
        # 也可以考虑使用注意力机制去融合这些特征
        # 考虑使用 pixelShuffle 进行上采样
        if self.islast:
            ############## sum融合
            x2 = self.transC1(x2)
            x3 = self.transC2(x3)
            x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3_up = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x1_ori = self.conv_last1(x1)
            x2_ori = self.conv_last2(x2_up)
            x3_ori = self.conv_last3(x3_up)
            return x1_ori + x2_ori + x3_ori
        else:
            x3 = self.transC2(x3)
            if self.useShffule:
                x3_up = self.pixelShullfe1(x3)
            else:
                x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = x3_up + x2
            x4 = self.transC1(x4)
            if self.useShffule:
                x4_up = self.pixelShullfe2(x4)
            else:
                x4_up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
            x_out = x4_up + x1

            return x_out  # 内部可以考虑加一个 short skip connection


# 多路分支注意力Block
class PBlock(nn.Module):
    """ 多路分支.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, pool_size=3, useCA=False,usePA=False,drop_path=0.):
        super(PBlock,self).__init__()
        # self.norm_layer1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # self.conv1_first = nn.Conv2d(dim,2*dim,kernel_size=1)
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=2 * dim, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(in_channels=2*dim, out_channels=(dim//2)*3, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=(dim//2)*3, out_channels=dim, kernel_size=(1, 1), stride=1, padding=0, dilation=1, groups=1, bias=False))

        self.path2 = nn.Sequential(nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=(1, 1),stride=1,padding=0,dilation=1,groups=1,bias=False),
                                   nn.Conv2d(2*dim, 2*dim, kernel_size=7, padding=3, groups=dim),
                                   nn.Conv2d(in_channels=2 * dim, out_channels=(dim // 2) * 3, kernel_size=(1, 1),
                                             stride=1, padding=0, dilation=1, groups=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=(dim // 2) * 3, out_channels=dim, kernel_size=(1, 1), stride=1,
                                             padding=0, dilation=1, groups=1, bias=False)
                                   )

        self.path3 = nn.Sequential(nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=(1, 1),stride=1,padding=0,dilation=1,groups=1,bias=False),
                                   nn.Conv2d(2*dim, 2*dim, kernel_size=(1,13), padding=(0,13//2), groups=dim),
                                   nn.Conv2d(2*dim, 2*dim, kernel_size=(13,1), padding=(13//2,0), groups=dim),
                                   nn.Conv2d(in_channels=2 * dim, out_channels=(dim // 2) * 3, kernel_size=(1, 1),
                                             stride=1, padding=0, dilation=1, groups=1, bias=False),
                                   nn.GELU(),
                                   nn.Conv2d(in_channels=(dim // 2) * 3, out_channels=dim, kernel_size=(1, 1), stride=1,
                                             padding=0, dilation=1, groups=1, bias=False)
                                   )

        self.path4 = nn.Sequential(nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2),
                                   nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=(1, 1),stride=1,padding=0,dilation=1,groups=1,bias=False))
        self.duoAtten = duoInput_Attention(dim)
        self.conv1_trans = nn.Conv2d(4*dim,dim,kernel_size=1)

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.useCA = useCA
        self.usePA = usePA
        if useCA:
            self.CA = PSAModule(dim,dim)
        if usePA:
            self.PA = ResidualPA(dim,dim)

    def forward(self, x):
        input = x
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)

        x1,x2,x3 = self.duoAtten(x1,x2,x3)

        x_all = torch.cat((x1,x2,x3,x4),1).contiguous()
        x_all = self.conv1_trans(x_all) + x

        x_all = x_all.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x_all = self.norm(x_all)
        x_all = self.pwconv1(x_all)
        x_all = self.act(x_all)
        x_all = self.grn(x_all)
        x_all = self.pwconv2(x_all)
        x_all = x_all.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.useCA:
            x_all = self.CA(x_all)
        if self.usePA:
            x_all = self.PA(x_all)

        x = input + self.drop_path(x_all)
        return x

class PBlock_conv(nn.Module):
    """ 多路分支残差块.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim=64, depth=6, drop_path=0.):
        super(PBlock_conv,self).__init__()
        self.basic_block = nn.Sequential(*[PBlock(dim=dim,useCA=True,usePA=True) for j in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        input = x.clone()
        x = self.basic_block(x)
        x = self.conv(x)

        return x + input


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# embed_dim 默认为 96 ### 增大通道数(128), Lblocks数=6,
class mySRNet(nn.Module):
    def __init__(self, inc=3,outc=3,upscale=2,num_feat = 128,embed_dim=96,f_groups=4,f_stages=2,Lblocks=8,layers=8 ,img_size=64,upsampler='pixelshuffle', img_range=1.,norm_layer=GroupNorm):
        super(mySRNet, self).__init__()

        self.upsampler = upsampler
        self.img_range = img_range
        self.upscale = upscale

        if inc == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        #####################################################################################################

        ################################### 1, shallow feature extraction ############
        self.conv_first = nn.Conv2d(inc, embed_dim, 3, 1, 1)
        # self.conv_first = nn.Sequential(
        #     nn.Conv2d(inc, embed_dim, 3, 1, 1),
        #     LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        # )
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        # PoolFormerBlock(dim=inc, use_layer_scale=False)
        self.L_blocks = nn.ModuleList()
        self.num_Lblocks = Lblocks
        # self.nums_layer = layers ParaHybridStage ParaHybridStage_back
        # fzf：RTB
        for i in range(self.num_Lblocks):
            layer = ParaHybridStage_back(dim=embed_dim)
            self.L_blocks.append(layer)

        self.conv1 = nn.Conv2d(int(self.num_Lblocks * embed_dim), embed_dim, kernel_size=1)  # depth*embed_dim
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1,groups=embed_dim,bias=False) ## groups=embed_dim,

        #####################################################################################################
        ################################ 3, high quality image reconstruction nn.LeakyReLU(inplace=True) ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(int(embed_dim), num_feat, 3, 1, 1),
                                                      nn.SiLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, outc, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, outc,
                                            (img_size, img_size))


    def up_forward_features(self, x):
        u = x.clone()
        # for i in range(self.num_layers):
        #     x = self.layers[i](x)
        x = self.extractFrequenceInfoB(x)

        x = self.deepLastConv(x) + u
        # 多尺度特征提取
        x = self.layers[0](x)

        # depth=6
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5]), 1).contiguous()
        return x

    def forward_features(self, x):  # 经过深层特征提取（HRBCT）之后的输出
        retainV = []
        # u = x.clone()
        for i in range(self.num_Lblocks):
            x = self.L_blocks[i](x)
            retainV.append(x)
        # 6个
        # x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5]), 1).contiguous()
        # 8个
        x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3], retainV[4], retainV[5],retainV[6], retainV[7]), 1).contiguous()
        return x1

    def DFF(self, x):  # 深度特征融合模块
        x1 = self.conv1(x)
        # 这个3x3的卷积层也可以去掉看看效果
        x1 = self.conv3(x1)
        # 这个CA和PA可以去掉之后再看看
        # x1 = self.PA(x1) # 消融实验2
        return x1

    def forward(self, x):
        # u = x.clone()
        H, W = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # 上采样重建分支
        x_up = F.interpolate(x, scale_factor=int(self.upscale), mode='bilinear', align_corners=False)

        ##### 常规重建分支
        x = self.conv_first(x)  # 经过浅层特征提取

        x = self.DFF(self.forward_features(x)) + x  # 经过深层特征提取和特征融合

        # x = self.DFF(self.forward_features(x)) + x
        # x = self.PPM(x)

        # 图像上采样重建
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        else :
            x = self.upsample(x)

        x = x + x_up

        x = x / self.img_range + self.mean

        # ILR = F.interpolate(u, scale_factor=self.scale, mode='bilinear', align_corners=False)
        # x[:, :, :H * self.upscale, :W * self.upscale] + ILR
        return x[:, :, :H * self.upscale, :W * self.upscale]



if __name__ == '__main__':


    model = mySRNet()
    device = torch.device('cuda')
    model.eval()
    model = model.cuda()
    from fvcore.nn import FlopCountAnalysis, parameter_count
    
    tensor = torch.rand(1, 3, 64, 64).cuda()
    
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs:(G) ", flops.total()/(10**9))
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:(M)", total_params/(10**6))
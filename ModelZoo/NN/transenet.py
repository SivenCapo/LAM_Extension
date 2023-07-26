from ..NN import common
import torch
import torch.nn as nn
from einops import rearrange, repeat
MIN_NUM_PATCHES = 12


def make_model(args, parent=False):
    return TransENet(args)


class BasicModule(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, block_type='basic', bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()
        self.block_type = block_type

        m_body = []
        if block_type == 'basic':
            n_blocks = 10
            m_body = [
                common.BasicBlock(conv, n_feat, n_feat, kernel_size, bias=bias, bn=bn)
                # common.ResBlock(conv, n_feat, kernel_size)
                for _ in range(n_blocks)
            ]
        elif block_type == 'residual':
            n_blocks = 5
            m_body = [
                common.ResBlock(conv, n_feat, kernel_size)
                for _ in range(n_blocks)
            ]
        else:
            print('Error: not support this type')
        self.body = nn.Sequential(*m_body)

    def forward(self, x):

        res = self.body(x)
        if self.block_type == 'basic':
            out = res + x
        elif self.block_type == 'residual':
            out = res

        return out


class TransENet(nn.Module):

    def __init__(self, conv=common.default_conv):
        super(TransENet, self).__init__()
        self.scale = 4
        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)

        rgb_mean = (0.4916, 0.4991, 0.4565)  # UCMerced data
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(1, rgb_mean, rgb_std)

        # define head body
        m_head = [
            conv(3, 64, kernel_size),
        ]
        self.head = nn.Sequential(*m_head)

        # define main body
        self.feat_extrat_stage1 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)
        self.feat_extrat_stage2 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)
        self.feat_extrat_stage3 = BasicModule(conv, n_feats, kernel_size, block_type='residual', act=act)

        reduction = 4
        self.stage1_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage2_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.stage3_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.up_conv1x1 = conv(n_feats, n_feats // reduction, 1)
        self.span_conv1x1 = conv(n_feats // reduction, n_feats, 1)

        self.upsampler = common.Upsampler(conv, self.scale, n_feats, act=False)

        # define tail body
        self.tail = conv(n_feats, 3, kernel_size)
        self.add_mean = common.MeanShift(1, rgb_mean, rgb_std, 1)

        # define transformer
        image_size = 256 // self.scale
        patch_size = 8
        dim = 512
        en_depth = 8
        de_depth = 1
        heads = 6
        mlp_dim = 512
        channels = n_feats // reduction
        dim_head = 32
        dropout = 0.0

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.patch_size = patch_size
        self.patch_to_embedding_low1 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low2 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_low3 = nn.Linear(patch_dim, dim)
        self.patch_to_embedding_high = nn.Linear(patch_dim, dim)

        self.embedding_to_patch = nn.Linear(dim, patch_dim)

        self.encoder_stage1 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage2 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_stage3 = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)
        self.encoder_up = TransformerEncoder(dim, en_depth, heads, dim_head, mlp_dim, dropout)

        self.decoder1 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)
        self.decoder2 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)
        self.decoder3 = TransformerDecoder(dim, de_depth, heads, dim_head, mlp_dim, dropout)


    def forward(self, x):

        x = self.sub_mean(x)
        x = self.head(x)

        # feature extraction part
        feat_stage1 = self.feat_extrat_stage1(x)
        feat_stage2 = self.feat_extrat_stage2(x)
        feat_stage3 = self.feat_extrat_stage3(x)
        feat_ups = self.upsampler(feat_stage3)

        feat_stage1 = self.stage1_conv1x1(feat_stage1)
        feat_stage2 = self.stage2_conv1x1(feat_stage2)
        feat_stage3 = self.stage3_conv1x1(feat_stage3)
        feat_ups = self.up_conv1x1(feat_ups)

        # transformer part:
        p = self.patch_size

        feat_stage1 = rearrange(feat_stage1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        feat_stage2 = rearrange(feat_stage2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_stage3 = rearrange(feat_stage3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        feat_ups = rearrange(feat_ups, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)

        feat_stage1 = self.patch_to_embedding_low1(feat_stage1)
        feat_stage2 = self.patch_to_embedding_low2(feat_stage2)
        feat_stage3 = self.patch_to_embedding_low3(feat_stage3)
        feat_ups = self.patch_to_embedding_high(feat_ups)

        # encoder
        feat_stage1 = self.encoder_stage1(feat_stage1)
        feat_stage2 = self.encoder_stage2(feat_stage2)
        feat_stage3 = self.encoder_stage3(feat_stage3)
        feat_ups = self.encoder_up(feat_ups)

        feat_ups = self.decoder3(feat_ups, feat_stage3)
        feat_ups = self.decoder2(feat_ups, feat_stage2)
        feat_ups = self.decoder1(feat_ups, feat_stage1)

        feat_ups = self.embedding_to_patch(feat_ups)
        feat_ups = rearrange(feat_ups, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=256 // p, p1=p, p2=p)

        feat_ups = self.span_conv1x1(feat_ups)

        x = self.tail(feat_ups)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, m=None, **kwargs):
        return self.fn(x, m, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, m=None, **kwargs):
        x = self.norm(x)
        if m is not None: m = self.norm(m)
        return self.fn(x, m, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class MixedAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, m, mask=None):

        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(m)
        v = self.to_v(m)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual2(PreNorm2(dim, MixedAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, x, m, mask=None):
        for attn1, attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, m, mask=mask)
            x = ff(x)
        return x

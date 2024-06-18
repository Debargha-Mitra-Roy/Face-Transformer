""" Credits - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cross_vit.py """

import torch
from torch import nn, einsum
from torch.nn import Parameter
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math


# ======= SoftMax Loss =======#
class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
        if device_id=None, it will be trained on CPU without model parallel.
    """

    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, label):
        if self.device_id == None:
            out = F.linear(x, self.weight, self.bias)
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat(
                    (out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1
                )
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


# ======= ArcFace Loss =======#
class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
        if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta+m)
    """

    def __init__(
        self,
        in_features,
        out_features,
        device_id,
        s=64.0,
        m=0.50,
        easy_margin=False,
    ):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # ======= cos(theta) & phi(theta) =======#
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat(
                    (
                        cosine,
                        F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(
                            self.device_id[0]
                        ),
                    ),
                    dim=1,
                )
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # ======= Convert label to one-hot =======#
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


# ======= CosFace Loss =======#
class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
        if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # ======= cos(theta) & phi(theta) =======#
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(
                input=self.weight, chunks=len(self.device_id), dim=0
            )
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            abc = F.normalize(temp_x)
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight).t())
            for i in range(0, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat(
                    (
                        cosine,
                        F.linear(F.normalize(temp_x), F.normalize(weight).t()).cuda(
                            self.device_id[0]
                        ),
                    ),
                    dim=1,
                )
        phi = cosine - self.m
        # ======= Convert label to one-hot =======#
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features = "
            + str(self.in_features)
            + ", out_features = "
            + str(self.out_features)
            + ", s = "
            + str(self.s)
            + ", m = "
            + str(self.m)
            + ")"
        )


# ======= SFace Loss =======#
class SFaceLoss(nn.Module):
    r"""Implement of SFace (https://arxiv.org/pdf/2205.12010.pdf):"""

    def __init__(
        self,
        in_features,
        out_features,
        device_id,
        s=64.0,
        k=80.0,
        a=0.80,
        b=1.22,
    ):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.k = k
        self.a = a
        self.b = b
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight, gain=2, mode="out")

    def forward(self, input, label):
        # ======= cos(theta) & phi(theta) =======#
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat(
                    (
                        cosine,
                        F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(
                            self.device_id[0]
                        ),
                    ),
                    dim=1,
                )
        # ======= s*cos(theta) =======#
        output = cosine * self.s

        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        zero_hot = torch.ones(cosine.size())
        if self.device_id != None:
            zero_hot = zero_hot.cuda(self.device_id[0])
        zero_hot.scatter_(1, label.view(-1, 1), 0)

        WyiX = torch.sum(one_hot * output, 1)
        with torch.no_grad():
            # theta_yi = torch.acos(WyiX)
            theta_yi = torch.acos(WyiX / self.s)
            weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
        intra_loss = -weight_yi * WyiX

        Wj = zero_hot * output
        with torch.no_grad():
            # theta_j = torch.acos(Wj)
            theta_j = torch.acos(Wj / self.s)
            weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
        inter_loss = torch.sum(weight_j * Wj, 1)

        loss = intra_loss.mean() + inter_loss.mean()
        Wyi_s = WyiX / self.s
        Wj_s = Wj / self.s
        return (
            output,
            loss,
            intra_loss.mean(),
            inter_loss.mean(),
            Wyi_s.mean(),
            Wj_s.mean(),
        )


# ======= Helpers =======#
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# ======= FeedForward =======#
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ======= Attention =======#
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        x = self.norm(x)
        context = default(context, x)

        if kv_include_self:
            context = torch.cat(
                (x, context), dim=1
            )  # Cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# ======= Transformer encoder, for small and large patches =======#
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ======= Projecting CLS tokens, in the case that small and large patch tokens have different dimensions =======#
class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = (
            nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()
        )

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


# ======= Cross Attention Transformer   =======#
class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ProjectInOut(
                            sm_dim,
                            lg_dim,
                            Attention(
                                lg_dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        ProjectInOut(
                            lg_dim,
                            sm_dim,
                            Attention(
                                sm_dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                    ]
                )
            )

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(
            lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens)
        )

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = (
                sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True)
                + sm_cls
            )
            lg_cls = (
                lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True)
                + lg_cls
            )

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens


# ======= Multi-scale Encoder =======#
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head=64,
        dropout=0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                        Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                        CrossTransformer(
                            sm_dim=sm_dim,
                            lg_dim=lg_dim,
                            depth=cross_attn_depth,
                            heads=cross_attn_heads,
                            dim_head=cross_attn_dim_head,
                            dropout=dropout,
                        ),
                    ]
                )
            )

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens


# ======= Patch-based image to token embedder =======#
class ImageEmbedder(nn.Module):
    def __init__(self, *, dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]

        return self.dropout(x)


# ======= CrossViT class  =======#
class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        loss_type,
        GPU_ID,
        sm_dim,
        lg_dim,
        num_class=10572,
        sm_patch_size=8,
        sm_enc_depth=1,
        sm_enc_heads=8,
        sm_enc_mlp_dim=2048,
        sm_enc_dim_head=64,
        lg_patch_size=14,
        lg_enc_depth=4,
        lg_enc_heads=8,
        lg_enc_mlp_dim=2048,
        lg_enc_dim_head=64,
        cross_attn_depth=2,
        cross_attn_heads=8,
        cross_attn_dim_head=64,
        depth=3,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(
            dim=sm_dim,
            image_size=image_size,
            patch_size=sm_patch_size,
            dropout=emb_dropout,
        )
        self.lg_image_embedder = ImageEmbedder(
            dim=lg_dim,
            image_size=image_size,
            patch_size=lg_patch_size,
            dropout=emb_dropout,
        )

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head,
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head,
            ),
            dropout=dropout,
        )

        self.sm_mlp_head = nn.Sequential(
            nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_class)
        )
        self.lg_mlp_head = nn.Sequential(
            nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_class)
        )

        self.GPU_ID = GPU_ID
        self.loss_type = loss_type

        # Small Dimension
        if self.loss_type == "None":
            print("No loss for CrossViT")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=sm_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=sm_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=sm_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=sm_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

        # Large Dimension
        if self.loss_type == "None":
            print("No loss for CrossViT")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=lg_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=lg_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=lg_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=lg_dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

    def forward(self, img, label=None):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)

        emb = sm_logits + lg_logits

        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb

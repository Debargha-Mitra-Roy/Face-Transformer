import torch
from torch import nn
from torch.nn import (
    Parameter,
    Sequential,
    BatchNorm1d,
    Linear,
)
import torch.nn.functional as F
from torchvision import models

# https://github.com/lukemelas/EfficientNet-PyTorch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import efficientnet

from einops import rearrange, repeat

import math

MIN_NUM_PATCHES = 16


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
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

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


# ======= Residual =======#
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# ======= PreNorm =======#
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ======= Feed Forward =======#
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
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
        self.scale = dim**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        return out


# ======= Transformer =======#
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim, heads=heads, dim_head=dim_head, dropout=dropout
                                ),
                            )
                        ),
                        Residual(
                            PreNorm(
                                dim,
                                FeedForward(dim, mlp_dim, dropout=dropout),
                            )
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# ======= ViT_face =======#
class ViT_face(nn.Module):
    def __init__(
        self,
        *,
        loss_type,
        GPU_ID,
        num_class,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_size // patch_size) ** 2  # NonOverlap-1,2,3

        # num_patches = (2 * (image_size // patch_size) - 1) ** 2  # Overlap-1
        # self.channels = channels  # Overlap-2
        # self.dim = dim  # Overlap-3

        patch_dim = channels * patch_size**2

        # assert (
        #     num_patches > MIN_NUM_PATCHES
        # ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)  # NonOverlap-4
        # self.patch_to_embedding = nn.Conv2d(
        #     in_channels=self.channels,
        #     out_channels=self.dim,
        #     kernel_size=self.patch_size,
        #     stride=self.patch_size // 2,
        # )  # Overlap-4

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
        )
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID
        if self.loss_type == "None":
            print("No loss for ViT_face")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

    def forward(self, img, label=None, mask=None):
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)

        # ======= Overlapping Patch =======#
        # # (Batch, Channel, Height, Width) -> (B, D, 2*(H/P)-1, 2*(W/P)-1)
        # out = self.patch_to_embedding(img)
        # # (B, D, 2*(H/P)-1, 2*(W/P)-1) -> (B, D, Np)
        # out = torch.flatten(out, start_dim=2, end_dim=3)
        # # (B, D, Np) -> (B, Np, D)
        # x = out.transpose(1, 2)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        emb = self.mlp_head(x)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb


# ======= EfficientNet_V2_face =======#
class EfficientNet_V2_face(nn.Module):
    def __init__(
        self,
        GPU_ID,
        pretrained=True,
        fine_tune=True,
        loss_type="CosFace",
        dim=512,
        num_class=10572,
    ):
        super().__init__()
        if pretrained:
            print("[INFO]: Loading pre-trained weights...")
        else:
            print("[INFO]: Not loading pre-trained weights...")

        weights = (
            models.EfficientNet_V2_S_Weights.DEFAULT
        )  # models.EfficientNet_V2_S_Weights.IMAGENET1K_V1

        self.model = models.efficientnet_v2_s(weights=weights)
        self.loss_type = loss_type

        if fine_tune:
            print("[INFO]: Fine-tuning all layers...")
            for params in self.model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print("[INFO]: Freezing hidden layers...")
            for params in self.model.parameters():
                params.requires_grad = False

        self.GPU_ID = GPU_ID

        if self.loss_type == "None":
            print("No loss for EfficientNet_V2_face")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

        self.model.classifier = Sequential(
            BatchNorm1d(1280), Linear(1280, dim), BatchNorm1d(dim)
        )

    def forward(self, img, label=None):
        emb = self.model(img)
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb


# ======= EfficientNet_V2_ViT =======#
class EfficientNet_V2_ViT(nn.Module):
    def __init__(
        self,
        GPU_ID,
        pretrained=True,
        fine_tune=True,
        loss_type="CosFace",
        dim=512,
        num_class=10572,
    ):
        super().__init__()
        if pretrained:
            print("[INFO]: Loading pre-trained weights...")
        else:
            print("[INFO]: Not loading pre-trained weights...")

        weights = (
            models.EfficientNet_V2_S_Weights.DEFAULT
        )  # models.EfficientNet_V2_S_Weights.IMAGENET1K_V1

        self.pt_model = models.efficientnet_v2_s(weights=weights)
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        if self.loss_type == "None":
            print("No loss for EfficientNetV2_ViT")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

        self.trimmed_model = nn.Sequential(*list(self.pt_model.children())[:-2])

        if fine_tune:
            print("[INFO]: Fine-tuning all layers...")
            for params in self.trimmed_model.parameters():
                params.requires_grad = True
        elif not fine_tune:
            print("[INFO]: Freezing hidden layers...")
            for params in self.trimmed_model.parameters():
                params.requires_grad = False

        self.vit = ViT_face(
            loss_type="CosFace",
            GPU_ID=GPU_ID,
            num_class=10572,
            image_size=4,  # Since [112 x 112] spatial size creates [1280 x 4 x 4] shape feature
            patch_size=1,  # Modified from 8 to 1
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            channels=1280,  # As channels=3 dafaults
            dropout=0.1,
            emb_dropout=0.1,
        )

    def forward(self, img, label=None):
        x = self.trimmed_model(img)
        emb = self.vit(x)

        # To get output.data as in x
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb


# ======= EfficientNet_V1_ViT =======#
class EfficientNet_V1_ViT(nn.Module):
    def __init__(
        self,
        GPU_ID,
        pretrained=True,
        fine_tune=True,
        loss_type="CosFace",
        dim=512,
        num_class=10572,
    ):
        super().__init__()

        self.efficient_net = EfficientNet.from_pretrained("efficientnet-b0")
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        if self.loss_type == "None":
            print("No loss for EfficientNetV1_ViT")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

        if fine_tune:
            print("[INFO]: Fine-tuning Only last 3 Blocks from 16 blocks...")
            for i in range(
                0, len(self.efficient_net._blocks)
            ):  # Have 16 blocks in total
                for index, param in enumerate(
                    self.efficient_net._blocks[i].parameters()
                ):
                    if i >= len(self.efficient_net._blocks) - 3:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        elif not fine_tune:
            print("[INFO]: Freezing hidden layers...")
            for param in self.efficient_net.parameters():
                param.requires_grad = False

        self.vit = ViT_face(
            loss_type="CosFace",
            GPU_ID=GPU_ID,
            num_class=10572,
            image_size=28,  # For reduction_2 level for (112, 112) image
            patch_size=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            channels=24,  # Output of reduction level 2 is (B, 24, 28, 28) for (112, 112) image
            dropout=0.1,
            emb_dropout=0.1,
        )

    def forward(self, img, label=None):
        # [1280 x 7 x 7] for [3 x 224 x 224] resolution and [1280 x 4 x 4] for [3 x 112 x 112] image
        # x = self.efficient_net.extract_features(img)
        endpoints = self.efficient_net.extract_endpoints(img)
        x = endpoints[
            "reduction_2"
        ]  # Returns tensor of shape (B, 40, 28, 28) for (224, 224) image and (B, 40, 14, 14) for (112, 112) image in reduction_3
        emb = self.vit(x)

        # To get output.data as in x
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb


# ======= EfficientNet_Trim_ViT =======#
class EfficientNet_Trim_ViT(nn.Module):
    def __init__(
        self,
        GPU_ID,
        pretrained=True,
        fine_tune=False,
        loss_type="CosFace",
        dim=512,
        num_class=10572,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.GPU_ID = GPU_ID

        if self.loss_type == "None":
            print("No loss for EfficientNet_Trim_ViT")
        else:
            if self.loss_type == "Softmax":
                self.loss = Softmax(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "CosFace":
                self.loss = CosFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "ArcFace":
                self.loss = ArcFace(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )
            elif self.loss_type == "SFace":
                self.loss = SFaceLoss(
                    in_features=dim,
                    out_features=num_class,
                    device_id=self.GPU_ID,
                )

        if fine_tune:
            print("[INFO]: This flag has no utilization...")
        elif not fine_tune:
            print("[INFO]: Freezing hidden layers...")

        blocks_args, global_params = efficientnet(
            width_coefficient=1.0,
            depth_coefficient=1.0,
            image_size=112,
            dropout_rate=0.2,
            drop_connect_rate=0.2,
            num_classes=10572,
            include_top=False,
        )

        self.pt_model = EfficientNet(
            blocks_args=blocks_args, global_params=global_params
        )
        self.layers = list(self.pt_model._modules.keys())

        # [
        #     "_conv_stem",
        #     "_bn0",
        #     "_blocks",
        #     "_conv_head",
        #     "_bn1",
        #     "_avg_pooling",
        #     "_dropout",
        #     "_fc",
        #     "_swish",
        # ]  # Convertion from Odict_keys

        self.layer_count = 0
        for l in self.layers:
            if l != "_blocks":
                self.layer_count += 1
            else:
                self.pt_model._blocks = nn.Sequential(
                    *[self.pt_model._blocks[i] for i in range(3)]
                )  # Solve ModuleList Problem
                break

        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pt_model._modules.pop(self.layers[-i])

        self.pt_model_trim = nn.Sequential(self.pt_model._modules)
        self.pt_model = None

        self.vit = ViT_face(
            loss_type="CosFace",
            GPU_ID=GPU_ID,
            num_class=10572,
            image_size=28,  # For reduction_2 level for (112, 112) image
            patch_size=4,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            channels=24,  # Output of reduction level 2 is (B, 24, 28, 28) for (112, 112) image
            dropout=0.1,
            emb_dropout=0.1,
        )

    def forward(self, img, label=None):
        x = self.pt_model_trim(img)
        emb = self.vit(x)

        # To get output.data as in x
        if label is not None:
            x = self.loss(emb, label)
            return x, emb
        else:
            return emb

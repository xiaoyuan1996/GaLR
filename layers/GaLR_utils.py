# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
import math
from layers import seq2vec
from torch.autograd import Variable

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = self.fc(x)
        return self.linear(out)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_k = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_q = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])
        self.linear_merge = nn.Linear(__C['fusion']['mca_HIDDEN_SIZE'], __C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])

    def forward(self, v, k, q, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C['fusion']['mca_MULTI_HEAD'],
            self.__C['fusion']['mca_HIDDEN_SIZE_HEAD']
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C['fusion']['mca_HIDDEN_SIZE']
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C['fusion']['mca_HIDDEN_SIZE'],
            mid_size=__C['fusion']['mca_FF_SIZE'],
            out_size=__C['fusion']['mca_HIDDEN_SIZE'],
            dropout_r=__C['fusion']['mca_DROPOUT_R'],
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, x_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm1 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout2 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm2 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

        self.dropout3 = nn.Dropout(__C['fusion']['mca_DROPOUT_R'])
        self.norm3 = LayerNorm(__C['fusion']['mca_HIDDEN_SIZE'])

    def forward(self, x, y, x_mask=None, y_mask=None):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True):
        super(ExtractFeature, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']

        self.resnet = resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = finetune

        self.pool_2x2 = nn.MaxPool2d(4)

        self.up_sample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_sample_4 = nn.Upsample(scale_factor=4, mode='nearest')

        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # Lower Feature
        f2_up = self.up_sample_2(f2)
        lower_feature = torch.cat([f1, f2_up], dim=1)

        # Higher Feature
        f4_up = self.up_sample_2(f4)
        higher_feature = torch.cat([f3, f4_up], dim=1)
        # higher_feature = self.up_sample_4(higher_feature)

        # batch * 512
        feature = f4.view(f4.shape[0], 512, -1)
        solo_feature = self.linear(torch.mean(feature,dim=-1))

        # torch.Size([10, 192, 64, 64])
        # torch.Size([10, 768, 64, 64])
        # torch.Size([10, 512])
        return lower_feature, higher_feature, solo_feature

class VSA_Module(nn.Module):
    def __init__(self, opt = {}):
        super(VSA_Module, self).__init__()

        # extract value
        channel_size = opt['multiscale']['multiscale_input_channel']
        out_channels = opt['multiscale']['multiscale_output_channel']
        embed_dim = opt['embed']['embed_dim']

        # sub sample
        self.LF_conv = nn.Conv2d(in_channels=192, out_channels=channel_size, kernel_size=3, stride=4)
        self.HF_conv = nn.Conv2d(in_channels=768, out_channels=channel_size, kernel_size=1, stride=1)

        # visual attention
        self.conv1x1_1 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=channel_size*2, out_channels=out_channels, kernel_size=1)

        # solo attention
        self.solo_attention = nn.Linear(in_features=256, out_features=embed_dim)

    def forward(self, lower_feature, higher_feature, solo_feature):

        # b x channel_size x 16 x 16
        lower_feature = self.LF_conv(lower_feature)
        higher_feature = self.HF_conv(higher_feature)

        # concat
        concat_feature = torch.cat([lower_feature, higher_feature], dim=1)

        # residual
        concat_feature = higher_feature.mean(dim=1,keepdim=True).expand_as(concat_feature) + concat_feature

        # attention
        main_feature = self.conv1x1_1(concat_feature)
        attn_feature = torch.sigmoid(self.conv1x1_2(concat_feature).view(concat_feature.shape[0],1,-1)).view(concat_feature.shape[0], 1, main_feature.shape[2], main_feature.shape[3])
        atted_feature = (main_feature*attn_feature).squeeze(dim=1).view(attn_feature.shape[0], -1)

       # solo attention
        solo_att = torch.sigmoid(self.solo_attention(atted_feature))
        solo_feature = solo_feature*solo_att

        return l2norm(solo_feature, -1)

class Skipthoughts_Embedding_Module(nn.Module):
    def __init__(self, vocab, opt, out_dropout=-1):
        super(Skipthoughts_Embedding_Module, self).__init__()
        self.opt = opt
        self.vocab_words = vocab

        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'], self.opt['seq2vec']['dropout'])

        self.to_out = nn.Linear(in_features=2400, out_features=self.opt['embed']['embed_dim'])
        self.dropout = out_dropout

    def forward(self, input_text ):
        x_t_vec = self.seq2vec(input_text)
        out = F.relu(self.to_out(x_t_vec))
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12
# ====================================================================
# About GCN
class GCN(nn.Module):
    def __init__(self , dim_in=20 , dim_out=20, dim_embed = 512):
        super(GCN,self).__init__()

        self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)

        self.out = nn.Linear(dim_out * dim_in, dim_embed)

    def forward(self, A, X):
        batch, objects, rep = X.shape[0], X.shape[1], X.shape[2]

        # first layer
        tmp = (A.bmm(X)).view(-1, rep)
        X = F.relu(self.fc1(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # second layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc2(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # third layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc3(tmp))
        X = X.view(batch, -1)

        return l2norm(self.out(X), -1)


















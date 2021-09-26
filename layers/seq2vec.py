# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------

# A revision version from Skip-thoughs
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import skipthoughts
from skipthoughts import BayesianUniSkip

def factory(vocab_words, opt , dropout=0.25):
    if opt['arch'] == 'skipthoughts':
        st_class = getattr(skipthoughts, opt['type'])
        seq2vec = st_class(opt['dir_st'],
                           vocab_words,
                           dropout=dropout,
                           fixed_emb=opt['fixed_emb'])

    else:
        raise NotImplementedError
    return seq2vec

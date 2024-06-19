'''Classification mlp head module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import einops.layers.torch as einops_torch


class MLPHead(nn.Module):
    '''Final classification MLP layer.
    '''

    def __init__(self, embedding_dim, num_classes, mlp_ratio=4):
        '''Param init.
        '''
        super(MLPHead, self).__init__()


        self.classification_head = nn.Sequential(nn.LayerNorm(embedding_dim),
                                                 nn.Linear(embedding_dim, embedding_dim*mlp_ratio),
                                                 nn.GELU(),
                                                 nn.Linear(embedding_dim*mlp_ratio, num_classes))


    def forward(self, x):
        extracted_cls_token = x[:, 0, :].squeeze(1) #first position in the patch num dimension. That's where the CLS token was initialized. Should be the size of [batch size, patch embedding] since we removed the 1 in the first dimension.

        out = self.classification_head(extracted_cls_token)

        return out




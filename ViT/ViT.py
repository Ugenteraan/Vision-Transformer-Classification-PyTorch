'''Main Vision Transformer module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding



class VisionTransformer(nn.Sequential):
    '''Full ViT architecture.
    '''

    def __init__(self, 
    			image_height, 
    			image_width, 
    			image_depth, 
    			patch_size,
    			embedding_dim, 
    			transformer_network_depth, 
    			num_classes, 
    			device, 
    			**kwargs):
        '''Combines all the modules in sequence.
        '''

        
        super().__init__(
                PatchEmbedding(image_height=image_height, 
                			   image_width=image_width, 
                			   image_depth=image_depth, 
                			   patch_size=patch_size, 
                			   embedding_dim=embedding_dim,
                			   device=device).to(device),

                TransformerEncoderNetwork(transformer_network_depth=transformer_network_depth, 
                						  patch_embedding_dim=patch_embedding_dim, 
                						  device=device, 
                						  **kwargs).to(device),

                MLPHead(patch_embedding_dim=patch_embedding_dim, 
                		num_classes=num_classes).to(device)
                )







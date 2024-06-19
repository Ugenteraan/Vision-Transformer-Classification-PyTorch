'''Patch Embeding layer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
import einops.layers.torch as einops_torch
import einops

from .positional_encoder import PositionalEncoder


class PatchEmbedding(nn.Module):
    '''The idea of patch embedding is to divide a given image into same-size grids and project each grids into another dimension linearly.
       This can be done effectively by using a conv layer with specific configuration. 
       Although, in this implementation, we will be doing it a bit manually using torch.unFold function. 
    '''

    def __init__(self, 
                 patch_size, 
                 image_size, 
                 image_depth, 
                 embedding_dim, 
                 device=None):
        '''Init variables.
        '''

        super(PatchEmbedding, self).__init__()
        
        self.device = device
        self.unfolding_func = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)).to(device)
        self.folding_func = nn.Fold(output_size=(image_size, image_size), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)).to(device)
        self.einops_rearrange = einops_torch.Rearrange('b e p -> b p e').to(device) #this is to change the position of the embedding and the number of patches dimension after Unfold.
        
        self.positional_encoding_module = PositionalEncoder(token_length=cls_token_concat_tensors.size(1), 
                                                            output_dim=cls_token_concat_tensors.size(2), 
                                                            n=1000, 
                                                            device=self.device)

        if not embedding_dim is None:
            self.patch_linear_layer = nn.Linear(in_features=patch_size*patch_size*image_depth, out_features=embedding_dim, bias=True).to(device) #to linearly project the patches. 


    def get_non_overlapping_patches(self, imgs):
        '''imgs: image tensors of [B, C, H, W]
            Perform the unfolding operation and return the patch without the linear projection. This function can be used by the loss calculation module later.
        '''
        
        patched_image_tensors = self.unfolding_func(imgs)
        rearranged_tensors = self.einops_rearrange(patched_image_tensors) 
        # rearranged_tensors = einops.rearrange(patched_image_tensors, 'b e p -> b p e')
        
        return rearranged_tensors.to(self.device)
    
    def make_patches_into_images(self, patches):
        '''Reverses the get_non_overlapping_patches(...) method.
        '''
        
        rearranged_patches = self.einops_rearrange(patches)
        # rearranged_patches = einops.rearrange(patches, 'b p e -> b e p')
        images = self.folding_func(rearranged_patches)
        
        return images.to(self.device)
        
    
    
    def forward(self, x):
        '''Creates linear projection out of the patches from the images.
        '''
        
        patched_image_tensors = self.get_non_overlapping_patches(x)
        linear_projected_patches = self.patch_linear_layer(patched_image_tensors).to(self.device)

        cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim)).to(self.device)
        batched_cls_token = einops.repeat(cls_token, '() n e -> b n e', b=linear_projected_patches.size(0))

        cls_concat_tensor = torch.cat([batched_cls_token, linear_projected_patches], dim=1)


        self.positional_encoding_tensor = positional_encoding_module()


    
    
#used this to verify the correctness of the patching and un-patching operation implemented.
# if __name__ == '__main__':
    
#     x = torch.randn((2, 3, 224, 224))
    
#     p = PatchEmbed(patch_size=16, image_size=224, image_depth=3, embedding_dim=768, device=torch.device('cpu'))
    
#     x_ = p.get_non_overlapping_patches(imgs=x)
#     print(x_.size())
#     y = p.make_patches_into_images(x_)
#     print(y.size())
    
#     print(torch.equal(x, y))




    
import torch 
import torch.nn as nn 
from .networks import (
    AttentionBlock, 
    ResnetBlock, 
    Downsample, 
    Upsample, 
    SinPosEmb, 
    activation_ftn 
)

"""
Note: Mostly based on https://github.com/ermongroup/SDEdit/blob/main/models/diffusion.py
"""



class SBDiffE(nn.Module): 
    def __init__(self, config):
        """
        A score function approximator. 
        We use U-Net architecture with residual blocks and attention blocks. 
        This module takes $X_t$ and $t$(or more precisely, a timestep embedding) as an input, 
        and outputs $\varepsilon_\theta(X_t, t) \approx \nabla_{X_t} \log p_t(X_t)$. 

        Input: config (all required configurations.) 

        - in_channels, out_channels: input image channels, output image channels 
        - num_res_blocks: number of resudial blocks 
        - apply_attention_resolutions: image resolutions that we apply attention blocks. 
            (To note, we do not apply attention blocks to every resolution of the image.) 
        - resolution: Image resolution(input and output) 
        - num_resolutions: Number of image resolutions. 
            (To note: We use power of 2 for the resolutions. 
            For example, if we do the up/down sampling of 
            256 -> 128 -> 64 -> 32 -> 64 -> 128 -> 256, then num_resolutions == 4.) 
        - hidden_channels: number of feature maps for the conv layers. 
        - time_embedding: timestep embedding dimension. We simply use time_embedding = 4 * hidden_channels. 
        - channel_factor: multiplication factor for the channel. 
        
        """
        super(SBDiffE, self).__init__()
        self.config = config    
        self.in_channels, self.out_channels = config.dataset.in_channels, config.dataset.out_channels 
        self.channel_factor = tuple(config.model.channel_factor) 
        self.num_res_blocks = config.model.num_res_blocks 
        self.apply_attention_resolutions = config.model.apply_attention_resolutions 
        self.resolution = config.dataset.image_size 
        self.hidden_channels = config.model.hidden_channels
        self.time_embedding_dim = 4 * self.hidden_channels
        self.num_resolutions = len(config.model.channel_factor)           
        # self.activation = activation_ftn
        # timestep embedding.
        self.sinposemb = SinPosEmb(self.hidden_channels)
        self.timestep_embedding = nn.Module()
        self.timestep_embedding.dense = nn.ModuleList([
            torch.nn.Linear(self.hidden_channels,
                            self.time_embedding_dim),
            torch.nn.Linear(self.time_embedding_dim,
                            self.time_embedding_dim),
        ])

        # first convolutional layer. 
        self.conv_in = nn.Conv2d(
            self.in_channels, self.hidden_channels, 3, stride = 1, padding = 1
        )
        
        # downsampling layers.
        current_resolution = self.resolution
        in_channel_factor = (1,) + self.channel_factor
        self.down = nn.ModuleList()
        block_in = None 
        for i in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList() 
            block_in = self.hidden_channels * in_channel_factor[i]
            block_out = self.hidden_channels * self.channel_factor[i]
            for j in range(self.num_res_blocks):
                block.append(ResnetBlock(
                        in_channels = block_in, 
                        out_channels = block_out, 
                        embedding_dim = self.time_embedding_dim)
                )
                block_in = block_out
                if current_resolution in self.apply_attention_resolutions: 
                    attn.append(AttentionBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i != self.num_resolutions - 1: 
                down.downsample = Downsample(block_in)
                current_resolution = current_resolution // 2
            self.down.append(down)

        # middle layers. (Spatial dimensions are preserved during these modules.) 
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels = block_in,
            out_channels = block_in,
            embedding_dim = self.time_embedding_dim
        )
        self.mid.attn_1 = AttentionBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels = block_in,
            out_channels = block_in,
            embedding_dim = self.time_embedding_dim
        )

        # upsampling layers. 
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.hidden_channels * self.channel_factor[i]
            skip_in = self.hidden_channels * self.channel_factor[i]
            for num_block in range(self.num_res_blocks + 1):
                if num_block == self.num_res_blocks: 
                    skip_in = self.hidden_channels * in_channel_factor[i]
                block.append(ResnetBlock(
                        in_channels = block_in + skip_in, 
                        out_channels = block_out, 
                        embedding_dim = self.time_embedding_dim)
                )
                block_in = block_out 
                if current_resolution in self.apply_attention_resolutions:
                    attn.append(AttentionBlock(block_in))
            up = nn.Module()
            up.block = block 
            up.attn = attn
            if i != 0:
                up.upsample = Upsample(block_in)
                current_resolution = current_resolution * 2
            self.up.insert(0, up) 

        
        # final layers. 
        self.norm_out = nn.GroupNorm(
            num_groups = 32, num_channels = block_in, eps = 1e-6, affine = True
        )
        self.conv_out = nn.Conv2d(
            block_in, self.out_channels, 3, stride = 1, padding = 1
        )



    def forward(self, X_t, t): 
        assert X_t.shape[2] == X_t.shape[3] == self.resolution
        
        # timestep embedding 
        time_embedding = self.sinposemb(t)   # [batch size, embedding dim]
        time_embedding = self.timestep_embedding.dense[0](time_embedding)
        time_embedding = activation_ftn(time_embedding)
        time_embedding = self.timestep_embedding.dense[1](time_embedding)

        # first convolution 
        hs = [self.conv_in(X_t)]
        
        # downsamplings
        for i in range(self.num_resolutions): 
            for j in range(self.num_res_blocks):
                h = self.down[i].block[j](hs[-1], time_embedding)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_resolutions - 1: 
                hs.append(self.down[i].downsample(hs[-1]))
        
        # middle layers 
        h = hs[-1]
        h = self.mid.block_1(h, time_embedding)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, time_embedding)
        
        # upsamplings 
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].block[j](
                    torch.cat([h, hs.pop()], dim = 1), 
                    time_embedding
                )
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0: 
                h = self.up[i].upsample(h)
        # final conv 
        h = activation_ftn(self.norm_out(h))
        h = self.conv_out(h)
        return h        # this defines the epsilon network. 
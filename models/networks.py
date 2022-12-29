import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math



""" 
Helper classes and functionalities. 
"""
class SinPosEmb(nn.Module): 
    """
    A class for Sinusodial Position Embeddings. 
    We embed the positional embeddings using the attention modules. 
    For the technical details, see section 3 of "Attention is all you need" paper: 
    <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>. 
    """
    def __init__(self, embedding_dim): 
        super(SinPosEmb, self).__init__()
        assert embedding_dim % 2 == 0, "Given embedding dimension is not an even number."
        self.embedding_dim = embedding_dim 

    def forward(self, t): 
        device = t.device
        half_dim = self.embedding_dim // 2 
        embeddings = math.log(10000) / (half_dim - 1) 
        embeddings = torch.exp(torch.arange(half_dim, dtype = torch.float32) * (-embeddings))
        embeddings = embeddings.to(device)
        embeddings = t.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim = 1) 
        return embeddings 



def activation_ftn(x): 
    return x * torch.sigmoid(x)



""" 
Building Blocks. 
"""
class Upsample(nn.Module): 
    """
    Upsampling Block for U-Net architecture. 
    """
    def __init__(self, in_channels): 
        super(Upsample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, 3, stride = 1, padding = 1) 
    
    def forward(self, x): 
        x = F.interpolate(
            x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x
    


class Downsample(nn.Module): 
    """
    Downsampling Block for U-Net architecture.
    """
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride = 2, padding = 0) 

    def forward(self, x): 
        pads = (0, 1, 0, 1)
        x = F.pad(x, pads, mode = "constant", value = 0)
        x = self.conv(x)
        return x



class ResnetBlock(nn.Module): 
    """
    A residual block. 
    """
    def __init__(self, in_channels, out_channels, use_dropout = False, embedding_dim = 512) : 
        """
        Initialize the instance. 
        """
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels 
        out_channels = out_channels
        self.out_channels = out_channels
        self.use_dropout = use_dropout 
        self.embedding_dim = embedding_dim      # time embedding dimension. 

        # Layers in chronological orders. 
        self.norm1 = nn.GroupNorm(
            num_groups = 32, num_channels = in_channels, eps = 1e-6, affine = True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride = 1, padding = 1
        )
        self.time_embedding_mlp = nn.Linear(embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(
            num_groups = 32, num_channels=out_channels, eps = 1e-6, affine = True
        )
        # self.dropout = nn.Dropout(0.2) ## This should be modified. 
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride = 1, padding = 1
        )
        if self.in_channels != self.out_channels:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, out_channels, 3, stride = 1, padding = 1
            ) 


    def forward(self, x, timestep_embedding): 
        h = x 
        h = self.norm1(h)
        h = activation_ftn(h) 
        h = self.conv1(h) 
        h = h + self.time_embedding_mlp(activation_ftn(timestep_embedding))[:, :, None, None] 
        h = self.norm2(h) 
        h = activation_ftn(h) 
        # h = self.dropout(h) 
        h = self.conv2(h)
        if self.in_channels != self.out_channels : 
            x = self.conv_shortcut(x) 

        return x + h    # Ensure Residual connection. 
    


class AttentionBlock(nn.Module): 
    """
    Attention Block that is used for the network architecture. 
    We apply three gates: Q, V, K, and concatenate each of the gates. 
    For the technical details, see the paper "Attention is all you need": 
    <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>. 
    """
    def __init__(self, in_channels): 
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels 
        self.norm = nn.GroupNorm(
            num_groups = 32, num_channels = in_channels, eps = 1e-6, affine = True
        )
        self.Q = nn.Conv2d(
            in_channels, in_channels, 1, stride = 1, padding = 0
        )
        self.K = nn.Conv2d(
            in_channels, in_channels, 1, stride = 1, padding = 0
        )
        self.V = nn.Conv2d(
            in_channels, in_channels, 1, stride = 1, padding = 0
        )
        self.projection = nn.Conv2d(
            in_channels, in_channels, 1, stride = 1, padding = 0
        )

    
    def forward(self, x): 
        s = x 
        s = self.norm(s)
        q = self.Q(s) 
        k = self.K(s)
        v = self.V(s) 
        
        # compute ingredients of attn. 
        B, C, W, H = q.shape
        q = q.reshape(B, C, H * W) 
        q = q.permute(0, 2, 1) 
        k = k.reshape(B, C, H * W)

        # compute matrix multiplications. 
        q_mul_k = torch.bmm(q, k) 
        q_mul_k_scaled = q_mul_k * (int(C)** (-0.5))
        q_mul_k_softmaxed = F.softmax(q_mul_k_scaled, dim = 2)

        # attend the values. 
        v = v.reshape(B, C, H * W)
        q_mul_k_softmaxed = q_mul_k_softmaxed.permute(0, 2, 1) 
        h_ = torch.bmm(v, q_mul_k_softmaxed)
        h_ = h_.reshape(B, C, H, W) 
        h_ = self.projection(h_) 
        return x + h_   # Ensure the Residual connection
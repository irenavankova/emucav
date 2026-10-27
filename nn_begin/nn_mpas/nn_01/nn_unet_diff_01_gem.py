#!/usr/bin/env python3
import math
import torch
import torch.nn as nn


def get_same_padding(kernel_size, dilation):
    """Calculates padding to keep output size same as input for stride=1."""
    return (dilation * (kernel_size - 1)) // 2


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for Diffusion Timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConvBlock(nn.Module):
    """ConvBlock with Time Embedding Injection."""
    def __init__(self, in_c, out_c, time_emb_dim, kernel_size=3, dilation=1, use_bn=True, activation='silu'):
        super().__init__()
        padding = get_same_padding(kernel_size, dilation)
        
        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_c, out_c, kernel_size, dilation=dilation, padding=0)
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_c)
        )
        self.bn = nn.BatchNorm2d(out_c) if use_bn else nn.Identity()
        
        if activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv(x)
        # Project time embedding and inject across spatial dimensions: (B, out_c, 1, 1)
        time_tokens = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_tokens
        h = self.bn(h)
        return self.act(h)


class UNetDiffGem(nn.Module):
    """
    Conditional Diffusion U-Net for Image-to-Image / Field-to-Field Translation.
    
    Inputs:
      - y_t: Noisy target state tensor (B, out_channels, H_out, W_out)
      - x_cond: Clean condition input tensor (B, in_channels, H_in, W_in)
      - t: Discrete integer timesteps tensor (B,)
      
    Outputs:
      - Predicted noise tensor (B, out_channels, H_out, W_out)
    """
    def __init__(self, in_channels=3, out_channels=2, hide_channels=16, k=3, d=[1, 1], out_size=(36, 108)):
        super().__init__()
        self.out_size = out_size
        time_emb_dim = hide_channels * 4

        # Timestep Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hide_channels),
            nn.Linear(hide_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Condition Spatial Alignment: (H_in, W_in) -> (H_out, W_out)
        self.cond_up = nn.Upsample(size=self.out_size, mode='bilinear', align_corners=True)

        # Combined input channels = target out_channels + condition in_channels
        total_in_c = out_channels + in_channels

        # ENCODER
        self.enc_conv1 = DiffusionConvBlock(total_in_c, hide_channels, time_emb_dim, k, d[0])
        self.pool1 = nn.MaxPool2d(2) # (36, 108) -> (18, 54)
        
        self.enc_conv2 = DiffusionConvBlock(hide_channels, hide_channels * 2, time_emb_dim, k, d[1])
        self.pool2 = nn.MaxPool2d(2) # (18, 54) -> (9, 27)

        # BOTTLENECK
        self.bottleneck = nn.ModuleList([
            DiffusionConvBlock(hide_channels * 2, hide_channels * 4, time_emb_dim, k, d[1]),
            DiffusionConvBlock(hide_channels * 4, hide_channels * 2, time_emb_dim, k, d[1])
        ])

        # DECODER
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = DiffusionConvBlock(hide_channels * 4, hide_channels, time_emb_dim, k, d[0], use_bn=False)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = DiffusionConvBlock(hide_channels * 2, hide_channels, time_emb_dim, k, d[0], use_bn=False)

        # FINAL CONVOLUTION (predicts noise matching out_channels)
        self.final_conv = nn.Conv2d(hide_channels, out_channels, kernel_size=1)

    def forward(self, y_t, x_cond, t):
        # 1. Timestep embedding
        t_emb = self.time_mlp(t)

        # 2. Align spatial shape of x_cond and concatenate with y_t along channel dim
        if x_cond.shape[-2:] != self.out_size:
            x_cond_aligned = self.cond_up(x_cond)
        else:
            x_cond_aligned = x_cond
        
        x = torch.cat([y_t, x_cond_aligned], dim=1) # Shape: (B, out_c + in_c, 36, 108)

        # 3. Encoder
        enc1 = self.enc_conv1(x, t_emb)          # (B, hide, 36, 108)
        x = self.pool1(enc1)                    # (B, hide, 18, 54)
        
        enc2 = self.enc_conv2(x, t_emb)          # (B, hide*2, 18, 54)
        x = self.pool2(enc2)                    # (B, hide*2, 9, 27)

        # 4. Bottleneck
        for block in self.bottleneck:
            x = block(x, t_emb)                 # (B, hide*2, 9, 27)

        # 5. Decoder
        x = self.up1(x)                         # (B, hide*2, 18, 54)
        x = torch.cat([x, enc2], dim=1)         # (B, hide*4, 18, 54)
        x = self.dec_conv1(x, t_emb)            # (B, hide, 18, 54)

        x = self.up2(x)                         # (B, hide, 36, 108)
        x = torch.cat([x, enc1], dim=1)         # (B, hide*2, 36, 108)
        x = self.dec_conv2(x, t_emb)            # (B, hide, 36, 108)

        # 6. Final Convolution to predict noise
        out = self.final_conv(x)                # (B, out_channels, 36, 108)
        return out


def sample_diffusion(model, x_cond, out_channels, out_size, T, alpha, alpha_bar, beta, num_steps=50):
    """Generates predicted target tensor by sampling backward using DDIM for safe striding."""
    model.eval()
    device = x_cond.device
    N = x_cond.shape[0]

    # Move schedule tensors to the same device as x_cond/model
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)
    beta = beta.to(device)
    
    # Start from pure noise y_T ~ N(0, I)
    y = torch.randn(N, out_channels, *out_size, device=device)
    
    # DDIM step sizes allow us to skip timesteps safely
    step_stride = max(1, T // num_steps)
    timesteps_seq = list(range(0, T, step_stride))[::-1]
    
    with torch.no_grad():
        for i, t_val in enumerate(timesteps_seq):
            t_tensor = torch.full((N,), t_val, device=device, dtype=torch.long)
            pred_noise = model(y, x_cond, t_tensor)
            
            a_bar_t = alpha_bar[t_val]
            
            # Get alpha_bar for the next step (or 1.0 if we reached the end)
            if i + 1 < len(timesteps_seq):
                t_prev = timesteps_seq[i + 1]
                a_bar_t_prev = alpha_bar[t_prev]
            else:
                a_bar_t_prev = torch.tensor(1.0, device=device)
                
            # 1. Predict the clean image (y_0) from the current noisy step
            pred_y0 = (y - torch.sqrt(1.0 - a_bar_t) * pred_noise) / torch.sqrt(a_bar_t)
            
            # 2. Calculate direction pointing to the next step
            dir_yt = torch.sqrt(1.0 - a_bar_t_prev) * pred_noise
            
            # 3. Combine to form the image at the previous timestep
            y = torch.sqrt(a_bar_t_prev) * pred_y0 + dir_yt
            
    return y


if __name__ == "__main__":
    k = 3
    d = [1, 2]
    # Dummy inputs
    dummy_x_cond = torch.randn(2, 3, 36, 36)   # (B, in_c, 36, 36)
    dummy_y_t = torch.randn(2, 2, 36, 108)     # (B, out_c, 36, 108)
    dummy_t = torch.randint(0, 1000, (2,))     # Timesteps

    model = UNetDiffGem(
        in_channels=3, 
        out_channels=2, 
        hide_channels=16, 
        k=k, 
        d=d, 
        out_size=(36, 108)
    )

    pred_noise = model(dummy_y_t, dummy_x_cond, dummy_t)
    print(f"Input Condition shape: {dummy_x_cond.shape}")
    print(f"Noisy Target shape:    {dummy_y_t.shape}")
    print(f"Predicted Noise shape: {pred_noise.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {params:,}")
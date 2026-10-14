#!/usr/bin/env python3
import torch
import torch.nn as nn

def get_same_padding(kernel_size, dilation):
    """Calculates padding to keep output size same as input for stride=1."""
    return (dilation * (kernel_size - 1)) // 2

class ConvBlock(nn.Module):
    """A flexible block: ReflectionPad -> Conv -> BatchNorm -> Activation."""
    def __init__(self, in_c, out_c, kernel_size=3, dilation=1, use_bn=True, activation='silu'):
        super().__init__()
        padding = get_same_padding(kernel_size, dilation)
        
        layers = []
        layers.append(nn.ZeroPad2d(padding))
        layers.append(nn.Conv2d(in_c, out_c, kernel_size, dilation=dilation, padding=0))
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
            
        if activation == 'silu':
            layers.append(nn.SiLU(inplace=True))
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetSimpleGem(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, hide_channels=16, k=3, d=[1, 1], out_size=(36, 108)):
        super(UNetSimpleGem, self).__init__()
        self.out_size = out_size

        # ENCODER (Split to capture skip connections)
        # 36x36 -> 36x36
        self.enc_conv1 = ConvBlock(in_channels, hide_channels, k, d[0])
        self.pool1 = nn.MaxPool2d(2) # -> 18x18
        
        # 18x18 -> 18x18
        self.enc_conv2 = ConvBlock(hide_channels, hide_channels * 2, k, d[1])
        self.pool2 = nn.MaxPool2d(2) # -> 9x9

        # BOTTLENECK
        # 9x9 deeply processed
        self.bottleneck = nn.Sequential(
            ConvBlock(hide_channels * 2, hide_channels * 4, k, d[1]),
            ConvBlock(hide_channels * 4, hide_channels * 2, k, d[1]) 
        )

        # DECODER: Gradual upsampling with Skip Connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input channels = bottleneck output (hide*2) + enc_conv2 output (hide*2) = hide*4
        self.dec_conv1 = ConvBlock(hide_channels * 4, hide_channels, k, d[0], use_bn=False)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input channels = dec_conv1 output (hide) + enc_conv1 output (hide) = hide*2
        self.dec_conv2 = ConvBlock(hide_channels * 2, hide_channels, k, d[0], use_bn=False)

        # FINAL ASYMMETRIC UPSAMPLING
        # 36x36 -> 36x108 (No skip connection here since input doesn't have this shape)
        self.up_final = nn.Upsample(size=self.out_size, mode='bilinear', align_corners=True)
        self.final_conv = ConvBlock(hide_channels, out_channels, k, d[0], use_bn=False, activation='silu')

    def forward(self, x, opt_dec='upsConv2'):
        # 1. Encoder Path
        enc1 = self.enc_conv1(x)         # Shape: (B, hide, 36, 36)
        x = self.pool1(enc1)             # Shape: (B, hide, 18, 18)
        
        enc2 = self.enc_conv2(x)         # Shape: (B, hide*2, 18, 18)
        x = self.pool2(enc2)             # Shape: (B, hide*2, 9, 9)

        # 2. Bottleneck
        x = self.bottleneck(x)           # Shape: (B, hide*2, 9, 9)

        # 3. Decoder Path
        x = self.up1(x)                  # Upsample to 18x18
        x = torch.cat([x, enc2], dim=1)  # Skip Connection 1 (Concatenate channels)
        x = self.dec_conv1(x)            # Reduce channels back to hide_channels

        x = self.up2(x)                  # Upsample to 36x36
        x = torch.cat([x, enc1], dim=1)  # Skip Connection 2 (Concatenate channels)
        x = self.dec_conv2(x)            # Reduce channels to hide_channels

        # 4. Final Asymmetric Reshape
        x = self.up_final(x)             # Upsample to 36x108
        out = self.final_conv(x)         # Map to out_channels

        return out

# Test usage
if __name__ == "__main__":
    k = 3
    d = [1, 2]
    # Input tensor: Batch=2, Channels=1, H=36, W=36
    dummy_input = torch.randn(2, 1, 36, 36)

    # Initialize model
    model = UNetSimpleGem(
        in_channels=dummy_input.shape[1], 
        out_channels=2, 
        hide_channels=32, 
        k=k, 
        d=d,
        out_size=(36, 108)
    )

    output = model(dummy_input, opt_dec='upsConv2')
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {params:,}")
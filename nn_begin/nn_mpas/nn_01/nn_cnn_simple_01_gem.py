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
        #layers.append(nn.ReflectionPad2d(padding))
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


class CnnSimpleGem(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, hide_channels = 16, k=3, d=[1, 1]):
        super(CnnSimpleGem, self).__init__()

        # ENCODER: Using our flexible blocks
        # 36x36 -> 18x18 -> 9x9
        self.encoder = nn.Sequential(
            ConvBlock(in_c=in_channels, out_c=hide_channels, kernel_size=k, dilation=d[0], use_bn=True, activation='silu'),
            nn.MaxPool2d(2),
            ConvBlock(in_c=hide_channels, out_c=hide_channels*2, kernel_size=k, dilation=d[1], use_bn=True, activation='silu'),
            nn.MaxPool2d(2)
        )

        # BOTTLENECK (The "Bridge")
        # We process the 9x9 features deeply here before upsampling
        self.bottleneck = nn.Sequential(
            ConvBlock(hide_channels * 2, hide_channels * 4, k, d[1]),  # Expand
            ConvBlock(hide_channels * 4, hide_channels * 2, k, d[1])  # Squeeze back
        )

        # DECODER 3: Gradual upsampling
        self.decoder_upsConv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(in_c=hide_channels*2, out_c=hide_channels, kernel_size=k, dilation=d[0], use_bn=False, activation='silu'),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(in_c=hide_channels, out_c=hide_channels, kernel_size=k, dilation=d[0], use_bn=False, activation='silu'),

            nn.Upsample(size=(36, 108), mode='bilinear', align_corners=True),
            # Final layer: usually no BN, specific activation
            ConvBlock(in_c=hide_channels, out_c=out_channels, kernel_size=k, dilation=d[0], use_bn=False, activation='silu')
        )

        # DECODER 2: Aggressive upsampling
        self.decoder_upsConv2 = nn.Sequential(
            nn.Upsample(size=(36, 108), mode='bilinear', align_corners=True),
            ConvBlock(in_c=hide_channels*2, out_c=hide_channels, kernel_size=k, dilation=d[0], use_bn=False),
            ConvBlock(in_c=hide_channels, out_c=out_channels, kernel_size=k, dilation=d[0], use_bn=False, activation='sigmoid')
        )

    def forward(self, x, opt_dec='upsConv2'):
        x = self.encoder(x)
        x = self.bottleneck(x)
        if opt_dec == 'upsConv3':
            return self.decoder_upsConv3(x)
        return self.decoder_upsConv2(x)


# Test usage
if __name__ == "__main__":

    k = 3
    d = [1, 2]
    dummy_input = torch.randn(2, 1, 36, 36)

    model = CnnSimpleGem(in_channels=dummy_input.shape[1], out_channels=dummy_input.shape[1] + 1, hide_channels = 32, k=k, d=d)

    output = model(dummy_input, opt_dec='upsConv2')
    print(f"Output shape with k={k}, d={d}: {output.shape}")

    # Check Parameter Count (Optional but helpful)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {params:,}")
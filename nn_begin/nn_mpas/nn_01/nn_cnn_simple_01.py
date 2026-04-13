#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnSimple(nn.Module):
    def __init__(self,in_channels=3, out_channels=2):
        super(CnnSimple, self).__init__()

        # Encoder: Reduces 36x36 -> 18x18 -> 9x9
        self.encoder = nn.Sequential(
            # BLOCK 1
            # Cin x36x36 -> 16x36x36
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            #nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            # 16x36x36 -> 16x18x18
            nn.MaxPool2d(kernel_size=2, stride=2),

            # BLOCK 2
            # 16x18x18 -> 32x18x18
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            #nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            # 32x36x36 -> 32x9x9
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder_upsConv3 = nn.Sequential(
            # BLOCK 1
            # 32x9x9 -> 32x18x18 (Upsample with interpolation)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 32x18x18 -> 16x18x18
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.SiLU(inplace=True),

            # BLOCK 2
            # 16x18x18 -> 16x36x36
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 16x36x36 -> 16x36x36
            nn.Conv2d(16, 16, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.SiLU(inplace=True),

            # BLOCK 3
            # 16x36x36 -> 16x36x108 (Upsample width only)
            nn.Upsample(size=(36, 108), mode='bilinear', align_corners=True),
            # 16x36x108 -> Cout x 36x108
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.SiLU(inplace=True)
            #nn.SiLU(inplace=True) #Alright but zeros out
            #nn.Sigmoid() #Bad
        )

        # Decoder: Upsampling from 9x9 to 108x36 but avoiding checker board effects for the same kernel and stride sizes
        self.decoder_upsConv2 = nn.Sequential(
            # BLOCK 1
            # 32x9x9 -> 32x36x108 (Upsample with interpolation)
            nn.Upsample(size=(36, 108), mode='bilinear', align_corners=True),
            # 32x36x108 -> 16x36x108
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.SiLU(inplace=True),

            # BLOCK 2
            # 16x36x108 -> Cout x 36x108
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1, padding_mode='zeros')
            #nn.SiLU(inplace=True) #!makes things either better or really bad
            #nn.Sigmoid()
        )

    def forward(self, x, opt_dec='upsConv2'):
        x = self.encoder(x)
        if opt_dec == 'upsConv3':
            x = self.decoder_upsConv3(x)
        elif opt_dec == 'upsConv2':
            x = self.decoder_upsConv2(x)

        return x

# Test usage
if __name__ == "__main__":
    # 1. Initialize
    dummy_input = torch.randn(2, 3, 36, 36)
    model = CnnSimple(in_channels=dummy_input.shape[1], out_channels=dummy_input.shape[1]+1)

    # 2. Check Parameter Count (Optional but helpful)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {params:,}")

    # 3. Test with a dummy batch
    # Tip: Use a batch size > 1 to catch issues with BatchNorm or flattening
    try:
        output = model(dummy_input)
        print(f"Input {dummy_input.shape} -> Output {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")


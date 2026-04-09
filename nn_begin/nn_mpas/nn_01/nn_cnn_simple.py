#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnSimple(nn.Module):
    def __init__(self,in_channels=3, out_channels=2):
        super(CnnSimple, self).__init__()

        # Encoder: Reduces 36x36 -> 18x18 -> 9x9
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder: Upsampling from 9x9 to 108x36
        self.decoder = nn.Sequential(
            # First Step: 9x9 -> 36x18
            # Scale H by 4, W by 2
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 2), stride=(4, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Second Step: 36x18 -> 108x36
            # Scale H by 3, W by 2
            nn.ConvTranspose2d(16, out_channels, kernel_size=(3, 2), stride=(3, 2)),
            nn.ReLU(inplace=True)
            # Final activation (e.g., Sigmoid or Tanh) usually goes here
            # depending on your target data range.
        )

        self.decoder_upsample = nn.Sequential(
            nn.Upsample(size=(108, 36), mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder_upsample(x)
        return x

# Test usage
if __name__ == "__main__":
    # 1. Initialize
    model = CnnSimple(in_channels=3, out_channels=4)

    # 2. Check Parameter Count (Optional but helpful)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {params:,}")

    # 3. Test with a dummy batch
    # Tip: Use a batch size > 1 to catch issues with BatchNorm or flattening
    dummy_input = torch.randn(2, 3, 224, 224)
    try:
        output = model(dummy_input)
        print(f"Success! Input {dummy_input.shape} -> Output {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")


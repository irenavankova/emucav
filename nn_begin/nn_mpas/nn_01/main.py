#!/usr/bin/env python3
from pathlib import Path
from nn_cnn_simple import CnnSimple
from prep_data import preprocess_data, plot_channel_interactive
import torch


def main():
    # Prepare data
    print('-----Preprocessing data')

    fnamex = 'x700'
    in_variables = ['S_yz', 'T_yz', 'runoff_yz']  # Replace with your variable names
    out_variables = ['lifw_xy']  # Replace with your variable names
    root_dir = '/Users/irenavankova/Desktop/beb/rx'
    filepaths = sorted(Path(root_dir).rglob(f'data_annual_{fnamex}.nc'))

    data_in = preprocess_data(filepaths, in_variables)
    data_out = preprocess_data(filepaths, out_variables)

    data_tensor_in = torch.tensor(data_in, dtype=torch.float32)  # Shape: (N, C, H, W)
    data_tensor_out = torch.tensor(data_out, dtype=torch.float32)  # Shape: (N, C, H, W)

    in_channels = data_tensor_in.shape[1]
    out_channels = data_tensor_out.shape[1]

    print(f'Number of input channels = {in_channels}')
    print(f'Number of output channels = {out_channels}')

    # Train network
    print('-----Defining model')

    # Model, loss, optimizer
    model = CnnSimple(in_channels=in_channels, out_channels=out_channels)
    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Check if it loads correctly
    print(model)

    # Train network
    print('-----Training network')



if __name__ == "__main__":
    main()

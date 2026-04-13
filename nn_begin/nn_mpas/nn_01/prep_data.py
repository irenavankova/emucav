#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader

def load_nc_file(filepath, variables):
    """
    Load a .nc file and extract specified variables.

    Args:
        filepath: Path to the .nc file.
        variables: List of variable names to extract.

    Returns:
        A numpy array of shape (C, H, W).
    """
    ds = xr.open_dataset(filepath)
    data_list = []
    for var in variables:
        data = ds[var].values  # Shape: (H, W)
        data_list.append(data)
    return np.stack(data_list, axis=0)  # Shape: (C, H, W)

def preprocess_data(filepaths, variables, donanmask):
    """
    Preprocess all .nc files into a single tensor and normalize per-channel.
    """
    data_list = []
    for fp in filepaths:
        # Assuming load_nc_file returns (C, H, W)
        data_nc = load_nc_file(fp, variables)
        data_list.append(data_nc)

    data = np.stack(data_list)  # Result: (N, C, H, W)

    # 1. Identify channels containing NaNs
    # We check across N, H, and W (axes 0, 2, 3)
    nan_per_channel = np.isnan(data).any(axis=(0, 2, 3))
    channels_with_nans = np.where(nan_per_channel)[0]
    print(f"Channels with NaNs: {channels_with_nans.tolist()}")

    # 2. Generate the mask if requested
    if donanmask:
        # 0. Generate the NaN Mask (N, 1, H, W)
        # Since NaNs are at the same location for all channels,
        # we just check the first channel (index 0) along the C axis.
        mask = np.isnan(data[:, :1, :, :]).astype(np.float32)

    # 1. Compute min/max per channel (axis 0, 2, 3)
    data_min = np.nanmin(data, axis=(0, 2, 3), keepdims=True)
    data_max = np.nanmax(data, axis=(0, 2, 3), keepdims=True)

    # 2. Calculate the range
    data_range = data_max - data_min

    # 3. Handle division by zero:
    data_range[data_range == 0] = 1.0

    # 4. Apply Min-Max scaling. This remaps data to 0-1 range
    data = (data - data_min) / data_range

    # 4. Final Cleanup: Replace NaNs with 0 in the data channels
    data = np.nan_to_num(data, nan=0.0)

    # 5. Concatenate the mask as the last channel
    # Result: (N, C+1, H, W)
    if donanmask:
        data = np.concatenate([data, mask], axis=1)

    return data


def plot_channel_interactive(data_tensor, channel=0):
    channel_data = data_tensor[:, channel, :, :].numpy()
    vmin, vmax = np.nanmin(channel_data), np.nanmax(channel_data)
    print(vmin)
    print(vmax)

    n_samples = data_tensor.shape[0]

    # Turn on interactive mode
    plt.ion()

    for i in range(n_samples):
        snapshot = channel_data[i, :, :]
        print(i)
        print(np.nanmin(snapshot))
        print(np.nanmax(snapshot))

        plt.figure(figsize=(8, 6))
        plt.imshow(snapshot, cmap='magma', vmin=vmin, vmax=vmax)
        plt.colorbar(label=f'Channel {channel} Value')
        plt.title(f"Snapshot {i + 1}, Channel {channel}")

        # This is the magic line that forces the window to draw
        plt.draw()
        plt.pause(0.1)

        input(f"Showing snapshot {i + 1}/{n_samples}. Press Enter for next...")
        plt.close()

    plt.ioff()  # Turn off interactive mode when done

# Test usage
if __name__ == "__main__":
    print('Starting')

    fnamex = 'x700'
    input_channels = ['S_yz', 'T_yz', 'runoff_yz']  # Replace with your variable names
    output_channels = ['lifw_xy','Vbot_xy','Ubot_xy','Tbot_xy','Sbot_xy','Sbl_xy','Tbl_xy','ustar_xy']  # Replace with your variable names

    #output_channels = ['lifw_xy']  # Replace with your variable names


    root_dir = '/Users/irenavankova/Desktop/beb/rx'

    filepaths = sorted(Path(root_dir).rglob(f'data_annual_{fnamex}.nc'))

    print(filepaths)

    data_in = preprocess_data(filepaths, input_channels, donanmask = False)
    data_tensor_in = torch.tensor(data_in, dtype=torch.float32)  # Shape: (N, C, H, W)
    in_channels = data_tensor_in.shape[1]

    data_out = preprocess_data(filepaths, output_channels, donanmask = True)
    data_tensor_out = torch.tensor(data_out, dtype=torch.float32)  # Shape: (N, C, H, W)
    out_channels = data_tensor_out.shape[1]

    print(f'Number of input channels = {in_channels}')
    print(data_tensor_in.shape)
    print(f'Number of output channels = {out_channels}')
    print(data_tensor_out.shape)

    #plot_channel_interactive(data_tensor_in, channel=2)  # Plot the second channel (temperature)
    plot_channel_interactive(data_tensor_out, channel=0)  # Plot the second channel (temperature)


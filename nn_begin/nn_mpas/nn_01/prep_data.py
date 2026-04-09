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

def preprocess_data(filepaths, variables):
    """
    Preprocess all .nc files into a single tensor and normalize per-channel.
    """
    data_list = []
    for fp in filepaths:
        # Assuming load_nc_file returns (C, H, W)
        data_nc = load_nc_file(fp, variables)
        data_list.append(data_nc)

    data = np.stack(data_list)  # Result: (N, C, H, W)

    # 1. Compute min/max per channel (axis 0, 2, 3)
    data_min = np.nanmin(data, axis=(0, 2, 3), keepdims=True)
    data_max = np.nanmax(data, axis=(0, 2, 3), keepdims=True)

    # 2. Calculate the range
    data_range = data_max - data_min

    # 3. Handle division by zero:
    # If range is 0 (constant data), we prevent NaNs by replacing 0 with 1.
    # This keeps the constant data as is (or 0) after subtraction.
    data_range[data_range == 0] = 1.0

    # 4. Apply Min-Max scaling
    data = (data - data_min) / data_range

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


print('Starting')

fnamex = 'x700'
input_channels = ['S_yz', 'T_yz', 'runoff_yz']  # Replace with your variable names
output_channels = ['lifw_xy']  # Replace with your variable names

root_dir = '/Users/irenavankova/Desktop/beb/rx'

filepaths = sorted(Path(root_dir).rglob(f'data_annual_{fnamex}.nc'))

print(filepaths)

data_in = preprocess_data(filepaths, input_channels)
data_out = preprocess_data(filepaths, output_channels)

print('preprocessed')

data_tensor_in = torch.tensor(data_in, dtype=torch.float32)  # Shape: (N, C, H, W)
plot_channel_interactive(data_tensor_in, channel=2)  # Plot the second channel (temperature)

#data_tensor_out = torch.tensor(data_out, dtype=torch.float32)  # Shape: (N, C, H, W)
#plot_channel_interactive(data_tensor_out, channel=0)  # Plot the second channel (temperature)

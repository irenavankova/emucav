#!/usr/bin/env python3
import os
from pathlib import Path
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np


def plot_nc_files(root_dir, varxy, varyz):
    # Find all data_annual.nc files in subdirectories
    nc_files = sorted(Path(root_dir).rglob('data_annual.nc'))

    if not nc_files:
        print("No 'data_annual.nc' files found.")
        return

    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cbar1 = None
    cbar2 = None

    for i, file_path in enumerate(nc_files):
        print(f"[{i + 1}/{len(nc_files)}] Plotting: {file_path}")

        try:
            # Open the dataset
            ds = nc.Dataset(file_path)

            # Extract variables
            tbot_xy = ds.variables[varxy][:]
            t_yz = ds.variables[varyz][:]

            # Clear previous plots
            ax1.clear()
            ax2.clear()

            # 2. Remove old colorbars if they exist
            #if cbar1:
            #    cbar1.remove()
            #if cbar2:
            #    cbar2.remove()

            # Plot Tbot_xy (y, x)
            im1 = ax1.imshow(tbot_xy, origin='lower', aspect='auto', cmap='magma')
            ax1.set_title(f"{varxy}\n{file_path.parent.name}")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")

            # Plot T_yz (z, y)
            im2 = ax2.imshow(t_yz, origin='lower', aspect='auto', cmap='magma')
            ax2.set_title(f"{varyz}\n{file_path.parent.name}")
            ax2.set_xlabel("y")
            ax2.set_ylabel("z")

            print(np.nanmin(t_yz))
            print(np.nanmax(t_yz))

            # Create new colorbars and store them
            #cbar1 = fig.colorbar(im1, ax=ax1, label=varxy)
            #cbar2 = fig.colorbar(im2, ax=ax2, label=varyz)

            plt.draw()
            plt.pause(0.1)
            ds.close()

            input("Press [Enter] for next file (or Ctrl+C to stop)... ")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.ioff()
    print("Finished all files.")
    plt.show()


root_dir = '/Users/irenavankova/Desktop/Ocean2/rx/'
plot_nc_files(root_dir, 'lifw_xy', 'runoff_yz')
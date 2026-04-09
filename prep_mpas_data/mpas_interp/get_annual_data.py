#!/usr/bin/env python3
import os
import xarray as xr

# Path to the main folder containing subfolders
main_folder = '/Users/irenavankova/Desktop/beb/rx'
fnamex = 'x700'

# Loop over each subfolder
for folder_name in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder_name)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        xy_file = os.path.join(folder_path, f'output_data_xy_{fnamex}.nc')
        yz_file = os.path.join(folder_path, f'input_data_yz_{fnamex}.nc')

        # Check if both files exist
        if os.path.exists(xy_file) and os.path.exists(yz_file):
            # Load the NetCDF files
            xy_ds = xr.open_dataset(xy_file)
            yz_ds = xr.open_dataset(yz_file)

            # Function to average over the last 6 time steps
            def average_last_six_months(ds):
                return ds.isel(time=slice(-6, None)).mean(dim='time')

            # Apply the function to both datasets
            avg_xy = average_last_six_months(xy_ds)
            avg_yz = average_last_six_months(yz_ds)

            # Merge the averaged datasets
            merged_ds = xr.merge([avg_xy, avg_yz])

            # Save the merged dataset to a new NetCDF file in the same folder
            output_file = os.path.join(folder_path, f'data_annual_{fnamex}.nc')
            merged_ds.to_netcdf(output_file)

            # Close the datasets
            xy_ds.close()
            yz_ds.close()

            print(f"Processed and saved: {folder_name}")
        else:
            print(f"Skipping {folder_name}: Files not found.")
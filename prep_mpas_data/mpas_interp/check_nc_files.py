#!/usr/bin/env python3
import xarray
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
#dir_nc_save = '/Users/irenavankova/Desktop/Ocean1'  # Update this
dir_nc_save = '/Users/irenavankova/Desktop/beb/rx/rd_142B'
m2km = 1000.0  # Conversion factor
# Adjust these limits based on your specific domain

xmin, xmax = 459000, 639000
ymin, ymax = 5000, 75000
dx, dy = 2000, 2000
zmin, zmax = -700, 0
dz = 20
fnamex = 'x700'

# Load the combined datasets
ds_xy = xarray.open_dataset(f'{dir_nc_save}/output_data_xy_{fnamex}.nc')  # Update filename
ds_yz = xarray.open_dataset(f'{dir_nc_save}/input_data_yz_{fnamex}.nc')  # Update filename

# Select a specific time slice to verify (e.g., the first month)
t_idx = 0
time_label = ds_xy.time.values[t_idx]
print(f"Verifying data for time slice: {time_label}")

# -----------------------------------------------------------------------------
# 1. Visualization of XY fields (Using 'Ubot_xy' as in original)
# -----------------------------------------------------------------------------
var_xy = 'ustar_xy'
data_slice_xy = ds_xy[var_xy].isel(time=t_idx)
print(np.nanmin(data_slice_xy))
print(np.nanmax(data_slice_xy))

plt.figure(figsize=(8, 6))
plt.pcolormesh(ds_xy.x / m2km, ds_xy.y / m2km, data_slice_xy, cmap='hot_r', shading='auto')
plt.colorbar(label=var_xy)
plt.title(f'Interpolated XY Data: {var_xy}\nTime: {time_label}')
plt.ylabel('y (km)')
plt.xlabel('x (km)')
plt.xlim(np.array([xmin, xmax]) / m2km)
plt.ylim(np.array([ymin, ymax]) / m2km)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 2. Visualization of YZ fields (Using 'T_yz')
# -----------------------------------------------------------------------------
var_yz = 'T_yz'
data_slice_yz = ds_yz[var_yz].isel(time=t_idx)
print(np.nanmin(data_slice_yz))
print(np.nanmax(data_slice_yz))

plt.figure(figsize=(8, 6))
plt.pcolormesh(ds_yz.y / m2km, ds_yz.z, data_slice_yz, cmap='hot_r', shading='nearest')
plt.colorbar(label=var_yz)
plt.title(f'Interpolated YZ Data: {var_yz}\nTime: {time_label}')
plt.ylabel('z (m)')
plt.xlabel('y (km)')
plt.xlim(np.array([ymin, ymax]) / m2km)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 3. Visualization of YZ Runoff Projection
# -----------------------------------------------------------------------------
if 'runoff_yz' in ds_yz:
    runoff_slice = ds_yz['runoff_yz'].isel(time=t_idx)
    print(np.nanmin(runoff_slice))
    print(np.nanmax(runoff_slice))

    plt.figure(figsize=(8, 6))
    # Note: Using the scaling logic from your original code (data / dx * dz)
    plt.pcolormesh(ds_yz.y / m2km, ds_yz.z, runoff_slice / dx * dz, cmap='hot_r', shading='nearest')
    plt.colorbar(label='Runoff')
    plt.title(f'Projected Runoff YZ\nTime: {time_label}')
    plt.ylabel('z (m)')
    plt.xlabel('y (km)')
    plt.xlim(np.array([ymin, ymax]) / m2km)
    plt.ylim(np.array([zmin, zmax]))
    plt.tight_layout()
    plt.show()
else:
    print("Variable 'runoff_yz' not found in YZ dataset.")

# Close datasets
ds_xy.close()
ds_yz.close()
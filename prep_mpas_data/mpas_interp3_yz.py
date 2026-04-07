#!/usr/bin/env python3
import numpy as np
import xarray
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d


def interpolate_yz(T, H, yCell, ssh, mask, y_target, z_target):
    """
    Interpolates unstructured data to a regular 2D vertical cross-section.

    Parameters:
    -----------
    T : ndarray
        The variable to interpolate (e.g., Temperature), shape (nCells, nVertLevels)
    H : ndarray
        Layer thicknesses, shape (nCells, nVertLevels)
    yCell : ndarray
        Horizontal coordinates of the unstructured cells
    ssh : ndarray
        Sea surface height at the cells
    mask : ndarray
        Boolean mask to filter relevant cells
    y_target : ndarray
        The regular horizontal coordinates for the transect
    z_target : ndarray
        The regular vertical depth coordinates for the final grid

    Returns:
    --------
    final_data_interp : ndarray
        The data interpolated onto the (z_target, y_target) grid
    data_interp : ndarray
        The data after the first interpolation onto the levels along a y segment
    """

    n_vert = T.shape[1]
    n_y = len(y_target)

    # Initialize intermediate arrays
    data_interp = np.zeros((n_vert, n_y))
    data_interp_H = np.zeros((n_vert, n_y))
    ssh_interp_H = np.zeros(n_y)

    # --- Step 1: Horizontal Interpolation ---
    # Interpolate each vertical level onto the horizontal 'y' line
    for k in range(n_vert):
        level_data = T[mask, k]
        level_H = H[mask, k]
        level_y = yCell[mask]

        data_interp[k, :] = griddata(level_y, level_data, y_target, method='linear')
        data_interp_H[k, :] = griddata(level_y, level_H, y_target, method='linear')

        # Grab SSH from the top level (assuming k=0 or k=1 is surface) to anchor z-coordinates
        if k == 0:
            level_ssh = ssh[mask]
            ssh_interp_H = griddata(level_y, level_ssh, y_target, method='linear')

    # Calculate actual Z depths by subtracting cumulative thickness from SSH
    # Note: Ensure axis=0 matches your vertical dimension
    data_interp_z = ssh_interp_H - np.nancumsum(data_interp_H, axis=0)

    # --- Step 2: Vertical Interpolation ---
    # Create the final grid (Depth x Horizontal)
    final_data_interp = np.full((len(z_target), n_y), np.nan)

    for i in range(n_y):
        z_column = data_interp_z[:, i]
        val_column = data_interp[:, i]

        # Masking NaNs is crucial for interp1d
        mask_valid = ~np.isnan(z_column) & ~np.isnan(val_column)

        if np.any(mask_valid):
            # Sort check: interp1d requires strictly increasing x-values
            # If z_column is decreasing (depth), we flip it
            idx_sort = np.argsort(z_column[mask_valid])

            f_interp = interp1d(
                z_column[mask_valid][idx_sort],
                val_column[mask_valid][idx_sort],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            final_data_interp[:, i] = f_interp(z_target)

    return final_data_interp, data_interp

fdir = '/Users/irenavankova/Library/CloudStorage/GoogleDrive-irena.vanek@gmail.com/My Drive/Research/LANL/SGR/idealized/sg_pull_w_fraz_yesC/rd/rd_112B'
#fdir = '/Users/irenavankova/Desktop/test_old'

ds = xarray.open_dataset(f'{fdir}/timeSeriesStatsMonthly.0002-12-01.nc')
ds.load()
#timeMonthly_avg_velocityX
#timeMonthly_avg_activeTracers_temperature
T = np.squeeze(ds.timeMonthly_avg_activeTracers_temperature.data)
#S = np.squeeze(ds.timeMonthly_avg_activeTracers_salinity.data)
H = np.squeeze(ds.timeMonthly_avg_layerThickness.data)
ssh = np.squeeze(ds.timeMonthly_avg_ssh)

dsMesh = xarray.open_dataset(f'{fdir}/init.nc')
dsMesh.load()
xCell = np.squeeze(dsMesh.xCell.data)
yCell = np.squeeze(dsMesh.yCell.data)

target_x = 700*1000 # Should be outside the cavity for older mpas outputs
#target_x = xCell.max()-2000
ymin = 0*1000
ymax = 80*1000
zmin = -700
zmax = 0
dz = 20
resfac = 5*4

y = np.linspace(ymin, ymax, 4*resfac)
z = np.arange(zmin, zmax, dz)
y_grid, z_grid = np.meshgrid(y,z)

# Only use points within a small 'epsilon' of target_x
buffer = 4000
mask = (xCell > target_x - buffer) & (xCell < target_x + buffer)
print(yCell[mask].shape)

final_data_interp, data_interp = interpolate_yz(T, H, yCell, ssh, mask, y, z)

# =============================================================================
# Optional: Visualize the original and interpolated data
# =============================================================================
# Plot the original 1D data (if coordinates are 1D)

#prop_1D = sgr;
prop_grid = data_interp
#prop_1D = FloatingMask; prop_grid = FM_grid

#v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
#v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

plt.figure(figsize=(12, 5))

# Plot the interpolated 2D data
plt.subplot(1, 2, 1)
nv = ds.dims['nVertLevels']
l = np.arange(0, nv, 1)
plt.pcolormesh(y, -l, prop_grid, cmap='hot_r', shading='nearest')
plt.colorbar(label='Value')
plt.title('Part 1')
plt.xlim([ymin, ymax])

plt.subplot(1, 2, 2)
plt.pcolormesh(y_grid, z_grid, final_data_interp, cmap='hot_r', shading='nearest')
plt.colorbar(label='Value')
plt.title('Part 2')
plt.xlim([ymin, ymax])


plt.tight_layout()

#plt.savefig('interpolation_comparison.png')
plt.show()
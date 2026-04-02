#!/usr/bin/env python3
import numpy as np
import xarray
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d

def apply_masks_xy(xCell, yCell, variable):
    """
    Apply specific masks to the variable based on xCell and yCell coordinates.

    Parameters:
    - xCell, yCell: Coordinates
    - variable: The variable to mask

    Returns:
    - Masked variable
    """
    mask = ((yCell > 76000) & (yCell < 78000) & (xCell > 510000) & (xCell < 520000)) | (yCell > 77000) | (
                yCell < 4000) | ((yCell < 9000) & (xCell < 520000))
    variable[mask] = np.nan

    return variable

def interpolate_xy(xCell, yCell, variable, x_grid, y_grid, mask=None):
    """
    Interpolate a variable onto a regular grid.

    Parameters:
    - xCell, yCell: Original coordinates
    - variable: The variable to interpolate
    - x_grid, y_grid: Target grid coordinates
    - mask: Optional mask to apply before interpolation

    Returns:
    - Interpolated variable on the target grid
    """
    if mask is not None:
        variable = variable * mask

    pts_Cell = np.column_stack((xCell, yCell))
    pts_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    var_grid = griddata(
        pts_Cell,          # Source coordinates (N, 2)
        variable,         # Source values (N,)
        pts_grid,         # Target coordinates (M, 2)
        method='linear',  # Use 'linear' for bilinear interpolation
        fill_value=np.nan
    )

    return var_grid.reshape(y_grid.shape)

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

#--------------------------------------------------------------------------------------------------------
# Load data
fdir = '/Users/irenavankova/Library/CloudStorage/GoogleDrive-irena.vanek@gmail.com/My Drive/Research/LANL/SGR/idealized/sg_pull_w_fraz_yesC/rd/rd_112B'
ds = xarray.open_dataset(f'{fdir}/timeSeriesStatsMonthly.0002-12-01.nc')
ds.load()
dsMesh = xarray.open_dataset(f'{fdir}/init.nc')
dsMesh.load()

var_map_xy_2D = {
    'lifw': ds.timeMonthly_avg_landIceFreshwaterFluxTotal,
    'ustar': ds.timeMonthly_avg_landIceFrictionVelocity,
    'Tbl': ds.timeMonthly_avg_landIceBoundaryLayerTracers_landIceBoundaryLayerTemperature,
    'Sbl': ds.timeMonthly_avg_landIceBoundaryLayerTracers_landIceBoundaryLayerSalinity
}

var_map_xy_3D = {
    'Sbot': ds.timeMonthly_avg_activeTracers_salinity,
    'Tbot': ds.timeMonthly_avg_activeTracers_temperature,
    'Ubot': ds.timeMonthly_avg_velocityX,
    'Vbot': ds.timeMonthly_avg_velocityY
}

var_map_yz_3D = {
    'S': ds.timeMonthly_avg_activeTracers_salinity,
    'T': ds.timeMonthly_avg_activeTracers_temperature
}

ssh = np.squeeze(ds.timeMonthly_avg_ssh)
H = np.squeeze(ds.timeMonthly_avg_layerThickness.data)


# Define grid for interpolation
xmin, xmax = 459000, 639000
ymin, ymax = 5000, 75000
dx, dy = 2000, 2000
zmin, zmax = -700, 0
dz = 20

buffer = 4000
target_x = 600*1000
#target_x = xCell.max()-2000

#--------------------------------------------------------------------------------------------------------
x = np.arange(xmin, xmax + dx, dx)
y = np.arange(ymin, ymax + dy, dy)
x_grid, y_grid = np.meshgrid(x, y)
z = np.arange(zmin, zmax, dz)

# Dictionaries to store the processed results
data_processed_xy = {}
data_processed_yz = {}
data_XY = {}
data_YZ = {}
data_YZi1 = {}

# Grid
areaCell = np.squeeze(dsMesh.areaCell.data)
FloatingMask = np.squeeze(dsMesh.landIceFloatingMask.data)
landIceMask = np.squeeze(dsMesh.landIceMask.data)
xCell = np.squeeze(dsMesh.xCell.data)
yCell = np.squeeze(dsMesh.yCell.data)
max_level = dsMesh['maxLevelCell'] - 1

#@@@@@@XY
# Apply masks to XY data
for name, data_obj in var_map_xy_2D.items():
    # Squeeze and initial data extraction
    data_processed_xy[name] = np.squeeze(data_obj.data)
    data_processed_xy[name] = apply_masks_xy(xCell, yCell, data_processed_xy[name])

for name, data_obj in var_map_xy_3D.items():
    # Extract bottom, squeeze and initial data extraction
    data_bot = data_obj.isel(nVertLevels=max_level).data
    # Process and store
    data_processed_xy[name] = np.squeeze(data_bot)
    data_processed_xy[name] = apply_masks_xy(xCell, yCell, data_processed_xy[name])

# Adjust masks
FloatingMask = FloatingMask.astype(float)
FloatingMask[FloatingMask < 1] = np.nan
landIceMask = landIceMask.astype(float)
landIceMask[landIceMask < 1] = np.nan

# Interpolate all xy variables
for name, data_array in data_processed_xy.items():
    data_XY[f"{name}_grid"] = interpolate_xy(
        xCell, yCell, data_array, x_grid, y_grid, landIceMask * FloatingMask
    )

#@@@@@@YZ

# Only use points within a small 'epsilon' of target_x
mask_near = (xCell > target_x - buffer) & (xCell < target_x + buffer)

for name, data_obj in var_map_yz_3D.items():
    # Squeeze and initial data extraction
    data_processed_yz[name] = np.squeeze(data_obj.data)

# Interpolate all yz variables
for name, data_array in data_processed_yz.items():
    data_YZ[f"{name}_grid"], data_YZi1[f"{name}_grid1"] = interpolate_yz(data_array, H, yCell, ssh, mask_near, y, z)

#--------------------------------------------------------------------------------------------------------

# Visualization of XY fields
prop_1D = data_processed_xy['Ubot']; prop_grid = data_XY['Ubot_grid']
v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(xCell, yCell, c=prop_1D, vmin=v_min, vmax=v_max, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('Original 1D Data')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

plt.subplot(1, 2, 2)
plt.pcolormesh(x_grid, y_grid, prop_grid, vmin=v_min, vmax=v_max, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('Interpolated 2D Data')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

plt.tight_layout()
plt.show()

# Visualization of YZ fields
prop1 = data_YZi1['T_grid1']; prop2 = data_YZ['T_grid']
v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

plt.figure(figsize=(12, 5))
# Plot the interpolated 2D data
plt.subplot(1, 2, 1)
nv = ds.dims['nVertLevels']
l = np.arange(0, nv, 1)
plt.pcolormesh(y, -l, prop1, cmap='hot_r', shading='nearest')
plt.colorbar(label='Value')
plt.title('Part 1')
plt.xlim([ymin, ymax])

plt.subplot(1, 2, 2)
plt.pcolormesh(y, z, prop2, cmap='hot_r', shading='nearest')
plt.colorbar(label='Value')
plt.title('Part 2')
plt.xlim([ymin, ymax])
plt.tight_layout()

plt.show()

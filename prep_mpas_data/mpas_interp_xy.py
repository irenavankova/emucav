#!/usr/bin/env python3
import numpy as np
import xarray
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

#--------------------------------------------------------------------------------------------------------
# Load data
fdir = '/Users/irenavankova/Library/CloudStorage/GoogleDrive-irena.vanek@gmail.com/My Drive/Research/LANL/SGR/idealized/sg_pull_w_fraz_yesC/rd/rd_112B'
#fdir = '/Users/irenavankova/Desktop/test_old'

ds = xarray.open_dataset(f'{fdir}/timeSeriesStatsMonthly.0002-12-01.nc')
ds.load()
dsMesh = xarray.open_dataset(f'{fdir}/init.nc')
dsMesh.load()

var_map_xy_2D = {
    'lifw': ds.timeMonthly_avg_landIceFreshwaterFlux,
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

# Define grid for interpolation
xmin, xmax = 459000, 639000
ymin, ymax = 5000, 75000
dx, dy = 2000, 2000

#--------------------------------------------------------------------------------------------------------
x = np.arange(xmin, xmax + dx, dx)
y = np.arange(ymin, ymax + dy, dy)
x_grid, y_grid = np.meshgrid(x, y)

# Dictionaries to store the processed results
data_processed_xy = {}
data_XY = {}

# Grid
areaCell = np.squeeze(dsMesh.areaCell.data)
#FloatingMask = np.squeeze(dsMesh.landIceFloatingMask.data)
landIceMask = np.squeeze(dsMesh.landIceMask.data)
xCell = np.squeeze(dsMesh.xCell.data)
yCell = np.squeeze(dsMesh.yCell.data)
max_level = dsMesh['maxLevelCell'] - 1

# Apply masks to data
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
#FloatingMask = FloatingMask.astype(float)
#FloatingMask[FloatingMask < 1] = np.nan
landIceMask = landIceMask.astype(float)
#landIceMask[landIceMask < 1] = np.nan

# Interpolate all xy variables
for name, data_array in data_processed_xy.items():
    data_XY[f"{name}_grid"] = interpolate_xy(
        xCell, yCell, data_array, x_grid, y_grid, mask=None #landIceMask #* FloatingMask
    )

#--------------------------------------------------------------------------------------------------------

# Visualization of XY fields
prop_1D = data_processed_xy['Tbot']; prop_grid = data_XY['Tbot_grid']
v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.scatter(xCell, yCell, c=prop_1D, vmin=v_min, vmax=v_max, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('Original 1D Data')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

plt.subplot(2, 1, 2)
plt.pcolormesh(x_grid, y_grid, prop_grid, vmin=v_min, vmax=v_max, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('Interpolated 2D Data')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

plt.tight_layout()
plt.show()
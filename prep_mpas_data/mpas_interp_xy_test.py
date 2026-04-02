#!/usr/bin/env python3
import numpy as np
import xarray
import numpy # for arrays!
#import scipy.io  # load matfiles
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

opt_save = 0

fdir = '/Users/irenavankova/Library/CloudStorage/GoogleDrive-irena.vanek@gmail.com/My Drive/Research/LANL/SGR/idealized/sg_pull_w_fraz_yesC/rd/rd_132B'

ds = xarray.open_dataset(f'{fdir}/timeSeriesStatsMonthly.0002-12-01.nc')
ds.load()

#lifw = np.squeeze(ds.timeMonthly_avg_landIceFreshwaterFluxTotal.data)
ustar = np.squeeze(ds.timeMonthly_avg_landIceFrictionVelocity.data)
#Tbl = np.squeeze(ds.timeMonthly_avg_landIceBoundaryLayerTracers_landIceBoundaryLayerTemperature.data)
#Sbl = np.squeeze(ds.timeMonthly_avg_landIceBoundaryLayerTracers_landIceBoundaryLayerSalinity.data)
T = ds['timeMonthly_avg_velocityX.data']

dsMesh = xarray.open_dataset(f'{fdir}/init.nc')
#dsMesh = xarray.open_dataset(f'{fdir}/restart.0003-01-01_00.00.00.nc')
dsMesh.load()
areaCell = np.squeeze(dsMesh.areaCell.data)
FloatingMask = np.squeeze(dsMesh.landIceFloatingMask.data)
landIceMask = np.squeeze(dsMesh.landIceMask.data)
xCell = np.squeeze(dsMesh.xCell.data)
yCell = np.squeeze(dsMesh.yCell.data)

max_level = dsMesh['maxLevelCell'] - 1  # Subtract 1 if your file uses 1-based indexing
#max_level = dsMesh['minLevelCell']  # Subtract 1 if your file uses 1-based indexing
Tbot = np.squeeze(T.isel(nVertLevels=max_level))

sgr = Tbot

# Set the values in sgr to 0 where the mask is True
mask = ((yCell > 76000) & (yCell < 78000) & (xCell > 510000) & (xCell < 520000)) | (yCell > 77000) | (yCell < 4000) | ((yCell < 9000) & (xCell < 520000))
sgr[mask] = np.NaN

FloatingMask = FloatingMask.astype(float)
FloatingMask[FloatingMask < 1] = np.NaN
landIceMask = landIceMask.astype(float)
landIceMask[landIceMask < 1] = np.NaN

sgr = sgr*landIceMask*FloatingMask

xmin = 459*1000
xmax = 639*1000
ymin = 5*1000
ymax = 75*1000
dx = 2000
dy = dx

x = np.arange(xmin, xmax+dx, dx)
y = np.arange(ymin, ymax+dy, dy)
x_grid, y_grid = np.meshgrid(x, y)

pts_Cell = np.column_stack((xCell, yCell))

pts_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))

sgr_grid = griddata(
    pts_Cell,          # Source coordinates (N, 2)
    sgr,         # Source values (N,)
    pts_grid,   # Target coordinates (M, 2)
    method='linear',  # Use 'linear' for bilinear interpolation
    fill_value=0.0
)

FM_grid = griddata(
    pts_Cell,          # Source coordinates (N, 2)
    FloatingMask,         # Source values (N,)
    pts_grid,   # Target coordinates (M, 2)
    method='linear',  # Use 'linear' for bilinear interpolation
    fill_value=np.NaN
)

LI_grid = griddata(
    pts_Cell,          # Source coordinates (N, 2)
    landIceMask,         # Source values (N,)
    pts_grid,   # Target coordinates (M, 2)
    method='linear',  # Use 'linear' for bilinear interpolation
    fill_value=np.NaN
)

# Reshape the interpolated data to match the 2D grid shape
sgr_grid = sgr_grid.reshape(y_grid.shape)
FM_grid = FM_grid.reshape(y_grid.shape)
LI_grid = LI_grid.reshape(y_grid.shape)

FM_grid[FM_grid < 0.9999] = 0
FM_grid[FM_grid > 0] = 1
FM_grid[FM_grid < 1] = np.NaN
sgr_grid = sgr_grid*FM_grid*LI_grid

print(np.shape(sgr_grid))
print(np.shape(x_grid))

sum_1d = np.nansum(sgr*areaCell)
print(f"Sum of the interpolated 1D field: {sum_1d}")
sum_2d = np.nansum(sgr_grid)*(x[2]-x[1])*(y[2]-y[1])
print(f"Sum of the interpolated 2D field: {sum_2d}")

#sgr_grid = sgr_grid*sum_1d/sum_2d

sum_2d = np.nansum(sgr_grid)*(x[2]-x[1])*(y[2]-y[1])
print(f"Sum of the corrected 2D field: {sum_2d}")


# =============================================================================
# Optional: Visualize the original and interpolated data
# =============================================================================
# Plot the original 1D data (if coordinates are 1D)

prop_1D = sgr; prop_grid = sgr_grid
#prop_1D = FloatingMask; prop_grid = FM_grid

v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(xCell, yCell, c=prop_1D, vmin=v_min, vmax=v_max, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('Original 1D Data')
plt.xlim([xmin, xmax])

# Plot the interpolated 2D data
plt.subplot(1, 2, 2)
plt.pcolormesh(x_grid, y_grid, prop_grid, vmin=v_min, vmax=v_max, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('Interpolated 2D Data')
plt.xlim([xmin, xmax])

plt.tight_layout()

#plt.savefig('interpolation_comparison.png')
plt.show()


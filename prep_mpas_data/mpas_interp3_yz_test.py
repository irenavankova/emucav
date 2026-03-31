#!/usr/bin/env python3
import numpy as np
import xarray
import numpy # for arrays!
#import scipy.io  # load matfiles
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

opt_save = 0

fdir = '/Users/irenavankova/Library/CloudStorage/GoogleDrive-irena.vanek@gmail.com/My Drive/Research/LANL/SGR/idealized/sg_pull_w_fraz_yesC/rd/rd_142N'

ds = xarray.open_dataset(f'{fdir}/timeSeriesStatsMonthly.0002-12-01.nc')
ds.load()
T = np.squeeze(ds.timeMonthly_avg_activeTracers_temperature.data)
S = np.squeeze(ds.timeMonthly_avg_activeTracers_salinity.data)
H = np.squeeze(ds.timeMonthly_avg_layerThickness.data)
ssh = np.squeeze(ds.timeMonthly_avg_ssh)

dsMesh = xarray.open_dataset(f'{fdir}/restart.0003-01-01_00.00.00.nc')
dsMesh.load()
xCell = np.squeeze(dsMesh.xCell.data)
yCell = np.squeeze(dsMesh.yCell.data)

target_x = 700*1000
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

nv = ds.dims['nVertLevels']
l = np.arange(0, nv, 1)

data_interp = np.zeros((nv, len(y)))
data_interp_y = np.zeros((nv, len(y)))
data_interp_H = np.zeros((nv, len(y)))

for k in range(ds.dims['nVertLevels']):
    level_data = T[mask, k]
    level_H = H[mask, k]
    level_y = yCell[mask]
    level_ssh = ssh[mask]

    # 2. Use square brackets [] for indexing
    # 3. Use : alone to select the full dimension
    data_interp[k, :] = griddata(level_y, level_data, y, method='linear')
    data_interp_H[k, :] = griddata(level_y, level_H, y, method='linear')
    data_interp_y[k, :] = y

    if k == 1:
        ssh_interp_H = griddata(level_y, level_ssh, y, method='linear')

data_interp_z = ssh_interp_H - np.nancumsum(data_interp_H, axis=0)

print(data_interp_y.shape)
print(data_interp_z.shape)
print(y.shape)
print(l.shape)

# Second step
final_data_interp = np.full(y_grid.shape, np.nan)

# Loop through each horizontal (y) location
for i in range(len(y)):
    # Get the vertical column of data and its corresponding irregular z-coordinates
    # We use data_interp[::-1, i] if your z-coordinates need to be strictly increasing
    z_column = data_interp_z[:, i]
    val_column = data_interp[:, i]

    # Remove NaNs to avoid interpolation errors
    mask_valid = ~np.isnan(z_column) & ~np.isnan(val_column)

    if np.any(mask_valid):
        # Create the 1D interpolation function for this specific column
        # bounds_error=False and fill_value=np.nan handles points outside the model range
        f_interp = interp1d(
            z_column[mask_valid],
            val_column[mask_valid],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate onto the regular z values for this column
        final_data_interp[:, i] = f_interp(z)

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





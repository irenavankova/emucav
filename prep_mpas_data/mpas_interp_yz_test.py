#!/usr/bin/env python3
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


fdir = '/Users/irenavankova/Library/CloudStorage/GoogleDrive-irena.vanek@gmail.com/My Drive/Research/LANL/SGR/idealized/sg_pull_w_fraz_yesC/rd/rd_112E'

ds = xr.open_dataset(f'{fdir}/timeSeriesStatsMonthly.0002-12-01.nc')
ds.load()
sgr = np.squeeze(ds.timeMonthly_avg_subglacialRunoffFlux.data)

dsMesh = xr.open_dataset(f'{fdir}/restart.0003-01-01_00.00.00.nc')
dsMesh.load()
landIceDraft = np.squeeze(dsMesh.landIceDraft.data)
yCell = np.squeeze(dsMesh.yCell.data)
areaCell = np.squeeze(dsMesh.areaCell.data)

# --- Example Usage with your desired axes ---

# 1. Define your specific axes (e.g., every 500m in Y, every 10m in Z)
# Note: These represent the "edges" of the boxes.
# The resulting grid will have size (len(y_axis)-1, len(z_axis)-1)
y_edges = np.arange(0, 80000, 2000)
z_edges = np.arange(landIceDraft.min(), 0, 10)
y_centers = np.squeeze(0.5 * (y_edges[:-1] + y_edges[1:]))
z_centers = np.squeeze(0.5 * (z_edges[:-1] + z_edges[1:]))

def project_to_custom_grid(runoff_flux, y_cell, ice_draft, y_axis, z_axis):
    """
    Projects runoff flux onto a user-defined structured grid.

    Parameters:
    runoff_flux: np.array (1, nCells) or (nCells,)
    y_cell:      np.array (nCells)
    ice_draft:   np.array (nCells)
    y_axis:      np.array (1D array of desired Y bin edges)
    z_axis:      np.array (1D array of desired Z bin edges)
    """

    # Remove the Time dimension (size 1)
    s_data = np.squeeze(runoff_flux)

    # The 'bins' argument accepts a list of the two edge arrays [y_edges, z_edges]
    grid_s, _, _ = np.histogram2d(
        y_cell,
        ice_draft,
        bins=[y_axis, z_axis],
        weights=s_data
    )

    return grid_s

# 2. Project
projected_runoff = project_to_custom_grid(
    sgr,
    yCell,
    landIceDraft,
    y_edges,
    z_edges
)

# 3. Quick check of the shape
print(f"Projected grid shape: {projected_runoff.shape}")
print(f"y-axis: {y_centers.shape}")
print(f"z-axis: {z_centers.shape}")

# 4. Rescale
fac = 1e6
sum_1d = np.nansum(sgr*areaCell) # horizontal area
print(f"Sum of the interpolated 1D field: {sum_1d/fac}")
sum_2d = np.nansum(projected_runoff)*(y_centers[2]-y_centers[1])*(z_centers[2]-z_centers[1]) # vertical area
print(f"Sum of the interpolated 2D field: {sum_2d/fac}")

projected_runoff = projected_runoff*sum_1d/sum_2d

sum_2d = np.nansum(projected_runoff)*(y_centers[2]-y_centers[1])*(z_centers[2]-z_centers[1])
print(f"Sum of the corrected 2D field: {sum_2d/fac}")

#
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
pcm = plt.pcolormesh(y_centers, z_centers, projected_runoff.T, cmap='hot_r')
plt.colorbar(label='Value')
plt.title('projected_runoff')
plt.show()


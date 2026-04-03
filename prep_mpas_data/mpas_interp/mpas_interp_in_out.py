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

def project_draft_to_yz(runoff_flux, y_cell, ice_draft, y_axis, z_axis):
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
    projected_flux, _, _ = np.histogram2d(
        y_cell,
        ice_draft,
        bins=[y_axis, z_axis],
        weights=s_data
    )

    return projected_flux

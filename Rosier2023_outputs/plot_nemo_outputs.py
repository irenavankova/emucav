#!/usr/bin/env python3

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random


def plot_nemo_velocities(file_U, file_V):
    """
    Plots a 4-panel figure showing horizontal velocity components:
    Left column: Top and Bottom U velocity components
    Right column: Top and Bottom V velocity components
    """
    # 1. Load Datasets safely
    try:
        ds_u = xr.open_dataset(file_U)
    except FileNotFoundError:
        print(f"Error: The U grid file '{file_U}' was not found.")
        return

    try:
        ds_v = xr.open_dataset(file_V)
    except FileNotFoundError:
        print(f"Error: The V grid file '{file_V}' was not found.")
        ds_u.close()
        return

    # 2. Extract and process variables (Mask zeros -> Interpolate forward -> Time Mean)
    # Process U components
    u_masked = ds_u["vozocrtx"].where(ds_u["vozocrtx"] != 0)
    u_interp = u_masked.interpolate_na(dim="depthu", method="nearest", fill_value="extrapolate")
    u_mean = u_interp.mean(dim="time_counter")
    utop = u_mean.isel(depthu=0)
    ubot = u_mean.isel(depthu=-1)

    # Process V components
    v_masked = ds_v["vomecrty"].where(ds_v["vomecrty"] != 0)
    v_interp = v_masked.interpolate_na(dim="depthv", method="nearest", fill_value="extrapolate")
    v_mean = v_interp.mean(dim="time_counter")
    vtop = v_mean.isel(depthv=0)
    vbot = v_mean.isel(depthv=-1)

    # 3. Define layout configuration (Left column = U, Right column = V)
    vel_configs = [
        {"data": utop, "title": "Top Zonal Velocity (U)", "ax_idx": (0, 0)},
        {"data": ubot, "title": "Bottom Zonal Velocity (U)", "ax_idx": (1, 0)},
        {"data": vtop, "title": "Top Meridional Velocity (V)", "ax_idx": (0, 1)},
        {"data": vbot, "title": "Bottom Meridional Velocity (V)", "ax_idx": (1, 1)},
    ]

    # 4. Initialize 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    # 5. Loop and plot with matched divergent colorbars for velocity
    for cfg in vel_configs:
        row, col = cfg["ax_idx"]
        ax = axes[row, col]

        # Force physical 1:1 aspect ratio
        ax.set_aspect("equal", adjustable="box")

        # Colorbar placement
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="4%")

        # Use divergent balance colormap for velocities (centered at 0)
        cfg["data"].plot(
            ax=ax,
            cmap="cmo.balance",
            cbar_kwargs={"label": "Velocity (m/s)", "cax": cax},
        )

        ax.set_title(cfg["title"], fontsize=11, weight="bold")
        ax.set_xlabel("X Grid Index" if row == 1 else "")
        ax.set_ylabel("Y Grid Index" if col == 0 else "")

    plt.subplots_adjust(wspace=0.35, hspace=0.2)
    fig.suptitle("NEMO Velocity Structure (U vs V)", fontsize=14, weight="bold", y=0.96)

    plt.show()

    # Close file connections
    ds_u.close()
    ds_v.close()


def plot_nemo_summary(file_bathy, file_outputs):
    # 1. Load Datasets safely
    try:
        ds_b = xr.open_dataset(file_bathy)
    except FileNotFoundError:
        print(f"Error: The bathymetry file '{file_bathy}' was not found.")
        return

    try:
        ds_o = xr.open_dataset(file_outputs)
    except FileNotFoundError:
        print(f"Error: The output file '{file_outputs}' was not found.")
        ds_b.close()
        return

    # 2. Extract Geometry variables
    bathy = ds_b["Bathymetry_isf"]
    draft = ds_b["isf_draft"]

    # 3. Process 2D Output variables
    stop = ds_o["stop"].where(ds_o["stop"] != 0).mean(dim="time_counter")
    ttop = ds_o["ttop"].where(ds_o["ttop"] != 0).mean(dim="time_counter")

    # Melt Rate conversion to m/yr (seconds in a year / ice density)
    seconds_per_year = 365.25 * 24 * 3600
    rho_ice = 1000.0
    melt_kg_m2_s = (
        ds_o["meltRate"].where(ds_o["meltRate"] != 0).mean(dim="time_counter")
    )
    melt = -melt_kg_m2_s * (seconds_per_year / rho_ice)

    # Top speed magnitude
    utop_mean = ds_o["utop"].where(ds_o["utop"] != 0).mean(dim="time_counter")
    vtop_mean = ds_o["vtop"].where(ds_o["vtop"] != 0).mean(dim="time_counter")
    speed_top = np.sqrt(utop_mean ** 2 + vtop_mean ** 2)

    # 4. Process 3D Output variables (Mask zeros -> Interpolate forward -> Time Mean -> Bottom Slice)
    salt_masked = ds_o["vosaline"].where(ds_o["vosaline"] != 0)
    temp_masked = ds_o["votemper"].where(ds_o["votemper"] != 0)

    salt_bottom = (
        salt_masked.interpolate_na(
            dim="deptht", method="nearest", fill_value="extrapolate"
        )
        .mean(dim="time_counter")
        .isel(deptht=-1)
    )
    temp_bottom = (
        temp_masked.interpolate_na(
            dim="deptht", method="nearest", fill_value="extrapolate"
        )
        .mean(dim="time_counter")
        .isel(deptht=-1)
    )

    # 5. Define configuration for all 8 panels (2 rows x 4 columns)
    plot_configs = [
        # Row 1: Geometry & Mixed Layer Inputs
        {"data": bathy, "cmap": "gist_earth", "title": "Seafloor Bathymetry", "unit": "Depth (m)"},
        {"data": draft, "cmap": "Blues", "title": "Ice Shelf Draft", "unit": "Draft (m)"},
        {"data": stop, "cmap": "cmo.haline", "title": "Top Salinity (stop)", "unit": "PSU"},
        {"data": ttop, "cmap": "cmo.thermal", "title": "Top Temperature (ttop)", "unit": "°C"},
        # Row 2:
        {"data": salt_bottom, "cmap": "cmo.haline", "title": "Bottom Salinity (vosaline)", "unit": "g/kg"},
        {"data": temp_bottom, "cmap": "cmo.thermal", "title": "Bottom Temperature (votemper)", "unit": "°C"},
        {"data": melt, "cmap": "RdBu_r", "title": "Ice Shelf Melt Rate", "unit": "m/yr"},
        {"data": speed_top, "cmap": "cmo.speed", "title": "Top Velocity Magnitude", "unit": "m/s"},
    ]

    # 6. Initialize 2x4 layout (No constrained_layout to prevent square plots from sizing colorbars oddly)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # 7. Loop and plot with perfectly matched colorbars and 1:1 square aspects
    for i, cfg in enumerate(plot_configs):
        ax = axes_flat[i]

        # Force the axis box to be square BEFORE plotting to represent the 64x64 grid to scale
        ax.set_aspect("equal", adjustable="box")

        # Create a divider to append a localized colorbar axis that matches the square's height
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="4%")

        # Pass 'cax' INSIDE cbar_kwargs so xarray routes it to the correct axis panel
        cfg["data"].plot(
            ax=ax,
            cmap=cfg["cmap"],
            cbar_kwargs={"label": cfg["unit"], "cax": cax},
        )

        ax.set_title(cfg["title"], fontsize=11, weight="bold")

        # Dynamic labels: Only outer boundaries keep text to avoid redundant overlaps
        ax.set_xlabel("X Grid Index" if i >= 4 else "")
        ax.set_ylabel("Y Grid Index" if i % 4 == 0 else "")

    # Clean up spacing margins between the square subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Force window manager to maximize the figure space on screen
    try:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    except AttributeError:
        try:
            mng.frame.Maximize(True)
        except AttributeError:
            pass

    plt.show()

    # Close file connections safely
    ds_b.close()
    ds_o.close()


# --- Execution ---
if __name__ == "__main__":
    # Define directories and matching string configurations

    snum = random.randint(1, 1000)

    cav_sample = f'SR_MLv1_1k_e{snum:05d}_00001'

    fdir = f"/Users/ivankova/Desktop/MLv1_1k/{cav_sample}"
    yr = 10

    file_bathy = f"{fdir}/bathy_meter.nc"

    # Grid file pointers dynamically swapped from the template string
    #file_out_T = f"{fdir}/{yr:04d}/{cav_sample}_1m_{yr:04d}0221_{yr + 1:04d}0225_grid_T.nc"
    file_out_T = f"{fdir}/{yr:04d}/{cav_sample}_1m_{yr:04d}0216_{yr + 1:04d}0220_grid_T.nc"
    file_out_U = file_out_T.replace("grid_T.nc", "grid_U.nc")
    file_out_V = file_out_T.replace("grid_T.nc", "grid_V.nc")

    print(f"Loading Model Data and Geometry...{snum:05d}")

    # 1. Run original summary plot (Scalar variables)
    plot_nemo_summary(file_bathy, file_out_T)

    # 2. Run new staggered velocity matrix plot
    print("Generating staggered U and V velocity matrix profile...")
    plot_nemo_velocities(file_out_U, file_out_V)
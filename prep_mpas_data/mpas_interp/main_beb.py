#!/usr/bin/env python3
import numpy as np
import xarray
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from mpas_interp_in_out import apply_masks_xy, interpolate_xy, interpolate_yz, project_draft_to_yz

#--------------------------------------------------------------------------------------------------------
# Load data
opt_save = 1
dir_fig_save = '/lcrc/group/e3sm/ac.vankova/beb/Ocean1'
dir_nc_save = dir_fig_save
#fdir_init = '/lcrc/group/e3sm/ac.vankova/compass/sg_tests/sg_pull_w_fraz_yesC/ra/ra_112A/ocean/isomip_plus/planar/2km/z-star/Ocean0/simulation'
#fdir = '/lcrc/group/e3sm/ac.vankova/compass/sg_tests/sg_pull_w_fraz_yesC/ra/ra_112A/ocean/isomip_plus/planar/2km/z-star/Ocean0/simulation'
fdir = '/lcrc/group/e3sm/ac.xylar/mpas_isomip_plus/isomip_plus_Ocean0-2_2k_5k_Jan_2019/isomip_plus_restart_fixed/ocean/isomip_plus/2km/Ocean1/forward'
fdir_init = fdir

year_init, year_end = 1, 21
month_init, month_end = 1, 13

# Define grid for interpolation
xmin, xmax = 459000, 639000
ymin, ymax = 5000, 75000
dx, dy = 2000, 2000
zmin, zmax = -700, 0
dz = 20

buffer = 4000
target_x = 700*1000
#target_x = xCell.max()-2000
m2km = 1000

x = np.arange(xmin, xmax + dx, dx)
y = np.arange(ymin, ymax + dy, dy)
x_grid, y_grid = np.meshgrid(x, y)
z = np.arange(zmin, zmax, dz)

y_centers = np.arange(ymin-dy/2, ymax + dy + dy/2, dy)
z_centers = np.arange(zmin-dz/2, zmax + dz/2, dz)


#--------------------------------------------------------------------------------------------------------
results_list_XY = []
results_list_YZ = []

for year in range(year_init, year_end):
    # Loop through months 1 to 12
    for month in range(month_init, month_end):
        # :02d ensures the number is always two digits (e.g., 1 becomes 01)
        filename = f'{fdir}/timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

        try:
            ds = xarray.open_dataset(filename)
            # --- Perform your analysis here ---
            print(f"Successfully opened: {filename}")

            # Always close the dataset to free up memory/file handles
            ds.close()

        except FileNotFoundError:
            print(f"File not found: {filename}")

        ds = xarray.open_dataset(filename)
        ds.load()
        dsMesh = xarray.open_dataset(f'{fdir_init}/init.nc')
        dsMesh.load()


        # Variables for XY output
        var_map_xy_2D = {
            'lifw': ds.timeMonthly_avg_landIceFreshwaterFlux,
            'ustar': ds.timeMonthly_avg_landIceFrictionVelocity,
            'Tbl': ds.timeMonthly_avg_landIceBoundaryLayerTracers_landIceBoundaryLayerTemperature,
            'Sbl': ds.timeMonthly_avg_landIceBoundaryLayerTracers_landIceBoundaryLayerSalinity
        }

        # Variables for YZ input
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

        if 'timeMonthly_avg_subglacialRunoffFlux' in ds:
            sgr = np.squeeze(ds.timeMonthly_avg_subglacialRunoffFlux.data)
        else:
            # Create an array of zeros with the same shape as ssh
            sgr = np.zeros_like(ssh)

        #--------------------------------------------------------------------------------------------------------
        # Dictionaries to store the processed results
        data_processed_xy = {}
        data_processed_yz = {}
        data_XY = {}
        data_YZ = {}
        data_YZi1 = {}

        # Masks
        #landIceMask = np.squeeze(dsMesh.landIceMask.data)
        landIceDraft = np.squeeze(dsMesh.landIceDraft.data)

        # Grid
        areaCell = np.squeeze(dsMesh.areaCell.data)
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
        #landIceMask = landIceMask.astype(float)
        #landIceMask[landIceMask < 1] = np.nan

        # Interpolate all xy variables
        for name, data_array in data_processed_xy.items():
            data_XY[f"{name}_xy"] = interpolate_xy(
                xCell, yCell, data_array, x_grid, y_grid, mask = None
            )

        #@@@@@@YZ

        # Only use points within a small 'epsilon' of target_x
        mask_near = (xCell > target_x - buffer) & (xCell < target_x + buffer)

        for name, data_obj in var_map_yz_3D.items():
            # Squeeze and initial data extraction
            data_processed_yz[name] = np.squeeze(data_obj.data)

        # Interpolate all yz variables
        for name, data_array in data_processed_yz.items():
            data_YZ[f"{name}_yz"], data_YZi1[f"{name}_grid1"] = interpolate_yz(data_array, H, yCell, ssh, mask_near, y, z)

        #@@@@@@ DRAFT TO YZ
        projected_runoff = project_draft_to_yz(
            sgr,
            yCell,
            landIceDraft,
            y_centers,
            z_centers
        )

        # Rescale
        fac = 1e6
        sum_1d = np.nansum(sgr*areaCell) # horizontal area
        print(f"Sum of the interpolated 1D field: {sum_1d/fac}")
        sum_2d = np.nansum(projected_runoff)*(y_centers[2]-y_centers[1])*(z_centers[2]-z_centers[1]) # vertical area
        print(f"Sum of the interpolated 2D field: {sum_2d/fac}")
        projected_runoff = projected_runoff*sum_1d/sum_2d
        sum_2d = np.nansum(projected_runoff)*(y_centers[2]-y_centers[1])*(z_centers[2]-z_centers[1])
        print(f"Sum of the corrected 2D field: {sum_2d/fac}")

        data_YZ['runoff_yz'] = projected_runoff.T

        #--------------------------------------------------------------------------------------------------------

        current_ds_xy = xarray.Dataset(
            data_vars={name: (("y", "x"), data) for name, data in data_XY.items()},
            coords={"y": y, "x": x}
        )

        # Add a time/iteration dimension so we can stack them later
        current_ds_xy = current_ds_xy.expand_dims(
            time=[f"{year:04d}-{month:02d}-01"]
        )

        results_list_XY.append(current_ds_xy)

        current_ds_yz = xarray.Dataset(
            data_vars={name: (("z", "y"), data) for name, data in data_YZ.items()},
            coords={"z": z, "y": y}
        )

        # Add a time/iteration dimension so we can stack them later
        current_ds_yz = current_ds_yz.expand_dims(
            time=[f"{year:04d}-{month:02d}-01"]
        )

        results_list_YZ.append(current_ds_yz)

        shapes = {name: data.shape for name, data in data_XY.items()}
        print(shapes)
        shapes = {name: data.shape for name, data in data_YZ.items()}
        print(shapes)
        print(projected_runoff.shape)

        #--------------------------------------------------------------------------------------------------------
        # Visualization of XY fields
        varxy = 'Tbot'
        prop_1D = data_processed_xy[f'{varxy}']; prop_grid = data_XY[f'{varxy}_xy']
        v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
        v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.scatter(xCell/m2km, yCell/m2km, c=prop_1D, vmin=v_min, vmax=v_max, cmap='hot_r')
        plt.colorbar()
        plt.title('Original Data'); plt.ylabel('y (km)'); plt.xlabel('x (km)')
        plt.xlim(np.array([xmin, xmax]) / m2km)
        plt.ylim(np.array([ymin, ymax]) / m2km)

        plt.subplot(2, 1, 2)
        plt.pcolormesh(x_grid/m2km, y_grid/m2km, prop_grid, vmin=v_min, vmax=v_max, cmap='hot_r')
        plt.colorbar()
        plt.title('Interpolated Data'); plt.ylabel('y (km)'); plt.xlabel('x (km)')
        plt.xlim(np.array([xmin, xmax]) / m2km)
        plt.ylim(np.array([ymin, ymax]) / m2km)
        plt.tight_layout()
        if opt_save == 1:
            plt.savefig(f'{dir_fig_save}/interp_xy_{varxy}.{year:04d}-{month:02d}-01.png', bbox_inches='tight', dpi=150)
        elif opt_save == -1:
            plt.show()

        # Visualization of YZ fields
        varyz = 'T'
        prop1 = data_YZi1[f'{varyz}_grid1']; prop2 = data_YZ[f'{varyz}_yz']
        v_min = np.nanmin([np.nanmin(prop_1D), np.nanmin(prop_grid)])
        v_max = np.nanmax([np.nanmax(prop_1D), np.nanmax(prop_grid)])

        plt.figure(figsize=(12, 5))
        # Plot the interpolated 2D data
        plt.subplot(1, 2, 1)
        nv = ds.dims['nVertLevels']
        l = np.arange(0, nv, 1)
        plt.pcolormesh(y/m2km, -l, prop1, cmap='hot_r', shading='nearest')
        plt.colorbar()
        plt.title('Interp levels'); plt.ylabel('level #'); plt.xlabel('y (km)')
        plt.xlim(np.array([ymin, ymax]) / m2km)

        plt.subplot(1, 2, 2)
        plt.pcolormesh(y/m2km, z, prop2, cmap='hot_r', shading='nearest')
        plt.colorbar()
        plt.title('Interp z'); plt.ylabel('z (m)'); plt.xlabel('y (km)')
        plt.xlim(np.array([ymin, ymax]) / m2km)
        plt.tight_layout()

        if opt_save == 1:
            plt.savefig(f'{dir_fig_save}/interp_yz_{varyz}.{year:04d}-{month:02d}-01.png', bbox_inches='tight', dpi=150)
        elif opt_save == -1:
            plt.show()

        # Visualization of YZ projection
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(yCell/m2km, landIceDraft, c=sgr, cmap='hot_r')
        plt.colorbar()
        plt.title('Original runoff'); plt.ylabel('z (m)'); plt.xlabel('y (km)')
        plt.xlim(np.array([ymin, ymax]) / m2km)
        plt.ylim(np.array([zmin, zmax]))

        plt.subplot(1, 2, 2)
        pcm = plt.pcolormesh(y/m2km, z, data_YZ[f'runoff_yz']/dx*dz, cmap='hot_r', shading='nearest')
        plt.colorbar()
        plt.title('Projected runoff'); plt.ylabel('z (m)'); plt.xlabel('y (km)')
        plt.xlim(np.array([ymin, ymax]) / m2km)
        plt.ylim(np.array([zmin, zmax]))
        plt.tight_layout()

        if opt_save == 1:
            plt.savefig(f'{dir_fig_save}/runoff_projected.{year:04d}-{month:02d}-01.png', bbox_inches='tight', dpi=150)
        elif opt_save == -1:
            plt.show()


# Save to NetCDF
if results_list_XY:
    # Combine all monthly datasets into one large dataset
    combined_ds = xarray.concat(results_list_XY, dim="time")
    combined_ds.to_netcdf(f'{dir_nc_save}/output_data_xy.nc')
    print("File saved successfully with XY variables:", list(combined_ds.data_vars))
else:
    print("No XY output data were processed.")

if results_list_YZ:
    # Combine all monthly datasets into one large dataset
    combined_ds = xarray.concat(results_list_YZ, dim="time")
    combined_ds.to_netcdf(f'{dir_nc_save}/input_data_yz.nc')
    print("File saved successfully with YZ variables:", list(combined_ds.data_vars))
else:
    print("No YZ input data were processed.")


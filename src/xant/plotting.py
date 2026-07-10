from typing import Optional, Union

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import leafmap.foliumap as leafmap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import animation
from xrench.units import ureg

from .utils.calc import round2base
from .utils.h3utils import h3xr2gpd

plt.rcParams["animation.html"] = "jshtml"

_DEG_DIMS = ["theta", "phi", "azimuth", "elevation", "lat", "lon", "tvx", "tvy", "au", "av"]


def plot_antenna_pattern(
    data: xr.DataArray,
    x: str,
    ax: plt.Axes | None = None,
    plot_type="line",
    projection: str | None = None,
    yspan: float = 40,
    ymax: float | None = None,
    quantity: str = "db",
    animate: bool = False,
    **kwargs,
) -> plt.Axes | animation.Animation:
    """
    Convenience function for platting antenna pattern. Takes xr.DataArray and will format polar antenna plot.

    Parameters
    ----------
    data : xr.DataArray
        Data from Antenna.request_data()
    x : str
        Dimension for x-axis
    ax : optional, plt.Axes
        Matplotlib axes to plot on
    plot_type : str
        Type of plot ("line", "pcolormesh",...)
    projection : str
        Plot project, default is polar
    yspan : float
        Span of y-axis in dB
    ymax : optional, float
        Max scale for y/v axis
    quantity : str
        Quantity to plot (db, phase)
    animate : bool
        Whether to animate the plot

    Returns:
    -------
    ax : plt.Axes, animation.Animation
        Axes handle of the plot or Animation instance for animated plots
    """
    # Make data copy
    data = data.copy()

    # Deal with projection
    if x in _DEG_DIMS and projection is None:
        projection = "polar"
    elif projection is None:
        projection = "rectilinear"

    # Make axis if none and format plot
    if ax is None:
        if projection == "map":
            # This is the map projection we want to plot *onto*
            map_proj = ccrs.LambertConformal(
                central_longitude=np.rad2deg(data.coords["lon"].mean().values),
                central_latitude=np.rad2deg(data.coords["lat"].mean().values),
            )

            # cimgt.Stamen.get_image = image_spoof  # reformat web request for street map spoofing
            # osm_img = cimgt.Stamen("terrain")  # spoofed, downloaded street map
            osm_img = cimgt.GoogleTiles(style="street")
            projection = osm_img.crs
            kwargs = {
                **kwargs,
                **dict(transform=ccrs.PlateCarree(), subplot_kws={"projection": map_proj}),
            }

        fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
    if isinstance(projection, str):
        if projection.lower() == "polar":
            ax.set_theta_zero_location("N")  # point the origin towards the top
            ax.set_theta_direction(-1)  # change direction to CCW
            ax.set_thetamin(-180)  # set the limits
            ax.set_thetamax(180)

            # Set theta ticks
            ax.set_thetagrids(np.linspace(-150, 180, 12))

    # Calculate Quantities
    if quantity.lower() == "db":
        data.data = (np.abs(data.data) ** 2).to("dB")
        data.attrs = {**data.attrs, **dict(long_name="Gain", units="dBi", description="Gain")}
    elif quantity.lower() == "phase":
        # Get rid of quantity
        data.data = data.data.magnitude
        # Calculate angle
        data = xr.ufuncs.angle(data)
        # Convert to degrees
        data.data = (data.data * ureg.radian).to("degree")
        # Add attributes
        data.attrs = {**data.attrs, **dict(long_name="Phase", units="Degree", description="Phase")}

    else:
        raise ValueError(f"Quantity {quantity} not understood.")

    # Convert radians to degrees for all dimensions that should be degrees
    updated = dict()
    for k in data.coords.keys():
        if k in _DEG_DIMS:
            u = data.coords[k] * ureg.radian
            u.data = u.data.to("degree").magnitude
            updated[k] = u
    data = data.assign_coords(updated)
    updated = dict()
    # Revert for x-dimension if polar
    if projection == "polar":
        if x in _DEG_DIMS:
            u = data.coords[x] * ureg.degree
            u.data = u.data.to("radian").magnitude
            updated[x] = u

    data = data.assign_coords(updated)

    # Squeeze singleton dimensions
    data = data.squeeze()

    # Get plot bounds
    datamax = round2base(data.data.max().item().magnitude)
    if ymax is None:
        datamax = round2base(data.data.max().item().magnitude)
    else:
        datamax = ymax
    datamin = datamax - yspan

    if len(data.shape) == 2 and animate is not False:
        # Initialize plot
        def init_func():
            # Scale yaxis - needs to be done after plotting
            if projection.lower() == "polar":
                ax.set_rmax(datamax)
                ax.set_rmin(datamin)
                polar_grids = np.linspace(datamin, datamax, 5)
                ax.set_rgrids(
                    polar_grids,
                    labels=[""] + [f"{i}" for i in polar_grids[1:-1]] + [f"{polar_grids[-1]} dBi"],
                    angle=45,
                    fmt=None,
                )
            else:
                ax.set_ylim(datamin, datamax)

            for line in ax.get_lines():
                line.remove()
            # Initialize plot
            data.isel({animate: 0}).plot.line(x=x)
            return ax.get_lines()

        def antenna_movie(i):
            # Remove line and reset color cycle
            lines = ax.get_lines()
            line = lines.pop(0)
            line.remove()
            ax.set_prop_cycle(None)

            # Select data
            data.isel({animate: i}).plot.line(x=x)

            # Scale yaxis - needs to be done after plotting
            if projection.lower() == "polar":
                ax.set_rmax(datamax)
                ax.set_rmin(datamin)
                polar_grids = np.linspace(datamin, datamax, 5)
                ax.set_rgrids(
                    polar_grids,
                    labels=[""] + [f"{i}" for i in polar_grids[1:-1]] + [f"{polar_grids[-1]} dBi"],
                    angle=45,
                    fmt=None,
                )
                # Remove ylabel
                ax.set_ylabel("")
            else:
                ax.set_ylim(datamin, datamax)

            return ax.get_lines()

        ani = animation.FuncAnimation(
            plt.gcf(),
            antenna_movie,
            save_count=data.coords[animate].shape[0],
            init_func=init_func,
            interval=10,
            blit=True,
        )

        return ani
    if len(data.shape) > 2 and animate is not False and isinstance(projection, ccrs.Projection):
        # Initialize plot
        extent = np.array(
            [
                data.lon.min() - 0.05,
                data.lon.max() + 0.05,
                data.lat.min() - 0.05,
                data.lat.max() + 0.05,
            ],
        )
        data.isel({animate: 0}).plot(
            transform=ccrs.PlateCarree(),  # the data's projection
            subplot_kws={"projection": map_proj},  # the plot's projection
            alpha=0.75,
            vmin=round2base(data.max().data.magnitude) - yspan,
            vmax=round2base(data.max().data.magnitude),
        )

        ax.set_extent(extent)  # set extents
        ax.set_xticks(
            np.linspace(extent[0], extent[1], 5),
            crs=ccrs.PlateCarree(),
        )  # set longitude indicators
        ax.set_yticks(
            np.linspace(extent[2], extent[3], 7)[1:],
            crs=ccrs.PlateCarree(),
        )  # set latitude indicators
        lon_formatter = LongitudeFormatter(
            number_format="0.2f",
            dateline_direction_label=True,
        )  # format lons
        lat_formatter = LatitudeFormatter(number_format="0.2f")  # format lats
        ax.xaxis.set_major_formatter(lon_formatter)  # set lons
        ax.yaxis.set_major_formatter(lat_formatter)  # set lats

        ax.set_xlabel("")
        ax.set_ylabel("")
        scale = np.ceil(
            -np.sqrt(2) * np.log(np.divide((extent[1] - extent[0]) / 2.0, 350.0)),
        )  # empirical solve for scale based on zoom
        scale = ((scale < 20) and scale) or 19  # scale cannot be larger than 19
        ax.add_image(osm_img, int(scale))  # add OSM with zoom specification

        def antenna_movie(i):
            # Delete the axes
            for ax in fig.axes:
                ax.remove()
            # Add axes in
            ax = plt.axes(
                projection=osm_img.crs,
            )  # project using coordinate reference system (CRS) of street map
            # Plot DataArray
            data.isel({animate: i}).plot(
                transform=ccrs.PlateCarree(),  # the data's projection
                subplot_kws={"projection": map_proj},  # the plot's projection
                alpha=0.75,
                vmin=round2base(data.max().data.magnitude) - yspan,
                vmax=round2base(data.max().data.magnitude),
            )
            ax.set_extent(extent)  # set extents
            ax.set_xticks(
                np.linspace(extent[0], extent[1], 5),
                crs=ccrs.PlateCarree(),
            )  # set longitude indicators
            ax.set_yticks(
                np.linspace(extent[2], extent[3], 7)[1:],
                crs=ccrs.PlateCarree(),
            )  # set latitude indicators
            lon_formatter = LongitudeFormatter(
                number_format="0.2f",
                dateline_direction_label=True,
            )  # format lons
            lat_formatter = LatitudeFormatter(number_format="0.2f")  # format lats
            ax.xaxis.set_major_formatter(lon_formatter)  # set lons
            ax.yaxis.set_major_formatter(lat_formatter)  # set lats

            ax.set_xlabel("")
            ax.set_ylabel("")
            scale = np.ceil(
                -np.sqrt(2) * np.log(np.divide((extent[1] - extent[0]) / 2.0, 350.0)),
            )  # empirical solve for scale based on zoom
            scale = ((scale < 20) and scale) or 19  # scale cannot be larger than 19
            ax.add_image(osm_img, int(scale))  # add OSM with zoom specification

            return (ax.collections[0],)

        ani = animation.FuncAnimation(
            plt.gcf(),
            antenna_movie,
            save_count=data.coords[animate].shape[0],
            interval=10,
            blit=True,
        )

        return ani

    # Plot
    if plot_type == "line":
        data.plot.line(x=x, **kwargs, ax=ax)
    else:
        data.plot(
            x=x,
            **{**dict(alpha=0.8, vmin=datamin, vmax=datamax, cmap=plt.cm.viridis), **kwargs},
        )

    # Scale yaxis - needs to be done after plotting
    if isinstance(projection, str):
        if projection.lower() == "polar":
            ax.set_rmax(datamax)
            ax.set_rmin(datamin)
            rgrids = np.linspace(datamin, datamax, 5)
            ax.set_rgrids(
                rgrids,
                labels=[""] + [f"{i}" for i in rgrids[1:-1]] + [f"{rgrids[-1]} dBi"],
                angle=45,
                fmt=None,
            )
            # Remove ylabel
            ax.set_ylabel("")
        elif plot_type != "quadmesh" and plot_type != "pcolormesh":
            ax.set_ylim(datamin, datamax)
    elif isinstance(projection, ccrs.Projection):
        extent = np.array(
            [
                data.lon.min() - 0.05,
                data.lon.max() + 0.05,
                data.lat.min() - 0.05,
                data.lat.max() + 0.05,
            ],
        )

        ax.set_extent(extent)  # set extents
        ax.set_xticks(
            np.linspace(extent[0], extent[1], 5),
            crs=ccrs.PlateCarree(),
        )  # set longitude indicators
        ax.set_yticks(
            np.linspace(extent[2], extent[3], 5)[1:],
            crs=ccrs.PlateCarree(),
        )  # set latitude indicators
        lon_formatter = LongitudeFormatter(
            number_format="0.2f",
            dateline_direction_label=True,
        )  # format lons
        lat_formatter = LatitudeFormatter(number_format="0.2f")  # format lats
        ax.xaxis.set_major_formatter(lon_formatter)  # set lons
        ax.yaxis.set_major_formatter(lat_formatter)  # set lats

        ax.set_xlabel("")
        ax.set_ylabel("")
        scale = np.ceil(
            -np.sqrt(2) * np.log(np.divide((extent[1] - extent[0]) / 2.0, 350.0)),
        )  # empirical solve for scale based on zoom
        scale = ((scale < 20) and scale) or 19  # scale cannot be larger than 19
        ax.add_image(osm_img, int(scale))  # add OSM with zoom specification

    return ax


def lm_fill_color(feature):
    return {"fillColor": feature["properties"]["color"], "fillOpacity": 0.5, "stroke": False}


def plot_lm_h3(data, center):
    dh3 = data.copy()
    dh3.data = (np.abs(dh3.data) ** 2).to("dB")
    gainh3 = h3xr2gpd(dh3)
    m = leafmap.Map(center=center, zoom=12, tiles="cartodb positron")
    m.add_gdf(gainh3, layer_name="Gain", style_callback=lm_fill_color)

    return m

"""Main module."""

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from hics import HCS
from hics.geo.geoutils import relative_azimuth
from hics.plotting import view_surface_profile
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pint import Quantity
from scipy.spatial.transform import Rotation
from shapely.geometry import LineString
from xrench.units import ureg
from xrench.xrutils import vector_norm

from ..antenna.core import Antenna
from ..antenna.polarization import thetaphi2xyz
from ..utils.calc import round2base
from ..utils.conversions import cartesian2uvw, uvw2phitheta
from . import propagators

plt.rcParams["animation.html"] = "jshtml"


def _setup_map_axes(ax: plt.Axes, osm_img: cimgt.GoogleTiles, extent: np.ndarray) -> None:
    """Configure map axes with extent, tick formatters, and OSM tile overlay."""
    ax.set_extent(extent)
    ax.set_xticks(
        np.linspace(extent[0], extent[1], 5),
        crs=ccrs.PlateCarree(),
    )
    ax.set_yticks(
        np.linspace(extent[2], extent[3], 7)[1:],
        crs=ccrs.PlateCarree(),
    )
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(number_format="0.2f", dateline_direction_label=True),
    )
    ax.yaxis.set_major_formatter(
        LatitudeFormatter(number_format="0.2f"),
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    scale = np.ceil(
        -np.sqrt(2) * np.log(np.divide((extent[1] - extent[0]) / 2.0, 350.0)),
    )
    scale = ((scale < 20) and scale) or 19
    ax.add_image(osm_img, int(scale))


def calculate_spatial_link(
    tx: Antenna,
    power: Quantity,
    rx: Antenna,
    propagation="fspl",
    additional_loss: Quantity | xr.DataArray = None,
    **kwargs,
):
    """
    Calculates the transmit gain, path loss, and receive gain between to antennas.

    Parameters
    ----------
    tx : Antenna
        Transmitting antenna
    power : Quantity
        Transmit power input to antenna
    rx : Antenna
        Receiving antenna
    propagation : str
        Propagation model to use.

    Returns:
    -------
    rxpower : xr.DataArray
        Received power
    txgain : xr.DataArray
        Total power (apolar) transmit gain
    rxgain : xr.DataArray
        Total power (apolar) transmit gain
    txrx : xr.DataArray
        Combine transmit and receive gain
    """
    # Get path loss first - because angles will be updated
    prop_loss = getattr(propagators, propagation.lower())(
        txcs=tx.hcs,
        rxcs=rx.hcs,
        frequency=tx.data.frequency,
        additional_loss=additional_loss,
        **kwargs,
    )
    prop_loss.data = prop_loss.data.to_base_units()

    # Create a tx cs where x/y plane points at horizon by creating a cs at the same location with the z or normal
    # vector pointing in the altitude dimension and the x-vector pointing along the longitude line

    # First need a way to check if coordinate system is on surface - best way is probably checking altitude
    not_surface_alt = -40000 * ureg.feet  # Units are meters
    if np.all(tx.hcs.llh[-1] > not_surface_alt) and np.all(rx.hcs.llh[-1] > not_surface_alt):
        txhorizon = HCS.from_crs(
            tx.hcs.llh,
            un="h",
            ux="lon",
            hagl=False,
        )
        rxhorizon = HCS.from_crs(
            rx.hcs.llh,
            un="h",
            ux="lon",
            hagl=False,
        )
    else:
        # Create new coordinate system objects that are pointed at each other
        untx = tx.hcs.relative_position(rx.hcs)

        # Get RX positions
        x = untx.sel(position="x")
        x = x.drop_vars("position")
        y = untx.sel(position="y")
        y = y.drop_vars("position")
        z = untx.sel(position="z")
        z = z.drop_vars("position")
        phi, theta = uvw2phitheta(*cartesian2uvw(x=x, y=y, z=z))
        untx /= vector_norm(untx, "position")
        untx = untx.assign_coords({"u": "z"})
        A = thetaphi2xyz(phi=phi, theta=theta)
        A = A.rename(dict(new_polarization="position"))
        A = A.rename(dict(polarization="u"))
        A = A.assign_coords({"u": ["x", "y"]})

        utx = xr.concat([A, untx], dim="u")

        # Convert nans to zeros
        utx = utx.fillna(0)

        txhorizon = HCS((0, 0, 0) * ureg.m, rotation=Rotation.from_matrix(utx.T), reference=tx.hcs)

        rxhorizon = HCS(
            txhorizon.relative_position(rx.hcs),
            rotation=Rotation.from_euler("ZXZ", [0, 180, 0], degrees=True),
            reference=txhorizon,
        )

    # Rotate the x-axis towards
    rel_azs = relative_azimuth(txhorizon, rxhorizon)

    # Get relative position of rx node to txhorizon coordinate system and tx node to rxhorizon coordinate system
    txh_rp = txhorizon.relative_position(rxhorizon)
    rxh_rp = rxhorizon.relative_position(txhorizon)

    # Compute horizon angles and format dataarray
    tx_ha = np.arctan2(
        txh_rp.sel(position="z"),
        np.sqrt(txh_rp.sel(position="x") ** 2 + txh_rp.sel(position="y") ** 2),
    )
    tx_ha = tx_ha.drop_vars("position")
    tx_ha.data = tx_ha.data.to("degree")
    rx_ha = np.arctan2(
        rxh_rp.sel(position="z"),
        np.sqrt(rxh_rp.sel(position="x") ** 2 + rxh_rp.sel(position="y") ** 2),
    )
    rx_ha = rx_ha.drop_vars("position")
    rx_ha.data = rx_ha.data.to("degree")

    # Modify horizon angle if angle given in propagation loss
    if tx_ha.dims == () and tx_ha.coords == dict():
        if "tx_angle" in prop_loss.coords.keys():
            tx_ha += prop_loss.tx_angle.copy().item() - tx_ha.copy()
        if "rx_angle" in prop_loss.coords.keys():
            rx_ha += prop_loss.rx_angle.copy().item() - rx_ha.copy()
    else:
        if "tx_angle" in prop_loss.coords.keys():
            tx_ha += prop_loss.tx_angle.copy() - tx_ha.copy()
        if "rx_angle" in prop_loss.coords.keys():
            rx_ha += prop_loss.rx_angle.copy() - rx_ha.copy()

    # Add to prop loss in not in
    if "tx_angle" not in prop_loss.coords.keys():
        prop_loss = prop_loss.assign_coords(tx_angle=tx_ha)
    if "rx_angle" not in prop_loss.coords.keys():
        prop_loss = prop_loss.assign_coords(rx_angle=rx_ha)

    # Point the txhorizon and rxhorizon coordinate systems towards each other
    if tx_ha.shape == () and rel_azs[0].shape == ():
        tx_rots = Rotation.from_euler(
            "XYX",
            [-tx_ha.item().to("degree").magnitude, -rel_azs[0].item().to("degree").magnitude, 90],
            degrees=True,
        )
        tx_rots = xr.DataArray(tx_rots, dims=tx_ha.dims, coords=tx_ha.coords)

        tx_pos = xr.DataArray(
            (0, 0, 0) * ureg.m,
            dims=("position",),
            coords=dict(position=["x", "y", "z"]),
        ) * xr.zeros_like(tx_rots)
    else:
        tx_rots = [
            Rotation.from_euler(
                "XYX",
                [-txa.to("degree").magnitude, -np.array([ra.to("degree").magnitude])[0], 90],
                degrees=True,
            )
            for txa, ra in zip(tx_ha.data, rel_azs[0].data, strict=False)
        ]
        tx_rots = xr.DataArray(tx_rots, dims=tx_ha.dims, coords=tx_ha.coords)

        tx_pos = xr.DataArray(
            (0, 0, 0) * ureg.m,
            dims=("position",),
            coords=dict(position=["x", "y", "z"]),
        ) * xr.zeros_like(tx_rots)

    txcs = HCS(tx_pos, rotation=tx_rots, reference=txhorizon)
    if rx_ha.shape == () and rel_azs[1].shape == ():
        rx_rots = Rotation.from_euler(
            "XYX",
            [-rx_ha.item().to("degree").magnitude, -rel_azs[1].item().to("degree").magnitude, 90],
            degrees=True,
        )
        rx_rots = xr.DataArray(rx_rots, dims=rx_ha.dims, coords=rx_ha.coords)
        rx_pos = xr.DataArray(
            (0, 0, 0) * ureg.m,
            dims=("position",),
            coords=dict(position=["x", "y", "z"]),
        ) * xr.zeros_like(rx_rots)

    else:
        rx_rots = [
            Rotation.from_euler(
                "XYX",
                [-rxa.to("degree").magnitude, -np.array([ra.to("degree").magnitude])[0], 90],
                degrees=True,
            )
            for rxa, ra in zip(rx_ha.data, rel_azs[1].data, strict=False)
        ]
        rx_rots = xr.DataArray(rx_rots, dims=rx_ha.dims, coords=rx_ha.coords)
        rx_pos = xr.DataArray(
            (0, 0, 0) * ureg.m,
            dims=("position",),
            coords=dict(position=["x", "y", "z"]),
        ) * xr.zeros_like(rx_rots)

    rxcs = HCS(rx_pos, rotation=rx_rots, reference=rxhorizon)

    # Determine TX gain towards RX positions
    zrs = xr.zeros_like(rx_rots)
    zrs.data = np.zeros(zrs.shape) * ureg.degree
    tx_gain = tx.request_data(
        theta=zrs,
        phi=zrs,
        coordinate_frame="phitheta",
        hcs=txcs,
    )

    # Determine TX gain towards RX positions
    rx_gain = rx.request_data(
        theta=zrs,
        phi=zrs,
        coordinate_frame="phitheta",
        hcs=rxcs,
    )

    # Should place RX power at RX node - grab accordingly
    llh = rxcs.llh

    # Add back in lat/lon/alt for plotting purposes
    tx_gain.coords["lat"] = llh[0]
    tx_gain.coords["lon"] = llh[1]
    tx_gain.coords["h"] = llh[2]

    rx_gain.coords["lat"] = llh[0]
    rx_gain.coords["lon"] = llh[1]
    rx_gain.coords["h"] = llh[2]

    # Drop spatial variables
    tx_gain = tx_gain.drop_vars("phi")
    tx_gain = tx_gain.drop_vars("theta")

    rx_gain = rx_gain.drop_vars("phi")
    rx_gain = rx_gain.drop_vars("theta")

    # Select correct polarization
    # Map theta/phi to vertical/horizontal
    pol_map = dict(theta="horizontal", phi="vertical")
    if "theta" in tx_gain.polarization:
        tx_gain = tx_gain.sel(polarization=["theta", "phi"])
        tx_gain = tx_gain.assign_coords(
            dict(polarization=[pol_map[k] for k in tx_gain.polarization.values]),
        )
    if "theta" in rx_gain.polarization:
        rx_gain = rx_gain.sel(polarization=["theta", "phi"])
        rx_gain = rx_gain.assign_coords(
            dict(polarization=[pol_map[k] for k in rx_gain.polarization.values]),
        )
        # Note that RX x-vector is opposite to TX so we need to convert, this should also mean that we don't need to
        # conjugate
        rx_gain.loc[dict(polarization="horizontal")] = (
            rx_gain.loc[dict(polarization="horizontal")] * -1
        )

    # In order to account for the potential asymmetric propagation loss of the individual horizontal and vertical
    # components
    if tx_gain.polarization.size == 1 and "polarization" in prop_loss.dims:
        # Apolar antenna — no polarization to project onto H/V ITM loss.
        # Use worst-case (max) polarization loss as a conservative estimate
        incident_pd = tx_gain / np.sqrt(prop_loss.max(dim="polarization"))
    else:
        incident_pd = tx_gain / np.sqrt(prop_loss)
    tx_pl_rx = incident_pd * rx_gain

    # Get RX gain by dividing the RX power by the incident power density
    if "polarization" in incident_pd.dims:
        incident_pol = incident_pd / vector_norm(incident_pd, "polarization")
        if rx_gain.polarization.size != 2:
            tx_pl_rx = vector_norm(incident_pd, "polarization") * rx_gain
    else:
        incident_pol = 1
    if rx_gain.polarization.size == 2:
        rx_gain = rx_gain * incident_pol

    # Need to compute total propagation loss for propagators that handle polarization
    if "itm" in propagation:
        # Add propagation loss magnitude
        total_prop = (
            vector_norm(tx_gain, "polarization") / vector_norm(incident_pd, "polarization")
        ) ** 2
        total_prop = (total_prop.assign_coords(polarization="total")).expand_dims("polarization")
        # RFLINK ITM includes the reference attenuation which can be different for h/v polarization
        if "rflink" in propagation and "ref" in prop_loss.coords:
            if "polarization" in prop_loss.ref.coords:
                reftot = vector_norm(prop_loss.ref, "polarization")
                total_prop = total_prop.assign_coords(ref=reftot)
        prop_loss = prop_loss.drop_vars("ipol")
        prop_loss = xr.concat([prop_loss, total_prop], "polarization")

    # Only return apolar magnitudes
    if tx_gain.polarization.size == 2:
        tx_gain = vector_norm(tx_gain, "polarization")
        tx_gain.data = (np.abs(tx_gain.data) ** 2) * ureg.dimensionless
    if rx_gain.polarization.size == 2:
        rx_gain = rx_gain.sum(dim="polarization")
        rx_gain.data = (np.abs(rx_gain.data) ** 2) * ureg.dimensionless
    if tx_pl_rx.polarization.size == 2:
        tx_pl_rx = tx_pl_rx.sum(dim="polarization")

    # Determine received power by multiplying power by the tx-path loss-rx product and convert to dBm
    rx_power = power.to("watt") * abs(tx_pl_rx) ** 2
    rx_power.data = rx_power.data.to("dBm")

    # Add gains as coords
    rx_power = rx_power.assign_coords(tx_power=power, tx_gain=tx_gain, rx_gain=rx_gain)

    # Attributes that xarray uses for plotting (long_name and units)
    attrs = dict(long_name="Power", units="dBm", description="Received Power")

    # Add attrs
    rx_power.attrs = {**rx_power.attrs, **attrs}

    # Squeeze singleton dimensions
    rx_power = rx_power.squeeze()

    prop_loss.attrs = {**prop_loss.attrs, **kwargs}

    return rx_power, prop_loss, incident_pol, txcs, rxcs


def transmit_power_density(
    tx: Antenna,
    power: Quantity,
    lats: Quantity,
    lons: Quantity,
    hs: Quantity,
    propagation: str = "fspl",
    mask_los: bool = True,
    convert_kwargs: dict | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Projects transmit antenna power onto surface. This is returns the transmitted EIRP minus the path loss in units of
    Watts and is equivalent to the received power of a polarization matched isotropic radiator.

    Parameters
    ----------
    tx : Antenna
        Transmitting antenna gain pattern
    power : Quantity
        Input power to the antenna
    lats : Quantity
        Latitudes to sample
    lons : Quantity
        Longitudes to sample
    hs : Quantity
        Altitudes to sample
    propagation : str
        Propagation model to use
    mask_los : bool
        Mask the line of sight with nans
    **kwargs : dict
        Extra kwargs to be passed to antenna data request
    """
    # Initialize
    if convert_kwargs is None:
        convert_kwargs = dict()
    # Get gain data: TODO - should the angles be compensated for ITM?
    data = tx.request_data(
        lat=lats,
        lon=lons,
        h=hs,
        coordinate_frame="llh",
        convert_kwargs=convert_kwargs,
        **kwargs,
    )

    # Convert to power
    data.data = data.data * np.sqrt(power.to("watt"))

    # Need to convert lat/lon back to degrees
    data = data.assign_coords(lon=(data.lon.data * ureg.radians).to("degrees"))
    data = data.assign_coords(lat=(data.lat.data * ureg.radians).to("degrees"))

    # Make a coordinate system for the receive points
    glat, glon, gh = xr.broadcast(data.lat, data.lon, data.h)
    glat.data = glat.data * ureg.degree
    glon.data = glon.data * ureg.degree
    gh.data = gh.data * ureg.m

    # Create HCS for RX
    rxcs = HCS.from_crs([glat, glon, gh])

    # Get path loss
    prop_loss = getattr(propagators, propagation.lower())(
        txcs=tx.hcs,
        rxcs=rxcs,
        frequency=data.frequency * ureg.Hz,
    )
    prop_loss.data = prop_loss.data.to_base_units()

    # Mask Line-of-sight
    if mask_los:
        prop_loss = propagators.mask_los(prop_loss, tx.hcs, rxcs)

    # Integrate the propagation loss to the transmit power
    data = data / np.sqrt(prop_loss)

    # Convert the sqrt(power) to a power
    data.data = abs(data).data ** 2

    # Convert to dBm
    data.data = data.data.to("dBm")

    # Attributes that xarray uses for plotting (long_name and units)
    attrs = dict(long_name="Power", units="dBm", description="Transmit power density")

    # Add attrs
    data.attrs = {**data.attrs, **attrs}

    return data


def plot_transmit_pd(
    data,
    polarization="apolar",
    units="dBm",
    extra_extents_deg=[0.05, 0.05],
    vspan=60,
    animate=False,
):
    # Convert data to dBm
    data = data.copy()
    data.data = data.data.to(units)

    # Drop vars so title is just time which is moving
    data = data.sel(polarization=polarization)
    data = data.squeeze()

    # Convert quantities
    if isinstance(data.lon.data, Quantity):
        data.lon.data = data.lon.data.to("degree").magnitude
    if isinstance(data.lat.data, Quantity):
        data.lat.data = data.lat.data.to("degree").magnitude

    # This is the map projection we want to plot *onto*
    map_proj = ccrs.LambertConformal(
        central_longitude=data.lon.data.mean(),
        central_latitude=data.lat.data.mean(),
    )

    osm_img = cimgt.GoogleTiles(style="street")

    # Create figure
    fig = plt.figure()  # open matplotlib figure
    ax = plt.axes(
        projection=osm_img.crs,
    )  # project using coordinate reference system (CRS) of street map

    extent = np.array(
        [
            data.lon.min() - extra_extents_deg[0],
            data.lon.max() + extra_extents_deg[0],
            data.lat.min() - extra_extents_deg[1],
            data.lat.max() + extra_extents_deg[1],
        ],
    )

    if len(data.shape) > 2 and animate:
        # Initialize plot
        data.isel({animate: 0}).plot(
            transform=ccrs.PlateCarree(),  # the data's projection
            subplot_kws={"projection": map_proj},  # the plot's projection
            alpha=0.75,
            vmin=round2base(data.max().data.magnitude) - vspan,
            vmax=round2base(data.max().data.magnitude),
        )

        _setup_map_axes(ax, osm_img, extent)

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
                vmin=round2base(data.max().data.magnitude) - vspan,
                vmax=round2base(data.max().data.magnitude),
            )
            _setup_map_axes(ax, osm_img, extent)

            return (ax.collections[0],)

        ani = animation.FuncAnimation(
            plt.gcf(),
            antenna_movie,
            save_count=data.coords[animate].shape[0],
            interval=10,
            blit=True,
        )

        return ani
    # Plot DataArray
    data.plot(
        transform=ccrs.PlateCarree(),  # the data's projection
        subplot_kws={"projection": map_proj},  # the plot's projection
        alpha=0.75,
        vmin=round2base(data.max().data.magnitude) - vspan,
        vmax=round2base(data.max().data.magnitude),
    )

    extent = np.array(
        [
            data.lon.min() - extra_extents_deg[0],
            data.lon.max() + extra_extents_deg[0],
            data.lat.min() - extra_extents_deg[1],
            data.lat.max() + extra_extents_deg[1],
        ],
    )
    _setup_map_axes(ax, osm_img, extent)

    return ax


def plot_link(
    data: xr.DataArray,
    units: str = "dBm",
    extra_extents_deg: tuple = (0.05, 0.05),
    vminmax: tuple | None = None,
    vspan: float = 60,
    ax: plt.Axes | None = None,
    colorbar: bool = True,
    cmap: matplotlib.colors.Colormap = plt.cm.viridis,
    **kwargs,
) -> plt.Axes:
    """
    Plot the result of calculate_spatial_link on a map projection.

    Parameters
    ----------
    data : xr.DataArray
        Data to be plotted as returned from the calculate_spatial_link method
    units : str
        Units to plot the data in
    extra_extents_deg : tuple
        How much to expand the plot limits beyond the data in (longitude, latitude)
    vminmax
    """
    # Convert data to dBm
    data = data.copy()
    data.data = data.data.to(units)

    # Drop vars so title is just time which is moving
    data = data.squeeze()

    # Convert quantities
    if isinstance(data.lon.data, Quantity):
        data.lon.data = data.lon.data.to("degree").magnitude
    if isinstance(data.lat.data, Quantity):
        data.lat.data = data.lat.data.to("degree").magnitude

    # This is the map projection we want to plot *onto*
    osm_img = cimgt.GoogleTiles(style="street")

    if ax is None:
        fig = plt.figure()  # open matplotlib figure
        ax = plt.axes(
            projection=osm_img.crs,
        )  # project using coordinate reference system (CRS) of street map
    else:
        fig = plt.gcf()
    if vminmax is None:
        vmin = round2base(data.max().data) - vspan
        vmax = round2base(data.max().data)
    else:
        vmin = vminmax[0]
        vmax = vminmax[1]

    p = ax.scatter(
        data.lon.data,
        data.lat.data,
        transform=ccrs.PlateCarree(),
        c=data.data,
        vmin=vmin,
        vmax=vmax,
        s=15,
        cmap=cmap,
        **kwargs,
    )

    extent = np.array(
        [
            data.lon.min() - extra_extents_deg[0],
            data.lon.max() + extra_extents_deg[0],
            data.lat.min() - extra_extents_deg[1],
            data.lat.max() + extra_extents_deg[1],
        ],
    )

    _setup_map_axes(ax, osm_img, extent)

    # Add in colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad="1.5%", axes_class=plt.Axes)

        fig.add_axes(ax_cb)
        cb = plt.colorbar(p, cax=ax_cb)

        cb.set_label(f"{data.attrs['long_name']} [{data.attrs['units']}]")

    return ax


# plot a scale bar with 4 subdivisions on the left side of the map
def scale_bar_left(
    ax,
    bars=4,
    length=None,
    location=(0.1, 0.05),
    linewidth=3,
    col="black",
    **kwargs,
):
    """
    Taken from: https://github.com/SciTools/cartopy/issues/490#issuecomment-376520100
    ax is the axes to draw the scalebar on.
    bars is the number of subdivisions of the bar (black and white chunks)
    length is the length of the scalebar in km.
    location is left side of the scalebar in axis coordinates.
    (ie. 0 is the left side of the plot)
    linewidth is the thickness of the scalebar.
    color is the color of the scale bar
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    # Make tmc aligned to the left of the map,
    # vertically at scale bar location
    sbllx = llx0 + (llx1 - llx0) * location[0]
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ["1", "2", "5"]:
                return int(x)
            return scale_number(x - 10**ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000 / bars]
    # Plot the scalebar chunks
    barcol = "white"
    for i in range(bars):
        # plot the chunk
        ax.plot(bar_xs, [sby, sby], transform=tmc, color=barcol, linewidth=linewidth)
        # alternate the color
        if barcol == "white":
            barcol = "dimgrey"
        else:
            barcol = "white"
        # Generate the x coordinate for the number
        bar_xt = sbx + i * length * 1000 / bars
        # Plot the scalebar label for that chunk
        ax.text(
            bar_xt,
            sby,
            str(i * length / bars),
            transform=tmc,
            horizontalalignment="center",
            verticalalignment="bottom",
            color=col,
            **kwargs,
        )
        # work out the position of the next chunk of the bar
        bar_xs[0] = bar_xs[1]
        bar_xs[1] = bar_xs[1] + length * 1000 / bars
    # Generate the x coordinate for the last number
    bar_xt = sbx + length * 1000
    # Plot the last scalebar label
    ax.text(
        bar_xt,
        sby,
        str(round(length)),
        transform=tmc,
        horizontalalignment="center",
        verticalalignment="bottom",
        color=col,
        **kwargs,
    )
    # Plot the unit label below the bar
    bar_xt = sbx + length * 1000 / 2
    bar_yt = y0 + (y1 - y0) * (location[1] / 4)
    ax.text(
        bar_xt,
        bar_yt,
        "km",
        transform=tmc,
        horizontalalignment="center",
        verticalalignment="bottom",
        color=col,
        **kwargs,
    )


def scatter_hist(
    data,
    xdim,
    vmin=-200,
    vmax=-50,
    xunit="km",
    title="",
    xhist: bool = True,
    yhist: bool = True,
):
    if isinstance(data, list):
        x = []
        y = []
        for d in data:
            if isinstance(d.coords[xdim].data, Quantity):
                x.append(d.coords[xdim].data.to(xunit).magnitude.ravel())
            else:
                x.append(d.coords[xdim].data.ravel())
            y.append(d.values.ravel())
        x = np.hstack(x)
        y = np.hstack(y)
    else:
        if isinstance(data.coords[xdim].data, Quantity):
            x = data.coords[xdim].data.to(xunit).magnitude.ravel()
        else:
            x = data.coords[xdim].data.ravel()
        y = data.values.ravel()

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure()

    ax = fig.add_axes(rect_scatter)
    if xhist:
        ax_histx = fig.add_axes(rect_histx, sharex=ax)
        ax_histx.tick_params(axis="x", labelbottom=False)
    if yhist:
        ax_histy = fig.add_axes(rect_histy, sharey=ax)
        ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(
        x,
        y,
        alpha=0.2,
        c=y,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylim(vmin, vmax)

    # Add histograms
    if xhist:
        ax_histx.hist(x, density=True)
    if yhist:
        ax_histy.hist(y, orientation="horizontal", density=True)

    if xunit == "":
        xunit_label = xunit
    else:
        xunit_label = f" [{xunit}]"
    ax.set_xlabel(f"{xdim[0].upper()}{xdim[1:]}{xunit_label}")
    if isinstance(data, list):
        ax.set_ylabel(f"{data[0].attrs['long_name']} [{data[0].attrs['units']}]")
    elif isinstance(data, xr.DataArray):
        ax.set_ylabel(f"{data.attrs['long_name']} [{data.attrs['units']}]")

    # Add title
    if xhist:
        ax_histx.set_title(title)
    else:
        ax.set_title(title)


def view_link_horizon(
    tx,
    rx,
    res,
    incident_pol,
    prop_loss,
    aspect: float = 1.0,
    limit_scale: float = 1.0,
    bbox_props: dict | None = None,
    fontsize=8,
    gsize=0.15,
    cs_kwargs: dict | None = None,
    data_kwargs: dict | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Creates line from transmitter and receiver and adds it to surface profile.

    Parameters
    ---------
    ax : Axes
        Axes to add lines to
    res : xr.DataArray
        Result returned from propagation method
    limit_scale : float
        How much to scale y-axis from what view_surface_profile returns
    """
    # Initialize optional kwargs
    if bbox_props is None:
        bbox_props = dict()
    if cs_kwargs is None:
        cs_kwargs = dict()
    if data_kwargs is None:
        data_kwargs = dict()

    # Create Surface profile
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(fig.get_figwidth() * 2)
    if "clutter_method" in prop_loss.attrs.keys():
        if "clutter_kwargs" in prop_loss.attrs.keys():
            lc_skip_ind = prop_loss.attrs["clutter_kwargs"].get("lc_skip_ind", None)
            lc = True
            lcidx = 0
    else:
        lc_skip_ind = None
        lc = False
        lcidx = -1
    ax = view_surface_profile(
        tx.hcs,
        rx.hcs,
        aspect=aspect,
        ax=ax,
        lc=lc,
        lc_skip_ind=lc_skip_ind,
        **cs_kwargs,
    )

    # Slice
    res = res.isel(**{k: v for k, v in data_kwargs.items() if k in res.dims})
    prop_loss = prop_loss.isel(**{k: v for k, v in data_kwargs.items() if k in prop_loss.dims})
    incident_pol = incident_pol.isel(
        **{k: v for k, v in data_kwargs.items() if k in incident_pol.dims},
    )

    # Store axes limit
    ylim = ax.get_ylim()

    # Create x-sampling
    xs = np.linspace(*ax.get_xlim(), 1001)

    # Y data for tx is x*tan(tx_angle) + h_tx
    yta = ax.get_lines()[lcidx + 2].get_ydata().item()
    xta = ax.get_lines()[lcidx + 2].get_xdata().item()
    ytx = xs * np.tan(res.tx_angle.item()) + yta
    # Plot tx horizon line
    lw = 0.8
    ax.plot(xs, ytx, ls="dashdot", lw=lw, color="k", label="TX Horizon")

    # Y data for rx is -x * tan(tx_angle) + (h_rx + dist * tan(rx_angle))
    yra = ax.get_lines()[lcidx + 3].get_ydata().item()
    xra = ax.get_lines()[lcidx + 3].get_xdata().item()
    yrx = -xs * np.tan(res.rx_angle.item()) + yra + xra * np.tan(res.rx_angle.item())
    # Plot rx horizon line
    ax.plot(xs, yrx, ls="dashdot", lw=lw, color="k", label="RX Horizon")

    # Plot LOS and First Fresnel Zone and 60% as dashed
    sl_m = (yra - yta) / (xra - xta)
    los_ang = np.arctan(sl_m)
    sl_b = yta - sl_m * xta
    sl_x = np.linspace(xta, xra, 1001)
    sl = sl_m * sl_x + sl_b
    sl_dist = np.sqrt((sl_x - sl_x[0]) ** 2 + (sl - yta) ** 2)
    wavelength = (ureg.speed_of_light / (res.frequency.item() * ureg.Hz)).to("m").magnitude
    los_line = LineString([(x, y) for x, y in zip(sl_x, sl, strict=False)])
    surface_line = LineString(
        [
            (x, y)
            for x, y in zip(
                ax.get_lines()[lcidx + 1].get_xdata(),
                ax.get_lines()[lcidx + 1].get_ydata(),
                strict=False,
            )
        ],
    )
    los_c = "k"
    if los_line.intersects(surface_line):
        los_c = "r"
    ax.plot(sl_x, sl, f"{los_c}-", lw=lw)
    for fscl, fls in zip([1, 0.6], ["-", "--"], strict=False):
        fresnel_radius = fscl * np.sqrt(2 * wavelength * sl_dist * (1 - (sl_dist / sl_dist[-1])))
        fresnel_radius_x = sl_x * np.cos(los_ang) - fresnel_radius * np.sin(los_ang)
        fresnel_radius_py = sl_x * np.sin(los_ang) + fresnel_radius * np.cos(los_ang) + sl[0]
        fresnel_radius_ny = sl_x * np.sin(los_ang) - fresnel_radius * np.cos(los_ang) + sl[0]
        fresnel_line = LineString(
            [(x, y) for x, y in zip(fresnel_radius_x, fresnel_radius_ny, strict=False)],
        )

        frc = "C7"
        if fresnel_line.intersects(surface_line):
            frc = "C1"
        ax.plot(fresnel_radius_x, fresnel_radius_py, f"{frc}{fls}", lw=lw)
        ax.plot(fresnel_radius_x, fresnel_radius_ny, f"{frc}{fls}", lw=lw)

    # Set limit
    ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * limit_scale)

    # Create new coordinate systems with zenith being normal
    txcs0 = HCS.from_crs([llh.isel(**cs_kwargs) for llh in tx.hcs.llh], hagl=False)
    rxcs0 = HCS.from_crs([llh for llh in rx.hcs.llh], hagl=False)

    # Get relative azimuths
    azs = relative_azimuth(tx.hcs, rx.hcs)
    azs = [az.isel(**cs_kwargs) for az in azs]
    # Rotate coordinate systems
    txcs = HCS(
        (0, 0, 0) * ureg.m,
        rotation=Rotation.from_euler("Z", azs[0].item().magnitude - 90, degrees=True),
        reference=txcs0,
    )
    rxcs = HCS(
        (0, 0, 0) * ureg.m,
        rotation=Rotation.from_euler("Z", azs[1].item().magnitude - 90, degrees=True),
        reference=rxcs0,
    )

    # Get TX gain
    thslice = np.arange(-180, 180, 0.5) * ureg.degree
    datatx = tx.request_data(
        theta=thslice,
        phi=0 * ureg.degree,
        hcs=txcs,
    )
    datatx = abs(
        datatx.isel(**{k: v for k, v in data_kwargs.items() if k in datatx.dims})
        .sel(polarization="apolar")
        .squeeze(),
    )

    # Get RX gain
    datarx = rx.request_data(theta=thslice, phi=0 * ureg.degree, hcs=rxcs)
    rdata_kwargs = {k: v for k, v in data_kwargs.items() if k in datarx.dims}
    datarx = datarx.isel(**rdata_kwargs)
    totrx = abs(datarx.sel(polarization="apolar").squeeze())

    # Get copol gain for rx if polarization present default to apolar if not present
    if set(["theta", "phi"]).issubset(set(datarx.polarization.values)):
        copolrx = datarx.sel(polarization=["theta", "phi"])
        copolrx = copolrx.assign_coords(polarization=["vertical", "horizontal"])

        # Compute RX gain in terms of incident polarization
        copolrx = abs((copolrx * incident_pol).sum(dim="polarization").squeeze())

    else:
        copolrx = totrx
    # Scaled gain for plotting
    gscale = (
        res.distance.item().magnitude * gsize / np.max((copolrx.max().item(), datatx.max().item()))
    )
    rrx = copolrx * gscale
    rtrx = totrx * gscale
    rtx = datatx * gscale
    yscale = 1

    # Make TX pattern
    gtx = rtx * np.sin(rtx.theta) + ax.get_lines()[lcidx + 2].get_xdata()
    gty = rtx * np.cos(rtx.theta) * yscale + ax.get_lines()[lcidx + 2].get_ydata()
    ax.plot(gtx, gty, color="C0", lw=0.8, label="TX Gain")
    ax.fill(
        gtx,
        gty,
        facecolor="C0",
        edgecolor="none",
        alpha=0.3,
        linewidth=0.8,
    )

    # Make Pattern Note that Rx points in negative x
    gtrx = rtrx * np.sin(-rtrx.theta) + ax.get_lines()[lcidx + 3].get_xdata()
    gtry = rtrx * np.cos(-rtrx.theta) * yscale + ax.get_lines()[lcidx + 3].get_ydata()
    ax.plot(
        gtrx,
        gtry,
        color="C0",
        lw=0.8,
        ls=":",
        label="RX Gain",
    )
    ax.fill(
        gtrx,
        gtry,
        facecolor="C0",
        edgecolor="none",
        alpha=0.1,
        linewidth=0.8,
    )

    grx = -rrx * np.sin(rrx.theta) + ax.get_lines()[lcidx + 3].get_xdata()
    gry = rrx * np.cos(rrx.theta) * yscale + ax.get_lines()[lcidx + 3].get_ydata()
    ax.plot(grx, gry, color="C0", lw=0.8, label="RX Co-Pol Gain")
    ax.fill(
        grx,
        gry,
        facecolor="C0",
        edgecolor="none",
        alpha=0.2,
        linewidth=0.8,
    )

    # Maybe or just annotate
    props = dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8,
    )
    bbox_props = {**props, **bbox_props}

    # Convert to dB
    tx_gain = res.tx_gain
    tx_gain.data = tx_gain.data.to("dB")
    rx_gain = res.rx_gain
    rx_gain.data = rx_gain.data.to("dB")
    prop_loss.data = prop_loss.data.to("dB")

    # place a text box in upper left in axes coords
    yspan = np.diff(ax.get_ylim()).squeeze()
    mid = np.mean(ax.get_ylim()).squeeze()
    if ax.get_lines()[lcidx + 2].get_ydata() >= mid:
        ypos = gty.min() - yspan * 0.1
    else:
        ypos = gty.max() + yspan * 0.1

    ax.text(
        ax.get_lines()[lcidx + 2].get_xdata().item(),
        ypos,
        f"TX Power: {res.tx_power.item().to('dBm').magnitude:.2f} dBm\nTX Gain: {tx_gain.item().magnitude:.2f} dB",
        transform=ax.transData,
        fontsize=fontsize,
        verticalalignment="center",
        bbox=bbox_props,
    )
    if ax.get_lines()[lcidx + 3].get_ydata() >= mid:
        ypos = gtry.min() - yspan * 0.1
    else:
        ypos = gtry.max() + yspan * 0.1
    ax.text(
        ax.get_lines()[lcidx + 3].get_xdata().item(),
        ypos,
        f"RX Gain: {rx_gain.item().magnitude:.2f} dB\nRX Power: {res.item().to('dBm').magnitude:.2f} dBm",
        transform=ax.transData,
        fontsize=fontsize,
        verticalalignment="center",
        horizontalalignment="right",
        bbox=bbox_props,
    )
    # Path Loss
    # intersection point
    yi = ytx[abs(yrx - ytx).argmin()]
    # Limit intersection point if outside of plot bounds
    if yi > ax.get_ylim()[1]:
        yl = ax.get_ylim()
        yi = yl[1] - 0.1 * np.diff(yl)
    if "polarization" in prop_loss.dims:
        ax.text(
            0.5,
            0.8,
            f"Total Path Loss: {prop_loss.sel(polarization='total').item().magnitude:.2f} dB\nH-pol Path Loss: {prop_loss.sel(polarization='horizontal').item().magnitude:.2f} dB\nV-pol Path Loss: {prop_loss.sel(polarization='vertical').item().magnitude:.2f} dB",
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=bbox_props,
        )
    else:
        ax.text(
            0.5,
            0.8,
            f"Total Path Loss: {prop_loss.item().magnitude:.2f} dB",
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="center",
            horizontalalignment="center",
            bbox=bbox_props,
        )

    # Add a Title
    ax.set_title("Link Detail")

    return ax

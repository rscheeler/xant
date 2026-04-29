from functools import partial

import geopandas as gpd
import h3
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from xrench.units import ureg


def ll2cell(row: gpd.GeoSeries, resolution: int = 5) -> str:
    """
    Take in GeoPandas row and determine H3 cell index from geometry at a specified resolution.
    """
    return h3.latlng_to_cell(row.geometry.y, row.geometry.x, resolution)


def cell2boundary(row: gpd.GeoSeries) -> Polygon:
    """
    Returns a shapely polygon from the h3 index.
    """
    points = h3.cell_to_boundary(row["h3"])
    points = [p[::-1] for p in points]
    return Polygon(points)


def ll2h3(df: gpd.GeoDataFrame, resolution: int) -> gpd.GeoDataFrame:
    """
    Add the h3 index and polygon as series to the df GeoDataFrame.
    """
    df["h3"] = df.apply(partial(ll2cell, resolution=resolution), axis=1)
    df["geometry"] = df.apply(cell2boundary, axis=1)
    return df


def get_databounds(df: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    """
    Find minimum, maximum, mean, and count of data points that fall within the same h3 polygon.
    """
    # Group by the h3 column and aggregate the gain column
    dmean = df.groupby(["h3"])[col].agg("mean")
    dmin = df.groupby(["h3"])[col].agg("min")
    dmax = df.groupby(["h3"])[col].agg("max")
    count = df.groupby(["h3"])[col].agg("count")

    # Assumes data is power and needs to be converted back to dB
    crit = gpd.pd.concat(
        [
            10 * np.log10(dmean).to_frame("mean"),
            10 * np.log10(dmin).to_frame("min"),
            10 * np.log10(dmax).to_frame("max"),
            count.to_frame("count"),
        ],
        axis=1,
    ).reset_index()
    crit = gpd.GeoDataFrame(crit, geometry=crit.apply(cell2boundary, axis=1), crs="EPSG:4326")
    return crit


def add_color(row, color_col: str = None, norm=None, cmap=None) -> str:
    """
    Return the hex color string for each row given the norm and color map.
    """
    return colors.rgb2hex(cmap(norm(row[color_col])))


def ll2h3gpd(
    h3data: xr.DataArray,
    resolution: int = 5,
    color_col: str = "mean",
    cmap=plt.get_cmap("viridis"),
    zspan: float = 40,
) -> gpd.GeoDataFrame:
    """
    Convert antenna data (in dimensionless power form) sampled in latitude and longitude to h3 cells
    of the specified resolution.
    """
    # Convert lat/lon to degrees
    h3data = h3data.assign_coords(
        dict(
            lat=(h3data.lat * ureg.radian).data.to("degree"),
            lon=(h3data.lon * ureg.radian).data.to("degree"),
        ),
    )

    # Convert to dataframe
    df = h3data.to_dataframe().reset_index()

    # Create geopandas dataframe
    gdf = gpd.GeoDataFrame(df[h3data.name], geometry=gpd.points_from_xy(df.lon, df.lat))

    # Add in H3 geometry
    gdf = ll2h3(gdf, resolution)

    # Get min,max,mean, and count dataframe
    gbounds = get_databounds(gdf, h3data.name)

    # Add color
    # Normalize the data to the range [0, 1] for color mapping
    norm = plt.Normalize(vmin=gbounds[color_col].max() - zspan, vmax=gbounds[color_col].max())
    # Get colors for each value in the column
    gbounds["color"] = gbounds.apply(
        partial(add_color, color_col=color_col, norm=norm, cmap=cmap),
        axis=1,
    )
    return gbounds


def h3xr2gpd(
    h3xrdata: xr.DataArray,
    cmap=plt.get_cmap("viridis"),
    zspan: float = 40,
) -> gpd.GeoDataFrame:
    """
    Convert antenna data (in dimensionless power form) to GeoPandas with H3 polygons.
    """
    # Convert to dataframe
    df = h3xrdata.to_dataframe().reset_index()

    # Create geopandas dataframe
    gdf = gpd.GeoDataFrame(df)

    # Add in H3 polygon geometry
    gdf["geometry"] = gdf.apply(cell2boundary, axis=1)
    gdf = gdf.set_crs("EPSG:4326")

    # Add color
    # Normalize the data to the range [0, 1] for color mapping
    norm = plt.Normalize(vmin=gdf[h3xrdata.name].max() - zspan, vmax=gdf[h3xrdata.name].max())
    # Get colors for each value in the column
    gdf["color"] = gdf.apply(
        partial(add_color, color_col=h3xrdata.name, norm=norm, cmap=cmap),
        axis=1,
    )
    return gdf

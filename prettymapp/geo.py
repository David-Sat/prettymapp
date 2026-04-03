from typing import Tuple, Optional

from osmnx.geocoder import geocode
from geopandas import GeoDataFrame
import pandas as pd
from pandas import DataFrame
from shapely.geometry import Polygon, Point, box


class GeoCodingError(Exception):
    pass


def validate_coordinates(lat: float, lon: float) -> None:
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise ValueError(
            "latitude (-90 to 90) and longitude (-180 to 180) coordinates "
            "are not within valid ranges."
        )


def get_aoi(
    address: Optional[str] = None,
    coordinates: Optional[Tuple[float, float]] = None,
    radius: int = 1000,
    rectangular: bool = False,
    aspect_ratio: Optional[Tuple[float, float]] = None,
) -> Polygon:
    """
    Gets round or rectangular shapely Polygon in in 4326 from input address or coordinates.

    Args:
        address: Address string
        coordinates: lat, lon
        radius: Radius in meter
        rectangular: Optionally return aoi as rectangular polygon, default False.
        aspect_ratio: Optional width, height ratio for rectangular AOIs. Defaults to
            a square (1, 1) when `rectangular=True`.

    Returns:
        shapely Polygon in 4326 crs
    """
    if address is not None:
        if coordinates is not None:
            raise ValueError(
                "Both address and latlon coordinates were provided, please "
                "select only one!"
            )
        try:
            lat, lon = geocode(address)
        except ValueError as e:
            raise GeoCodingError(f"Could not geocode address '{address}'") from e
    else:
        if coordinates is None:
            raise ValueError("Either 'address' or 'coordinates' must be provided.")
        lat, lon = coordinates
    validate_coordinates(lat, lon)

    df = GeoDataFrame(
        DataFrame([0], columns=["id"]), crs="EPSG:4326", geometry=[Point(lon, lat)]
    )
    df = df.to_crs(df.estimate_utm_crs())
    point = df.iloc[0].geometry

    if rectangular:
        square_aoi = _get_square_aoi(df, point, radius)
        if aspect_ratio is None or aspect_ratio == (1, 1):
            return square_aoi
        return _get_rectangular_aoi(square_aoi, aspect_ratio)
    else:
        poly = point.buffer(radius)

    df.geometry = [poly]
    df = df.to_crs(crs=4326)
    poly = df.iloc[0].geometry

    return poly


def _get_square_aoi(df: GeoDataFrame, center: Point, radius: int) -> Polygon:
    poly = center.buffer(radius)
    df.geometry = [poly]
    df = df.to_crs(crs=4326)
    return box(*df.iloc[0].geometry.bounds)


def _get_rectangular_aoi(square_aoi: Polygon, aspect_ratio: Tuple[float, float]) -> Polygon:
    width_ratio, height_ratio = aspect_ratio
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError("aspect_ratio values must be larger than 0.")

    minx, miny, maxx, maxy = square_aoi.bounds
    width = maxx - minx
    height = maxy - miny
    xmid = (minx + maxx) / 2
    ymid = (miny + maxy) / 2

    if width_ratio >= height_ratio:
        width *= width_ratio / height_ratio
    else:
        height *= height_ratio / width_ratio

    return box(
        xmid - width / 2,
        ymid - height / 2,
        xmid + width / 2,
        ymid + height / 2,
    )


def explode_multigeometries(df: GeoDataFrame) -> GeoDataFrame:
    """
    Explode all multi geometries in a geodataframe into individual polygon geometries.
    Adds exploded polygons as rows at the end of the geodataframe and resets its index.
    Args:
        df: Input GeoDataFrame
    """
    mask = df.geom_type.isin(["MultiPolygon", "MultiLineString", "MultiPoint"])
    outdf = df[~mask]
    df_multi = df[mask]
    for _, row in df_multi.iterrows():
        df_temp = GeoDataFrame(
            pd.DataFrame.from_records([row.to_dict()] * len(row.geometry.geoms)),
            crs="EPSG:4326",
        )
        df_temp.geometry = list(row.geometry.geoms)
        outdf = GeoDataFrame(
            pd.concat([outdf, df_temp], ignore_index=True), crs="EPSG:4326"
        )

    outdf = outdf.reset_index(drop=True)
    return outdf

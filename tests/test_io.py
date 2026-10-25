"""Tests for madrid_safety.io (geometry loading helpers)."""
import geopandas as gpd
from shapely.geometry import Point, Polygon

from madrid_safety import io as ms_io


def test_detect_crs_utm():
    """Madrid UTM bounds should be detected as EPSG:25830."""
    g = Point(440000, 4475000)
    assert ms_io.detect_crs_from_bounds([g]) == "EPSG:25830"


def test_detect_crs_wgs84():
    """WGS84 (lon/lat) bounds should be detected as EPSG:4326."""
    g = Point(-3.7, 40.4)
    assert ms_io.detect_crs_from_bounds([g]) == "EPSG:4326"


def test_detect_crs_empty_defaults_4326():
    """Empty iterable defaults to EPSG:4326."""
    assert ms_io.detect_crs_from_bounds([]) == "EPSG:4326"


def test_repair_geometries_fixes_bowtie():
    """A self-intersecting bowtie polygon is repaired to a valid geometry."""
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert not invalid.is_valid
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[invalid],
                            crs="EPSG:25830")
    out = ms_io.repair_geometries(gdf, label="test")
    assert len(out) == 1
    assert out.geometry.is_valid.all()

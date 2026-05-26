"""Search tools used by the agent.

Только `search_places` реализован как рабочий пример. Остальные функции —
заглушки, которые надо доработать.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from lib.data_types import DemandPoint, Place

from .geo_utils import haversine_km_array

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_places(csv_path: str | None = None) -> pd.DataFrame:
    """Load amenities table.

    Порядок поиска:
        1. явный `csv_path`,
        2. `data/processed/*amenities*.csv`,
        3. `data/raw/*amenities*.csv`,
        4. `data/sample_places.csv` — встроенный safety-net датасет.

    """
    if csv_path is not None:
        return pd.read_csv(csv_path)

    for sub in ("processed", "raw"):
        for p in sorted((DATA_DIR / sub).glob("*amenities*.csv")):
            return pd.read_csv(p)

    sample = DATA_DIR / "sample_places.csv"
    if sample.exists():
        return pd.read_csv(sample)

    raise FileNotFoundError(
        "No places CSV found. Run `python scripts/download_osm.py --city ...` "
        "or `python scripts/generate_sample_data.py`."
    )


def _load_demand(csv_path: str | None = None) -> pd.DataFrame:
    """Load demand table (residential buildings with population counts).

    Порядок поиска:
        1. явный `csv_path`,
        2. `data/processed/*demand*.csv`,
        3. `data/raw/*demand*.csv`.
    Возвращает только колонки lat, lon, count_people.
    """
    if csv_path is not None:
        path = Path(csv_path)
    else:
        path = None
        for sub in ("processed", "raw"):
            for p in sorted((DATA_DIR / sub).glob("*demand*.csv")):
                path = p
                break
            if path is not None:
                break

    if path is None:
        raise FileNotFoundError(
            "No demand CSV found. Expected a file matching '*demand*.csv' in "
            "data/processed/ or data/raw/."
        )

    df = pd.read_csv(path, usecols=["lat", "lon", "count_people"])
    return df.dropna(subset=["lat", "lon", "count_people"])


NAME_SCORE_CUTOFF = 83


def search_places(
    category: list[str] | None = None,
    name: str | None = None,
    near: tuple[float, float] | None = None,
    max_distance_km: float | None = None,
    limit: int | None = 20,
    csv_path: str | None = None,
) -> list[Place]:
    """Return amenities matching the filters, optionally sorted by distance.

    Args:
        list of categories: e.g. "cafe", "pharmacy". None = any.
        name: name of an amenity
        near: (lat, lon) anchor point for distance filtering/sorting.
        max_distance_km: keep only places within this radius from `near`.
        limit: max number of results. None = all results.
    """
    df = _load_places(csv_path)

    if category is not None and "amenity" in df.columns:
        df = df[df["amenity"].isin(category)]

    if name is not None and "name" in df.columns:
        df = df.reset_index(drop=True)
        names = df["name"].astype(str).tolist()
        matches = process.extract(
            name, names,
            scorer=fuzz.WRatio,
            limit=None,
            score_cutoff=NAME_SCORE_CUTOFF,
        )
        keep_positions = [pos for _, _, pos in matches]
        df = df.iloc[keep_positions]

    if near is not None and {"lat", "lon"}.issubset(df.columns):
        lat0, lon0 = near
        df = df.assign(
            distance_km=haversine_km_array(
                lat0, lon0, df["lat"].to_numpy(), df["lon"].to_numpy()
            )
        )
        if max_distance_km is not None:
            df = df[df["distance_km"] <= max_distance_km]
        if limit is None:
            df = df.sort_values("distance_km")
        else:
            df = df.nsmallest(limit, "distance_km")
    else:
        df = df.assign(distance_km=None)

    slice_df = (df if limit is None else df.head(limit)).copy()
    slice_df = slice_df.astype(object).where(pd.notna(slice_df), None)

    return [Place(**rec) for rec in slice_df.to_dict(orient="records")]


def nearest_places(
    point: tuple[float, float],
    category: list[str] | None = None,
    limit: int = 5,
) -> list[Place]:
    """Return the `limit` nearest places to `point`."""
    return search_places(category=category, near=point, limit=limit)


def search_by_name(
    name: str,
    point: tuple[float, float] | None,
) -> list[Place]:
    """Return sorted places which match to the `name`."""
    return search_places(near=point, name=name, limit=None)


def nearest_demand(
    point: tuple[float, float],
    limit: int = 50,
) -> list[DemandPoint]:
    """Return the `limit` nearest demand points to `point`."""
    lat0, lon0 = point
    df = _load_demand()
    df = df.assign(
        distance_km=haversine_km_array(
            lat0, lon0, df["lat"].to_numpy(), df["lon"].to_numpy()
        )
    )
    df = df.nsmallest(limit, "distance_km").astype(object).where(pd.notna(df), None)
    return [DemandPoint(**rec) for rec in df.to_dict(orient="records")]

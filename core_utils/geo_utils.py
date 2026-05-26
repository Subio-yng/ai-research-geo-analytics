"""Geospatial helpers: distances, heatmaps, simple aggregations."""
from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from typing import Any, Literal

import networkx as nx
import numpy as np
import osmnx as ox
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from geopy.geocoders import Nominatim

ox.settings.use_cache = True

EARTH_R = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_R * asin(sqrt(a))


def haversine_km_array(lat0: float, lon0: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorized great-circle distance from one anchor (lat0, lon0)
    to arrays of points. Mirrors haversine_km but operates on numpy arrays.
    """
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat - lat0
    dlon = lon - lon0
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(a))


def haversine_km_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def compute_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Alias of haversine_km, for readability in tools/agent."""
    return haversine_km(p1[0], p1[1], p2[0], p2[1])


def _coerce_points(raw: Any) -> list[tuple]:
    """LLM шлёт points по-разному: list[list], list[dict], JSON-строкой.

    Нормализуем в list[(lat, lon, weight)]
    """
    if raw is None:
        raise ValueError("heatmap: 'points' is required")
    if isinstance(raw, str):
        import json
        raw = json.loads(raw)
    pts: list[tuple] = []
    for item in raw:
        if isinstance(item, dict):
            lat, lon = item["lat"], item["lon"]
            w = item.get("weight")
            pts.append((lat, lon, w) if w is not None else (lat, lon))
        else:
            pts.append(tuple(item))
    return pts


def build_heatmap(points, **kwargs):  # noqa: ANN001 - returns folium.Map
    """Build a folium HeatMap from a list of (lat, lon) or (lat, lon, weight) points.

    TODO: расширить — радиус.
    """
    import folium
    from folium.plugins import HeatMap

    if not points:
        raise ValueError("No points to plot")
    center = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )
    m = folium.Map(location=center, zoom_start=kwargs.get("zoom_start", 13))

    weighted = [
        (p[0], p[1], p[2] if len(p) > 2 else 1.0)
        for p in points
    ]

    # TODO: debug
    if kwargs.get("legend", False):
        title = kwargs.get("legend_title", "Интенсивность")
        low = kwargs.get("legend_low", "низкая")
        high = kwargs.get("legend_high", "высокая")
        legend_html = f"""
        <div style="
            position: fixed; bottom: 30px; left: 30px; z-index: 9999;
            background: white; padding: 10px 14px; border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3); font: 13px sans-serif;">
          <b>{title}</b><br>
          <div style="
              width: 160px; height: 12px; margin: 6px 0;
              background: linear-gradient(to right, blue, lime, yellow, red);">
          </div>
          <span style="float:left;">{low}</span>
          <span style="float:right;">{high}</span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    HeatMap(weighted, radius=kwargs.get("radius", 32)).add_to(m)
    return m


Mode = Literal["walk", "drive", "bike"]
DEFAULT_SPEEDS: dict[str, float] = {"walk": 4.75, "bike": 10.0}


def _graph_for_points(
    points: list[tuple[float, float]], mode: Mode
) -> tuple[nx.MultiDiGraph, list[int]]:
    """Скачать граф, покрывающий все точки + запас, и привязать точки к узлам.

    Возвращает (граф, список узлов-перекрёстков по одному на каждую точку
    в том же порядке). Единый источник для isochrone и route_length.
    """
    if len(points) < 1:
        raise ValueError("Нужна хотя бы одна точка")

    lats = [p[0] for p in points]
    lons = [p[1] for p in points]

    center = (sum(lats) / len(lats), sum(lons) / len(lons))
    span_km = haversine_km(min(lats), min(lons), max(lats), max(lons))
    dist_m = max(span_km * 1000 / 2 * 1.3, 500.0)

    G = ox.graph_from_point(
        center, dist=dist_m, network_type=mode, simplify=True
    )

    nodes = [ox.nearest_nodes(G, X=lon, Y=lat) for lat, lon in points]
    return G, nodes


def _assign_time(G: nx.MultiDiGraph, mode: Mode) -> tuple[str, float]:
    """Проставить рёбрам вес-время. Возвращает (имя_атрибута, множитель_бюджета).

    Единицы согласованы локально:
      drive       -> travel_time в СЕКУНДАХ, бюджет = minutes * 60
      walk / bike -> time в МИНУТАХ,         бюджет = minutes * 1
    """
    if mode == "drive":
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        return "travel_time", 60.0

    mpm = DEFAULT_SPEEDS[mode] * 1000 / 60
    for _, _, _, data in G.edges(keys=True, data=True):
        data["time"] = data["length"] / mpm
    return "time", 1.0


def route_length(points: list[tuple[float, float]], mode: Mode = "walk") -> float:
    """Длина маршрута P1->P2->...->Pn по дорогам (в том числе пешеходным), в метрах.

    Точки проходятся строго в переданном порядке (не оптимизируется).
    Бросает ValueError, если между соседними точками нет пути по сети
    """
    if len(points) < 2:
        raise ValueError("Нужно минимум 2 точки для маршрута")

    G, nodes = _graph_for_points(points, mode)

    total_m = 0.0
    for i in range(len(nodes) - 1):
        u, v = nodes[i], nodes[i + 1]
        try:
            total_m += nx.shortest_path_length(G, u, v, weight="length")
        except nx.NetworkXNoPath:
            raise ValueError(
                f"Нет пути по сети между точкой {i+1} и {i+2} "
                f"({points[i]} -> {points[i+1]}) в режиме {mode!r}"
            )
    return total_m


def isochrone(
    points: list[tuple[float, float]], minutes: float, mode: Mode = "walk"
) -> list[tuple[float, float]]:
    """Какие из points[1:] достижимы за minutes по дорогам/путям от points[0].

    points[0] — позиция пользователя (центр), points[1:] — кандидаты.
    Возвращает подмножество points[1:], достижимое в режиме mode.
    """
    G, nodes = _graph_for_points(points, mode)
    weight_attr, budget_mult = _assign_time(G, mode)

    sub = nx.ego_graph(
        G, nodes[0], radius=minutes * budget_mult, distance=weight_attr
    )
    reachable_nodes = set(sub.nodes())

    return [
        points[i]
        for i in range(1, len(points))
        if nodes[i] in reachable_nodes
    ]


def geocode(location: str, city_hint: str = "") -> tuple[float, float] | None:
    """Resolve a place name to (lat, lon) via Nominatim. Returns None on failure.
    TODO: RateLimiter для многократного вызова функции за 1 сек.
    TODO: cache
    """
    query = f"{location}, {city_hint}".strip(", ") if city_hint else location
    geolocator = Nominatim(user_agent="geo-analytics-agent/1.0", timeout=5)

    result_coords: tuple[float, float] | None = None
    try:
        result = geolocator.geocode(query)
        if result:
            result_coords = (result.latitude, result.longitude)
    except (GeocoderTimedOut, GeocoderServiceError):
        pass

    return result_coords

"""Filtering helpers — skeleton."""
from __future__ import annotations

from typing import Iterable

from lib.data_types import Place


def filter_by_category(places: Iterable[Place], category: str) -> list[Place]:
    return [p for p in places if p.amenity == category]


def filter_by_rating(places: Iterable[Place], min_rating: float) -> list[Place]:
    return [p for p in places if p.rating is not None and p.rating >= min_rating]

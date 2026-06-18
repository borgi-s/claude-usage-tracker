"""Cached, filter-dependent compute shared by both apps."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import streamlit as st

import metrics


def _sig1(df: pl.DataFrame) -> tuple:
    if df.is_empty() or "ts" not in df.columns:
        return (0, None)
    return (df.height, str(df["ts"].max()))


@dataclass
class FilteredCompute:
    sessions: pl.DataFrame


@st.cache_data(show_spinner=False)
def _filtered(_key: tuple, _fdf: pl.DataFrame) -> FilteredCompute:
    return FilteredCompute(sessions=metrics.session_summaries(_fdf))


def filtered_compute(fdf: pl.DataFrame) -> FilteredCompute:
    return _filtered(_sig1(fdf), fdf)

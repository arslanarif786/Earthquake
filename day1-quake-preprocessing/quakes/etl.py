"""Part 2, Task A: extract, transform, load, with logging."""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from . import config
from .cleaning import clean
from .features import (
    add_location_features,
    add_quality_features,
    add_target,
    add_time_features,
    drop_leaky_columns,
    group_rare,
)
from .fetch import fetch_feed, geojson_to_df

log = logging.getLogger("quakes.etl")
DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch a USGS feed; use the instructor fallback GeoJSON if fetch fails."""
    try:
        payload = fetch_feed(feed)
        df = geojson_to_df(payload)
        log.info("Fetched feed %s: %d rows", feed, len(df))
        return df
    except Exception as exc:
        fallback = config.FALLBACK_PATH
        log.warning("Could not fetch feed %s (%s); trying fallback %s", feed, exc, fallback)
        if not fallback.exists():
            raise RuntimeError(
                f"Could not fetch USGS feed and fallback file does not exist: {fallback}"
            ) from exc
        with fallback.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        df = geojson_to_df(payload)
        log.info("Loaded fallback data: %d rows", len(df))
        return df


def read_stream_files() -> pd.DataFrame:
    """Concatenate every data/stream/*.jsonl file (empty DataFrame if none)."""
    files = sorted(config.STREAM_DIR.glob("*.jsonl"))
    frames = []
    for path in files:
        try:
            frame = pd.read_json(path, lines=True)
            if not frame.empty:
                frames.append(frame)
                log.info("Read %d stream rows from %s", len(frame), path.name)
        except ValueError:
            log.warning("Skipping empty or invalid stream file: %s", path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw feed rows and create the model-ready feature/target columns."""
    if raw.empty:
        return raw.copy()
    log.info("Transform started with %d raw rows", len(raw))
    out = clean(raw)
    log.info("After cleaning: %d rows", len(out))
    out = add_time_features(out)
    out = add_quality_features(out)
    out = add_location_features(out)
    out = add_target(out)
    if "region" in out.columns:
        out["region"] = group_rare(out["region"].fillna("Unknown"), top_k=15)
    out = drop_leaky_columns(out)
    log.info("After feature engineering and leakage removal: %d rows", len(out))
    return out


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Upsert rows by id into SQLite, replacing previous versions of an event."""
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("table must be a simple SQLite table name")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    if "id" not in df.columns:
        raise ValueError("Cannot upsert earthquake rows without an 'id' column")

    data = df.copy().drop_duplicates(subset=["id"], keep="last")
    # Store pandas timestamps as portable ISO text values in SQLite.
    for column in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[column]):
            data[column] = data[column].map(lambda value: value.isoformat() if pd.notna(value) else None)

    with sqlite3.connect(path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None
        if not exists:
            data.head(0).to_sql(table, connection, index=False)
        connection.executemany(
            f'DELETE FROM "{table}" WHERE id = ?',
            [(value,) for value in data["id"].dropna().tolist()],
        )
        data.to_sql(table, connection, if_exists="append", index=False)
    log.info("Upserted %d rows into SQLite table %s at %s", len(data), table, path)


def run_etl(mode: str = "batch") -> None:
    """Extract, optionally merge streams, transform, and load into SQLite."""
    if mode not in {"batch", "merge"}:
        raise ValueError("mode must be 'batch' or 'merge'")
    raw = extract()
    log.info("Extract stage: %d rows", len(raw))
    if mode == "merge":
        stream = read_stream_files()
        log.info("Stream merge stage: %d rows", len(stream))
        if not stream.empty:
            raw = pd.concat([raw, stream], ignore_index=True, sort=False)
            log.info("Combined feed and stream: %d rows", len(raw))
    ready = transform(raw)
    log.info("Load stage: %d model-ready rows", len(ready))
    load_sqlite(ready, DB_PATH)
    log.info("ETL completed successfully")


def main() -> None:
    """Configure file/console logging and run the requested ETL mode."""
    parser = argparse.ArgumentParser(description="USGS earthquake ETL")
    parser.add_argument("--mode", choices=("batch", "merge"), default="batch")
    args = parser.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    run_etl(args.mode)


if __name__ == "__main__":
    main()

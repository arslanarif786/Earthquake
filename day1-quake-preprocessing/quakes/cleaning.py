"""Task 3: cleaning."""
import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""
    out = df.copy()

    out["time"] = pd.to_datetime(out["time"], unit="ms", utc=True)
    out["updated"] = pd.to_datetime(out["updated"], unit="ms", utc=True)

    return out


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""
    out = df.copy()

    out = out.sort_values("updated")
    out = out.drop_duplicates(subset="id", keep="last")

    return out


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' (strip, lowercase) and keep only 'earthquake'."""
    out = df.copy()

    out["type"] = out["type"].astype(str).str.strip().str.lower()

    return out[out["type"] == "earthquake"].copy()


def extract_region(place: pd.Series) -> pd.Series:
    """Text after the last comma, or the whole string if there is no comma.
    Missing places become 'Unknown'.
    """
    return (
        place.fillna("Unknown")
        .astype(str)
        .str.rsplit(",", n=1)
        .str[-1]
        .str.strip()
    )


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    return (s < lower_bound) | (s > upper_bound)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""
    return df.dropna(subset=["mag"]).copy()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Chain the steps above (think about the order) and add a 'region' column."""
    out = df.copy()

    # Convert types first.
    out = epoch_ms_to_datetime(out)

    # Keep the latest version of each event.
    out = dedupe_latest(out)

    # Keep only actual earthquake events.
    out = keep_earthquakes(out)

    # Extract region from place.
    out["region"] = extract_region(out["place"])

    # Remove rows without the target.
    out = drop_missing_target(out)

    return out
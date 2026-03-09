from __future__ import annotations

import numpy as np
import pandas as pd


def _convert_sqft(value: str) -> float | None:
    text = str(value).strip()
    tokens = text.split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0]) + float(tokens[1])) / 2.0
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _remove_pp_sq_outliers(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, subdf in df.groupby("location"):
        mean = np.mean(subdf["pp_sq"])
        std = np.std(subdf["pp_sq"])
        filtered = subdf[(subdf["pp_sq"] > (mean - std)) & (subdf["pp_sq"] <= (mean + std))]
        frames.append(filtered)
    return pd.concat(frames, ignore_index=True)


def _remove_bhk_outliers(df: pd.DataFrame) -> pd.DataFrame:
    drop_indices: list[int] = []
    for _, location_df in df.groupby("location"):
        bhk_stats: dict[int, dict[str, float]] = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[int(bhk)] = {
                "mean": float(np.mean(bhk_df["pp_sq"])),
                "count": float(bhk_df.shape[0]),
            }
        for bhk, bhk_df in location_df.groupby("bhk"):
            prev = bhk_stats.get(int(bhk) - 1)
            if prev and prev["count"] > 5:
                drop_indices.extend(list(bhk_df[bhk_df["pp_sq"] < prev["mean"]].index))
    return df.drop(drop_indices, axis="index")


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df1 = pd.read_csv(csv_path)
    df2 = df1.drop(["area_type", "availability", "society", "balcony"], axis="columns")
    df3 = df2.dropna().copy()
    df3["bhk"] = df3["size"].apply(lambda x: int(x.split(" ")[0]))
    df3["total_sqft"] = df3["total_sqft"].apply(_convert_sqft)
    df3 = df3.dropna(subset=["total_sqft"]).copy()

    df3["location"] = df3["location"].str.strip()
    location_counts = df3.groupby("location")["location"].agg("count")
    less_than_10 = set(location_counts[location_counts <= 10].index)
    df3["location"] = df3["location"].apply(lambda x: "Other" if x in less_than_10 else x)

    df3["pp_sq"] = df3["price"] * 100000 / df3["total_sqft"]
    df4 = df3[~(df3["total_sqft"] / df3["bhk"] < 300)].copy()
    df5 = _remove_pp_sq_outliers(df4)
    df6 = _remove_bhk_outliers(df5)
    df7 = df6[df6["bath"] < (df6["bhk"] + 2)].copy()

    return df7[["location", "total_sqft", "bath", "bhk", "price"]].reset_index(drop=True)

# make_sf_blocks_centroids.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd

CRIME_DATA_DIR = Path(os.getenv("CRIME_DATA_DIR","crime_prediction_data"))
BLOCKS = CRIME_DATA_DIR / "sf_census_blocks_with_population.geojson"
OUT = CRIME_DATA_DIR / "sf_blocks_centroids.csv"

def log(x): print(x, flush=True)

def geoid11(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str.replace(" ","",regex=False)
         .str.zfill(11)
         .str[:11]
    )

def main():
    if not BLOCKS.exists():
        log(f"❌ yok: {BLOCKS}")
        return 2

    gdf = gpd.read_file(BLOCKS)
    log(f"📍 blocks okundu: {BLOCKS} ({len(gdf):,})")

    # normalize
    if "GEOID" not in gdf.columns:
        raise RuntimeError("BLOCKS içinde GEOID yok")

    gdf["GEOID"] = geoid11(gdf["GEOID"])

    gdf = gdf.to_crs("EPSG:4326")
    cent = gdf.dissolve(by="GEOID", as_index=True)["geometry"].centroid
    df = pd.DataFrame({
        "GEOID": cent.index,
        "centroid_lat": cent.y,
        "centroid_lon": cent.x
    }).reset_index(drop=True)

    df.to_csv(OUT, index=False)
    log(f"✅ yazıldı: {OUT} ({len(df):,})")

    return 0

if __name__=="__main__":
    raise SystemExit(main())

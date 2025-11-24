# make_neighbors_fr.py 

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True


# =============================================================================
# CONFIG / PATHS
# =============================================================================
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV  = Path(os.getenv("NEIGHBOR_INPUT_CSV",  str(BASE_DIR / "fr_crime_08.csv"))).resolve()
OUT_CSV = Path(os.getenv("NEIGHBOR_OUTPUT_CSV", str(BASE_DIR / "fr_crime_09.csv"))).resolve()

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))


# =============================================================================
# HELPERS
# =============================================================================
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r:,} satır × {c} sütun")


def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .str[:L]
         .str.zfill(L)
    )


def _pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for cand in cands:
        if isinstance(cand, (list, tuple)):
            for c in cand:
                if c.lower() in low:
                    return low[c.lower()]
        else:
            if cand.lower() in low:
                return low[cand.lower()]
    return None


def safe_save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_neighbors_file(base_dir: Path) -> Path:
    """
    neighbors.csv'yi mümkün olan her yerden bulmaya çalışır.
    Öncelik sırası:
      1) NEIGH_FILE env (tam yol)
      2) base_dir/neighbors.csv
      3) script klasörü ve üstleri
      4) repo içinde rglob
    """
    tried = []

    # 1) env ile verilmişse
    env_path = os.environ.get("NEIGH_FILE")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        tried.append(str(p))
        if p.exists():
            print(f"✅ neighbors.csv env ile bulundu: {p}")
            return p

    # 2) base_dir içinde
    p1 = (base_dir / "neighbors.csv").resolve()
    tried.append(str(p1))
    if p1.exists():
        print(f"✅ neighbors.csv BASE_DIR içinde bulundu: {p1}")
        return p1

    # 3) script klasörü ve üstleri
    here = Path(__file__).resolve().parent
    ups = [here, here.parent, here.parent.parent, here.parent.parent.parent]
    for up in ups:
        cand = (up / "neighbors.csv").resolve()
        tried.append(str(cand))
        if cand.exists():
            print(f"✅ neighbors.csv script çevresinde bulundu: {cand}")
            return cand

    # 4) repo içinde glob (son çare)
    try:
        root = here.parent  # genelde repo root burada olur
        for cand in root.rglob("neighbors.csv"):
            cand = cand.resolve()
            tried.append(str(cand))
            print(f"✅ neighbors.csv rglob ile bulundu: {cand}")
            return cand
    except Exception:
        pass

    # bulunamadı → açıklayıcı hata
    msg = "❌ Komşuluk dosyası (neighbors.csv) bulunamadı.\nDenediğim yollar:\n- " + "\n- ".join(tried)
    raise FileNotFoundError(msg)


def compute_neighbor_window(
    focal: pd.DataFrame,
    events: pd.DataFrame,
    window: pd.Timedelta,
    out_col: str
) -> pd.Series:
    """
    focal:  ['geoid','datetime']  (tüm satırlar)
    events: ['geoid_focal','datetime','crime_w']  (komşu suç eventleri)
    window: örn pd.Timedelta('24h')

    Mantık: cum(t) - cum(t-window)
    """
    focal_sorted  = focal.sort_values(["geoid", "datetime"]).reset_index()
    events_sorted = events.sort_values(["geoid_focal", "datetime"]).reset_index(drop=True)

    # events üzerinde cumulative sum
    events_sorted["cum"] = events_sorted.groupby("geoid_focal")["crime_w"].cumsum()

    # cum(t)
    left = pd.merge_asof(
        focal_sorted,
        events_sorted[["geoid_focal","datetime","cum"]],
        left_on="datetime",
        right_on="datetime",
        left_by="geoid",
        right_by="geoid_focal",
        direction="backward",
        allow_exact_matches=True
    )
    cum_t = left["cum"].fillna(0.0)

    # cum(t-window)
    focal_minus = focal_sorted.copy()
    focal_minus["datetime"] = focal_minus["datetime"] - window

    right = pd.merge_asof(
        focal_minus,
        events_sorted[["geoid_focal","datetime","cum"]],
        left_on="datetime",
        right_on="datetime",
        left_by="geoid",
        right_by="geoid_focal",
        direction="backward",
        allow_exact_matches=True
    )
    cum_tm = right["cum"].fillna(0.0)

    vals = (cum_t - cum_tm).clip(lower=0).astype("int64")

    # orijinal index sırasına geri döndür
    vals.index = left["index"].values
    return vals.reindex(focal.index, fill_value=0)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"❌ Girdi bulunamadı: {IN_CSV}")

    NEIGH_FILE = resolve_neighbors_file(BASE_DIR)

    print(f"▶︎ IN : {IN_CSV}")
    print(f"▶︎ NB : {NEIGH_FILE}")
    print(f"▶︎ OUT: {OUT_CSV}")

    # 1) crime df
    df = pd.read_csv(IN_CSV, low_memory=False)
    log_shape(df, "CRIME (load)")

    gcol = _pick_col(df.columns, ["geoid","GEOID","geography_id","tract"], "iris", "insee")
    dcol = _pick_col(df.columns, ["datetime","incident_datetime","timestamp"], "date")

    if not gcol:
        raise RuntimeError("❌ GEOID kolonu bulunamadı.")
    if not dcol:
        raise RuntimeError("❌ datetime/date kolonu bulunamadı.")

    df["geoid"] = _norm_geoid(df[gcol])
    df["datetime"] = pd.to_datetime(df[dcol], errors="coerce")

    nat_rate = df["datetime"].isna().mean()
    if nat_rate > 0:
        print(f"⚠️ datetime parse NaT oranı: {nat_rate:.3f}")

    # suç ağırlığı: crime_count varsa onu, yoksa Y_label, yoksa 1
    if "crime_count" in df.columns:
        crime_w = pd.to_numeric(df["crime_count"], errors="coerce").fillna(0)
    elif "Y_label" in df.columns:
        crime_w = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0)
    else:
        crime_w = pd.Series(1, index=df.index)

    df["crime_w"] = crime_w.astype("float32")

    # 2) neighbors
    nb = pd.read_csv(NEIGH_FILE, low_memory=False, dtype=str).dropna()
    s = _pick_col(nb.columns, ["geoid","src","source","from","GEOID"])
    t = _pick_col(nb.columns, ["neighbor","dst","target","to","NEIGHBOR"])

    if not s or not t:
        raise RuntimeError(f"❌ neighbors.csv başlıkları anlaşılamadı: {nb.columns.tolist()}")

    nb = nb.rename(columns={s:"geoid", t:"neighbor"})[["geoid","neighbor"]].dropna()
    nb["geoid"]    = _norm_geoid(nb["geoid"])
    nb["neighbor"] = _norm_geoid(nb["neighbor"])
    nb = nb.drop_duplicates()

    log_shape(nb, "NEIGHBORS (load+norm)")

    # 3) Komşu suç eventleri (sadece crime_w > 0)
    crimes = df.loc[df["crime_w"] > 0, ["geoid","datetime","crime_w"]].copy()
    crimes = crimes.dropna(subset=["datetime"])
    log_shape(crimes, "CRIMES (crime_w>0)")

    # neighbor’da olan suçları focal geoid’e bağla
    neigh_events = nb.merge(
        crimes,
        left_on="neighbor",
        right_on="geoid",
        how="inner",
        suffixes=("", "_nei")
    )
    neigh_events = neigh_events.rename(columns={"geoid_x":"geoid_focal"})
    neigh_events = neigh_events[["geoid_focal","datetime","crime_w"]].dropna()

    log_shape(neigh_events, "NEIGH_EVENTS (mapped)")

    # 4) Window hesapları (satır düşmeden df’e ekle)
    focal = df[["geoid","datetime"]].copy()

    df["neighbor_crime_24h"] = compute_neighbor_window(
        focal, neigh_events, pd.Timedelta(hours=24), "neighbor_crime_24h"
    )
    df["neighbor_crime_72h"] = compute_neighbor_window(
        focal, neigh_events, pd.Timedelta(hours=72), "neighbor_crime_72h"
    )
    df["neighbor_crime_7d"]  = compute_neighbor_window(
        focal, neigh_events, pd.Timedelta(days=7), "neighbor_crime_7d"
    )

    # 5) cleanup
    df = df.drop(columns=["crime_w"], errors="ignore")

    # 6) save
    safe_save_csv(df, OUT_CSV)
    log_shape(df, "OUT (saved)")
    print(f"✅ Yazıldı: {OUT_CSV.name}")

    # mini preview
    show_cols = ["geoid","datetime","neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()

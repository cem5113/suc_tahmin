# update_911_daily.py  —  911 günlük özet + GRID/EVENTS zenginleştirme
# Not: 1d/3d/7d (ve 30d) roll hesaplarına dokunulmadı.

from __future__ import annotations
import os, re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# ------------ ENV & YOLLAR ------------
P_911      = Path(os.getenv("FR_911_PATH", "sf_911_last_5_year.csv"))
COL_DT_911 = os.getenv("FR_911_DATE_COL", "incident_datetime")
COL_G_911  = os.getenv("FR_911_GEOID_COL", "GEOID")
COL_CAT    = os.getenv("FR_911_CAT_COL", "category")

GRID_IN    = Path(os.getenv("FR_GRID_DAILY_IN",  "fr_crime_grid_daily.csv"))
GRID_OUT   = Path(os.getenv("FR_GRID_DAILY_OUT", "fr_crime_grid_daily.csv"))
EV_IN      = Path(os.getenv("FR_EVENTS_DAILY_IN","fr_crime_events_daily.csv"))
EV_OUT     = Path(os.getenv("FR_EVENTS_DAILY_OUT","fr_crime_events_daily.csv"))

CAT_TOPK   = int(os.getenv("FR_CAT_TOPK", "8"))
WINS_Q     = float(os.getenv("FR_911_WINSOR_Q", "0"))  # 0 → kapalı
EMA_ALPHAS = tuple(float(x) for x in os.getenv("FR_911_EMA_ALPHAS", "0.3,0.5").split(","))

pd.options.mode.copy_on_write = True

# ------------ LOG ------------
def log(x: str) -> None:
    print(x, flush=True)

# ------------ IO ------------
def _read_table(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"❌ Bulunamadı: {p}")
        return pd.DataFrame()
    try:
        suf = "".join(p.suffixes).lower()
        if suf.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif suf.endswith(".csv.gz"):
            df = pd.read_csv(p, compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(p, low_memory=False)
        log(f"📖 Okundu: {p}  ({len(df):,}×{df.shape[1]})")
        return df
    except Exception as e:
        log(f"❌ Okuma hatası: {p} — {type(e).__name__}: {e}")
        return pd.DataFrame()

def _safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    # downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p}  ({len(df):,}×{df.shape[1]})")

# ------------ YARDIMCILAR ------------
def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str[:L].str.zfill(L)
    )

def _autodetect_dt_col(df: pd.DataFrame, pref: str) -> Optional[str]:
    if pref in df.columns: return pref
    for c in ("received_datetime","received_time","datetime","date_time","timestamp","created_at","time"):
        if c in df.columns: return c
    if "date" in df.columns and "time" in df.columns:
        return "date+time"
    if "date" in df.columns:
        return "date"
    return None

def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")
    return s[:24] or "cat"

# ------------ 911 → GÜNLÜK ------------
def make_911_daily(df911: pd.DataFrame, col_g: str, col_dt_hint: str) -> pd.DataFrame:
    # benzersiz olay
    for cid in ("call_id","incident_id","id"):
        if cid in df911.columns:
            before = len(df911)
            df911 = df911.drop_duplicates(subset=[cid]).copy()
            if len(df911) != before:
                log(f"🧹 Dedup: {before-len(df911)} satır çıkarıldı ({cid})")
            break

    # GEOID
    if col_g not in df911.columns:
        cand = [c for c in df911.columns if "geoid" in c.lower()]
        if not cand:
            log("⚠️ 911 verisinde GEOID yok → atlanıyor.")
            return pd.DataFrame()
        col_g = cand[0]
    df911["GEOID"] = _norm_geoid(df911[col_g])

    # tarih
    use_dt = _autodetect_dt_col(df911, col_dt_hint)
    if not use_dt:
        log("⚠️ 911 zaman sütunu bulunamadı → atlanıyor.")
        return pd.DataFrame()
    if use_dt == "date+time":
        dt = pd.to_datetime(
            df911["date"].astype(str).str.strip() + " " + df911["time"].astype(str).str.strip(),
            errors="coerce", utc=True
        )
    else:
        dt = pd.to_datetime(df911[use_dt], errors="coerce", utc=True)
    df911["date"] = dt.dt.date

    # günlük sayım
    daily_raw = (
        df911.dropna(subset=["GEOID","date"])
             .groupby(["GEOID","date"], as_index=False)
             .size()
             .rename(columns={"size":"n911_d"})
    )

    # winsor (opsiyonel)
    if 0 < WINS_Q < 1 and not daily_raw.empty:
        q = float(daily_raw["n911_d"].quantile(WINS_Q))
        daily_raw["n911_d"] = daily_raw["n911_d"].clip(upper=max(q, 1.0))
        log(f"🔧 Winsorize: q={WINS_Q} → üst sınır ≈ {q:.1f}")

    # tam takvim 0 doldur
    if daily_raw.empty:
        return daily_raw.assign(
            n911_prev_1d=np.int32(), n911_roll_3d=np.int32(),
            n911_roll_7d=np.int32(), n911_roll_30d=np.int32(),
            **{f"n911_ema_a{int(a*10)}": np.float32() for a in EMA_ALPHAS},
            n911_trend_7v30=np.float32()
        )

    all_days = pd.date_range(daily_raw["date"].min(), daily_raw["date"].max(), freq="D").date
    geoids = daily_raw["GEOID"].unique()
    idx = pd.MultiIndex.from_product([geoids, all_days], names=["GEOID","date"])
    daily = (daily_raw.set_index(["GEOID","date"])
                        .reindex(idx, fill_value=0)
                        .reset_index()
                        .sort_values(["GEOID","date"])
                        .reset_index(drop=True))

    # —— SIZINTISIZ ÖZETLER (dokunma: 1d/3d/7d/30d) ——
    daily["n911_prev_1d"]  = daily.groupby("GEOID")["n911_d"].shift(1)
    daily["n911_roll_3d"]  = daily.groupby("GEOID")["n911_d"].shift(1).rolling(3,  min_periods=1).sum()
    daily["n911_roll_7d"]  = daily.groupby("GEOID")["n911_d"].shift(1).rolling(7,  min_periods=1).sum()
    daily["n911_roll_30d"] = daily.groupby("GEOID")["n911_d"].shift(1).rolling(30, min_periods=1).sum()

    # EMA (opsiyonel): shift(1) tabanı
    for a in EMA_ALPHAS:
        a = float(a)
        daily[f"n911_ema_a{int(a*10)}"] = (
            daily.groupby("GEOID")["n911_d"].apply(lambda s: s.shift(1).ewm(alpha=a, adjust=False).mean())
        ).astype("float32")

    # tip/NaN temizliği
    for c in ("n911_d","n911_prev_1d","n911_roll_3d","n911_roll_7d","n911_roll_30d"):
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0).astype("int32")

    daily["n911_trend_7v30"] = (daily["n911_roll_7d"] - daily["n911_roll_30d"]).astype("float32")

    return daily

# ------------ 911 → KATEGORİ ------------
def make_911_category_rollings(df911: pd.DataFrame) -> pd.DataFrame:
    cat_col = None
    for c in (COL_CAT, "type", "call_type", "category_name", "event_type"):
        if c and c in df911.columns:
            cat_col = c; break
    if not cat_col:
        log("ℹ️ 911 kategori sütunu yok — kategori rolling atlandı.")
        return pd.DataFrame()

    if "date" not in df911.columns:
        use_dt = _autodetect_dt_col(df911, COL_DT_911)
        if not use_dt:
            log("ℹ️ Kategori için tarih türetilemedi.")
            return pd.DataFrame()
        if use_dt == "date+time":
            dt = pd.to_datetime(df911["date"].astype(str)+" "+df911["time"].astype(str), errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(df911[use_dt], errors="coerce", utc=True)
        df911["date"] = dt.dt.date

    df911["GEOID"] = _norm_geoid(df911.get(COL_G_911, df911["GEOID"]))
    cat = (df911.dropna(subset=["GEOID","date"])
                 .groupby(["GEOID","date",cat_col], as_index=False)
                 .size().rename(columns={"size":"n911_cat"}))
    if cat.empty:
        return pd.DataFrame()

    # en sık K kategori
    topk = cat.groupby(cat_col)["n911_cat"].sum().nlargest(CAT_TOPK).index.tolist()
    cat = cat[cat[cat_col].isin(topk)].copy()
    cat["_cat_slug"] = cat[cat_col].apply(_slug)

    out = []
    for geo in cat["GEOID"].unique():
        df_g = cat[cat["GEOID"] == geo].copy()
        days = pd.date_range(df_g["date"].min(), df_g["date"].max(), freq="D").date
        for cg in df_g["_cat_slug"].unique():
            serie = (df_g[df_g["_cat_slug"] == cg].set_index("date")["n911_cat"]
                     .reindex(days, fill_value=0))
            tmp = serie.reset_index().rename(columns={"index":"date","n911_cat":"cnt"})
            tmp["GEOID"], tmp["_cat_slug"] = geo, cg
            out.append(tmp)
    if not out:
        return pd.DataFrame()

    cat_full = pd.concat(out, ignore_index=True).sort_values(["GEOID","_cat_slug","date"])
    def _roll(g):
        g["cnt_prev_1d"]  = g["cnt"].shift(1)
        g["cnt_roll_7d"]  = g["cnt"].shift(1).rolling(7,  min_periods=1).sum()
        g["cnt_roll_30d"] = g["cnt"].shift(1).rolling(30, min_periods=1).sum()
        return g
    cat_full = cat_full.groupby(["GEOID","_cat_slug"], group_keys=False).apply(_roll)

    piv = cat_full.pivot_table(
        index=["GEOID","date"],
        columns="_cat_slug",
        values=["cnt_prev_1d","cnt_roll_7d","cnt_roll_30d"],
        fill_value=0, aggfunc="first"
    )
    piv.columns = [f"n911_{lvl2}_{lvl1}" for (lvl1,lvl2) in piv.columns.to_flat_index()]
    return piv.reset_index()

# ------------ GRID / EVENTS ENRICH ------------
def enrich_grid(grid: pd.DataFrame, g911: pd.DataFrame, gcat: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    if "GEOID" in out.columns:
        out["GEOID"] = _norm_geoid(out["GEOID"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.merge(g911, on=["GEOID","date"], how="left")
    if not gcat.empty:
        out = out.merge(gcat, on=["GEOID","date"], how="left")

    for c in ("n911_d","n911_prev_1d","n911_roll_3d","n911_roll_7d","n911_roll_30d"):
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int32")
    for c in ("n911_trend_7v30",) + tuple(f"n911_ema_a{int(a*10)}" for a in EMA_ALPHAS):
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
    if not gcat.empty:
        for c in gcat.columns:
            if c not in ("GEOID","date"):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

def enrich_events(events: pd.DataFrame, g911: pd.DataFrame, gcat: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if "GEOID" in out.columns:
        out["GEOID"] = _norm_geoid(out["GEOID"])
    if "date" not in out.columns:
        if "incident_datetime" in out.columns:
            out["date"] = pd.to_datetime(out["incident_datetime"], errors="coerce", utc=True).dt.date
        else:
            raise ValueError("OUT_EVENTS içinde 'date' yok ve 'incident_datetime' yok.")
    feats = g911[[
        "GEOID","date","n911_prev_1d","n911_roll_3d","n911_roll_7d",
        "n911_roll_30d", *[f"n911_ema_a{int(a*10)}" for a in EMA_ALPHAS], "n911_trend_7v30"
    ]].drop_duplicates()
    out = out.merge(feats, on=["GEOID","date"], how="left")
    if not gcat.empty:
        out = out.merge(gcat, on=["GEOID","date"], how="left")

    for c in ("n911_prev_1d","n911_roll_3d","n911_roll_7d","n911_roll_30d"):
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int32")
    for c in (tuple(f"n911_ema_a{int(a*10)}" for a in EMA_ALPHAS) + ("n911_trend_7v30",)):
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
    if not gcat.empty:
        for c in gcat.columns:
            if c not in ("GEOID","date"):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

# ------------ MAIN ------------
def main() -> int:
    log("🚀 update_911_daily.py (revize)")
    df911 = _read_table(P_911)
    if df911.empty:
        log("ℹ️ 911 verisi yok → adım atlandı.")
        return 0

    g911 = make_911_daily(df911, COL_G_911, COL_DT_911)
    if g911.empty:
        log("ℹ️ 911 günlük türetilemedi → adım atlandı.")
        return 0

    gcat = make_911_category_rollings(df911)  # opsiyonel
    if not gcat.empty:
        log(f"📊 Kategori rolling kolonları: {len(gcat.columns)-2}")

    # GRID
    grid = _read_table(GRID_IN)
    if not grid.empty:
        grid2 = enrich_grid(grid, g911, gcat)
        _safe_save_csv(grid2, GRID_OUT)
    else:
        log("ℹ️ GRID bulunamadı → atlandı.")

    # EVENTS
    ev = _read_table(EV_IN)
    if not ev.empty:
        ev2 = enrich_events(ev, g911, gcat)
        _safe_save_csv(ev2, EV_OUT)
    else:
        log("ℹ️ EVENTS bulunamadı → atlandı.")

    log("✅ Tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

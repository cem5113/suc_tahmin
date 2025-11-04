# update_911_daily.py — revize (dosya bulma + NN fallback + geniş tarih algılama)
from __future__ import annotations
import os, re
from pathlib import Path
import pandas as pd
import numpy as np

# ------------ ENV & yollar ------------
CRIME_DATA_DIR = Path(os.getenv("CRIME_DATA_DIR", ".")).resolve()

P_911      = Path(os.getenv("FR_911_PATH", "fr_911.csv"))
COL_DT_911 = os.getenv("FR_911_DATE_COL", "incident_datetime")
COL_G_911  = os.getenv("FR_911_GEOID_COL", "GEOID")
COL_CAT    = os.getenv("FR_911_CAT_COL", "category")

GRID_IN    = Path(os.getenv("FR_GRID_DAILY_IN",  "crime_prediction_data/fr_crime_grid_daily.csv"))
GRID_OUT   = Path(os.getenv("FR_GRID_DAILY_OUT", "crime_prediction_data/fr_crime_grid_daily.csv"))
EV_IN      = Path(os.getenv("FR_EVENTS_DAILY_IN","crime_prediction_data/fr_crime_events_daily.csv"))
EV_OUT     = Path(os.getenv("FR_EVENTS_DAILY_OUT","crime_prediction_data/fr_crime_events_daily.csv"))

# GEOID fallback için lookup (opsiyonel): CSV kolonları: GEOID,lat,lon
GEOID_LOOKUP = Path(os.getenv("FR_GEOID_LOOKUP", "sf_blocks_centroids.csv"))

CAT_TOPK   = int(os.getenv("FR_CAT_TOPK", "8"))
WINS_Q     = float(os.getenv("FR_911_WINSOR_Q", "0.999"))

def log(x): print(x, flush=True)

# ---------- Yol bulucu (CRIME_DATA_DIR altında da dene) ----------
def _resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p
    direct = (Path.cwd() / p).resolve()
    if direct.exists():
        return direct
    under = (CRIME_DATA_DIR / p).resolve()
    return under

def _read_csv(p: Path) -> pd.DataFrame:
    p2 = _resolve_path(p)
    if not p2.exists():
        log(f"❌ Bulunamadı: {p2} (P_911='{p}')")
        return pd.DataFrame()
    log(f"📖 Okunuyor: {p2}")
    df = pd.read_csv(p2, low_memory=False)
    log(f"✅ Okundu: {p2} ({len(df):,}×{df.shape[1]})")
    return df

def _save_csv(df: pd.DataFrame, p: Path):
    p2 = _resolve_path(p)
    p2.parent.mkdir(parents=True, exist_ok=True)
    # downcast + kaydet
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    tmp = p2.with_suffix(p2.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p2)
    log(f"💾 Yazıldı: {p2} ({len(df):,}×{df.shape[1]})")

def _norm_geoid(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)")[0]
         .fillna("")
         .apply(lambda x: x.zfill(11) if x else "")
    )

# ---- GENİŞLETİLMİŞ tarih kolonu autodetect ----
_DT_CANDIDATES = [
    # en sık kullandıkların
    "incident_datetime", "received_time", "received_datetime",
    "call_received_datetime", "call_time", "call_datetime", "call_timestamp",
    # genel varyantlar
    "datetime","occurred_at","timestamp","date_time","created_at","updated_at",
    "created_datetime","opened_datetime","reported_datetime","time",
    # salt tarih
    "date"
]

def autodetect_dt_col(df: pd.DataFrame, pref: str) -> str | None:
    if pref in df.columns: 
        log(f"🧭 Tarih kolonu (ENV): {pref}")
        return pref
    for c in _DT_CANDIDATES:
        if c in df.columns:
            log(f"🧭 Tarih kolonu (auto): {c}")
            return c
    if "date" in df.columns and "time" in df.columns:
        log("🧭 Tarih kolonu (date+time) kullanılıyor.")
        return "date+time"
    return None

def _slug(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:24] if s else "cat"

# ---------- lat/lon → GEOID fallback (opsiyonel, NN ile) ----------
def _geoid_from_latlon(df: pd.DataFrame) -> pd.Series:
    lat_cols = [c for c in df.columns if c.lower() in ("lat","latitude","y","lat_dd")]
    lon_cols = [c for c in df.columns if c.lower() in ("lon","lng","longitude","x","lon_dd")]
    if not lat_cols or not lon_cols:
        return pd.Series([], dtype=str)

    look = _read_csv(GEOID_LOOKUP)
    if look.empty or not all(k in look.columns for k in ["GEOID","lat","lon"]):
        return pd.Series([], dtype=str)

    try:
        from sklearn.neighbors import NearestNeighbors
        pts = look[["lat","lon"]].to_numpy(dtype=float)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pts)

        lat = pd.to_numeric(df[lat_cols[0]], errors="coerce")
        lon = pd.to_numeric(df[lon_cols[0]], errors="coerce")
        mask = lat.notna() & lon.notna()
        out = pd.Series(index=df.index, dtype=object)

        if mask.any():
            query = np.c_[lat[mask].to_numpy(), lon[mask].to_numpy()]
            dist, idx = nbrs.kneighbors(query, n_neighbors=1, return_distance=True)
            out.loc[mask] = look.iloc[idx[:,0]]["GEOID"].to_numpy()
        return out.fillna("")
    except Exception as e:
        log(f"ℹ️ GEOID NN fallback başarısız: {e}")
        return pd.Series([], dtype=str)

# ========== 911 → Günlük ==========
def make_911_daily(df911: pd.DataFrame, col_g: str, col_dt_hint: str) -> pd.DataFrame:
    # [8] dedup (varsa)
    for cid in ["call_id", "incident_id", "id"]:
        if cid in df911.columns:
            before = len(df911)
            df911 = df911.drop_duplicates(subset=[cid]).copy()
            if len(df911) != before: log(f"🧹 Dedup: {before-len(df911)} satır çıkarıldı ({cid})")
            break

    # GEOID normalize / fallback
    if col_g not in df911.columns:
        cand = [c for c in df911.columns if "geoid" in c.lower()]
        if cand:
            col_g = cand[0]
            df911["GEOID"] = _norm_geoid(df911[col_g])
        else:
            nn_geoid = _geoid_from_latlon(df911)
            if nn_geoid.empty or nn_geoid.isna().all():
                log("⚠️ 911 verisinde GEOID yok ve lat/lon fallback başarısız → atlanıyor.")
                return pd.DataFrame()
            df911["GEOID"] = _norm_geoid(nn_geoid)
    else:
        df911["GEOID"] = _norm_geoid(df911[col_g])

    # Eğer dosya zaten günlükse (opsiyonel hızlandırıcı)
    if "date" in df911.columns and ("n911_d" in df911.columns or "count" in df911.columns):
        df911["date"] = pd.to_datetime(df911["date"], errors="coerce").dt.date
        if "n911_d" not in df911.columns:
            df911 = df911.rename(columns={"count":"n911_d"})
        daily_raw = df911[["GEOID","date","n911_d"]].dropna().copy()
    else:
        # datetime parse → date
        use_dt = autodetect_dt_col(df911, col_dt_hint)
        if use_dt is None:
            log("⚠️ 911 zaman sütunu bulunamadı → atlanıyor.")
            return pd.DataFrame()
        if use_dt == "date+time":
            dt = pd.to_datetime(df911["date"].astype(str).str.strip() + " " +
                                df911["time"].astype(str).str.strip(),
                                errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(df911[use_dt], errors="coerce", utc=True)
        df911["date"] = dt.dt.date

        # günlük sayım (ham)
        daily_raw = (
            df911.dropna(subset=["GEOID","date"])
                 .groupby(["GEOID","date"], as_index=False)
                 .size()
                 .rename(columns={"size":"n911_d"})
        )

    # [8] winsorize (opsiyonel)
    if WINS_Q and 0 < WINS_Q < 1 and not daily_raw.empty:
        q = daily_raw["n911_d"].quantile(WINS_Q)
        daily_raw["n911_d"] = daily_raw["n911_d"].clip(upper=max(q, 1))
        log(f"🔧 Winsorize: q={WINS_Q} üst={q:.1f}")

    # [1] tam takvim reindex (eksik günleri 0)
    if not daily_raw.empty:
        all_days = pd.date_range(daily_raw["date"].min(), daily_raw["date"].max(), freq="D").date
        geoids = daily_raw["GEOID"].unique()
        idx = pd.MultiIndex.from_product([geoids, all_days], names=["GEOID","date"])
        daily = (daily_raw.set_index(["GEOID","date"])
                           .reindex(idx, fill_value=0)
                           .reset_index())
    else:
        daily = daily_raw

    # leakage engelle: shift(1) + rolling(3/7/30)
    daily = daily.sort_values(["GEOID","date"])
    daily["n911_prev_1d"]  = daily.groupby("GEOID")["n911_d"].shift(1)
    daily["n911_roll_3d"]  = daily.groupby("GEOID")["n911_d"].shift(1).rolling(3,  min_periods=1).sum()
    daily["n911_roll_7d"]  = daily.groupby("GEOID")["n911_d"].shift(1).rolling(7,  min_periods=1).sum()
    daily["n911_roll_30d"] = daily.groupby("GEOID")["n911_d"].shift(1).rolling(30, min_periods=1).sum()

    # [2] EMA/decay (shift(1) üstünden)
    for alpha in (0.3, 0.5):
        daily[f"n911_ema_a{int(alpha*10)}"] = (
            daily.groupby("GEOID")["n911_d"]
                 .apply(lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean())
        ).astype("float32")

    # NaN→0, tipler
    fill_int = ["n911_d","n911_prev_1d","n911_roll_3d","n911_roll_7d","n911_roll_30d"]
    daily[fill_int] = daily[fill_int].fillna(0).astype("int32")

    # [3] trend 7v30 (float)
    daily["n911_trend_7v30"] = (daily["n911_roll_7d"] - daily["n911_roll_30d"]).astype("float32")

    return daily

# ========== 911 → Kategori bazlı ==========
def make_911_category_rollings(df911: pd.DataFrame) -> pd.DataFrame:
    # kategori kolonu autodetect
    cat_col = None
    for c in [COL_CAT, "type", "call_type", "category_name", "event_type"]:
        if c in df911.columns:
            cat_col = c; break
    if cat_col is None:
        log("ℹ️ 911 kategori sütunu bulunamadı — kategori rolling atlandı.")
        return pd.DataFrame()

    # tarih hazır değilse üret
    if "date" not in df911.columns:
        use_dt = autodetect_dt_col(df911, COL_DT_911)
        if use_dt is None:
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
    topk = (cat.groupby(cat_col)["n911_cat"].sum().nlargest(CAT_TOPK).index.tolist())
    cat = cat[cat[cat_col].isin(topk)].copy()
    cat["_cat_slug"] = cat[cat_col].apply(_slug)

    # tam takvim per GEOID×cat
    out_list = []
    for geo in cat["GEOID"].unique():
        df_g = cat[cat["GEOID"]==geo].copy()
        days = pd.date_range(df_g["date"].min(), df_g["date"].max(), freq="D").date
        for cg in df_g["_cat_slug"].unique():
            df_gc = df_g[df_g["_cat_slug"]==cg].set_index("date")["n911_cat"]
            df_gc = df_gc.reindex(days, fill_value=0).reset_index().rename(columns={"index":"date","n911_cat":"cnt"})
            df_gc["GEOID"] = geo
            df_gc["_cat_slug"] = cg
            out_list.append(df_gc)
    if not out_list:
        return pd.DataFrame()

    cat_full = pd.concat(out_list, ignore_index=True)
    cat_full = cat_full.sort_values(["GEOID","_cat_slug","date"])

    # sızıntısız rolling
    def _roll_grp(g):
        g["cnt_prev_1d"]  = g["cnt"].shift(1)
        g["cnt_roll_7d"]  = g["cnt"].shift(1).rolling(7,  min_periods=1).sum()
        g["cnt_roll_30d"] = g["cnt"].shift(1).rolling(30, min_periods=1).sum()
        return g
    cat_full = cat_full.groupby(["GEOID","_cat_slug"], group_keys=False).apply(_roll_grp)

    # pivotla kolona (top-K ile sınırlı)
    piv = cat_full.pivot_table(index=["GEOID","date"],
                               columns="_cat_slug",
                               values=["cnt_prev_1d","cnt_roll_7d","cnt_roll_30d"],
                               fill_value=0, aggfunc="first")
    piv.columns = [f"n911_{lvl2}_{lvl1}" for (lvl1,lvl2) in piv.columns.to_flat_index()]
    piv = piv.reset_index()
    return piv

# ========== GRID / EVENTS enrich ==========
def enrich_grid(grid: pd.DataFrame, g911: pd.DataFrame, gcat: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    if not np.issubdtype(pd.Series(out["date"]).dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.merge(g911, on=["GEOID","date"], how="left")
    if not gcat.empty:
        out = out.merge(gcat, on=["GEOID","date"], how="left")
    base_int = ["n911_d","n911_prev_1d","n911_roll_3d","n911_roll_7d","n911_roll_30d"]
    for c in out.columns:
        if c in base_int: out[c] = out[c].fillna(0).astype("int32")
    for c in ["n911_ema_a3","n911_ema_a5","n911_trend_7v30"]:
        if c in out.columns: out[c] = out[c].fillna(0).astype("float32")
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
    feats = g911[["GEOID","date","n911_prev_1d","n911_roll_3d","n911_roll_7d",
                  "n911_roll_30d","n911_ema_a3","n911_ema_a5","n911_trend_7v30"]].drop_duplicates()
    out = out.merge(feats, on=["GEOID","date"], how="left")
    if not gcat.empty:
        out = out.merge(gcat, on=["GEOID","date"], how="left")
    for c in ["n911_prev_1d","n911_roll_3d","n911_roll_7d","n911_roll_30d"]:
        if c in out.columns: out[c] = out[c].fillna(0).astype("int32")
    for c in ["n911_ema_a3","n911_ema_a5","n911_trend_7v30"]:
        if c in out.columns: out[c] = out[c].fillna(0).astype("float32")
    if not gcat.empty:
        for c in gcat.columns:
            if c not in ("GEOID","date"):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

# ========== MAIN ==========
def main():
    log("🚀 update_911_daily.py (revize: dosya bulma + NN fallback + geniş tarih algılama)")
    log(f"🔧 CRIME_DATA_DIR = {CRIME_DATA_DIR}")
    log(f"🔧 FR_911_PATH    = {P_911}")

    df911 = _read_csv(P_911)
    if df911.empty:
        log("ℹ️ 911 verisi yok, işlem atlandı.")
        return 0

    g911 = make_911_daily(df911, COL_G_911, COL_DT_911)
    if g911.empty:
        log("ℹ️ 911 günlük türetilemedi, işlem atlandı.")
        return 0

    gcat = make_911_category_rollings(df911)
    if not gcat.empty:
        log(f"📊 Kategori rolling kolonları: {len(gcat.columns)-2}")

    # ---- GRID ----
    grid = _read_csv(GRID_IN)
    if not grid.empty:
        grid2 = enrich_grid(grid, g911, gcat)
        _save_csv(grid2, GRID_OUT)
    else:
        log("ℹ️ GRID bulunamadı, GRID zenginleştirme atlandı.")

    # ---- EVENTS ----
    ev = _read_csv(EV_IN)
    if not ev.empty:
        ev2 = enrich_events(ev, g911, gcat)
        _save_csv(ev2, EV_OUT)
    else:
        log("ℹ️ EVENTS bulunamadı, EVENTS zenginleştirme atlandı.")

    log("✅ Tamam.")

if __name__ == "__main__":
    main()

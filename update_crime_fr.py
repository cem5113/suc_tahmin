# update_crime_fr.py  (DAILY GEOID×DATE labeling from event-based fr_crime.csv)
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

# =========================
# Ayarlar (ENV ile özelleştirilebilir)
# =========================
EVENTS_PATH = Path(os.getenv("FR_EVENTS_PATH", "sf_crime_y.csv"))   # olay bazlı kaynak
OUT_PATH    = Path(os.getenv("FR_OUT_PATH",   "fr_crime.csv"))      # hedef: günlük grid çıktı
MIRROR_DIR  = Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data"))

# Label kolonu adı
YCOL = os.getenv("FR_YCOL", "Y_label")

# GEOID uzunluğu
GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# Tarih kolon adayları
DATE_CANDS = ["date", "datetime", "incident_date", "incident_datetime", "event_date"]

# Opsiyonel tarih aralığı override (tez/forecast kontrolü için)
FR_START_DATE = os.getenv("FR_START_DATE", "")  # "YYYY-MM-DD"
FR_END_DATE   = os.getenv("FR_END_DATE", "")    # "YYYY-MM-DD"

# =========================
# Yardımcılar
# =========================
def _abs(p: Path) -> Path:
    return p.expanduser().resolve()

def safe_read_csv(p: Path) -> pd.DataFrame:
    p = _abs(p)
    if not p.exists():
        print(f"ℹ️ Bulunamadı: {p}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, low_memory=False)
        print(f"📖 Okundu: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")
        return df
    except Exception as e:
        print(f"⚠️ Okunamadı: {p} → {e}")
        return pd.DataFrame()

def safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Kaydedildi: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

def normalize_geoid(s: pd.Series, L: int) -> pd.Series:
    out = s.astype(str).str.extract(r"(\d+)", expand=False)
    out = out.str[:L].str.zfill(L)
    return out

def find_date_col(df: pd.DataFrame) -> str | None:
    for c in DATE_CANDS:
        if c in df.columns:
            return c
    return None

def add_calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    """date -> takvim alanları"""
    df = df.copy()
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = dt.dt.date
    df["day_of_week"] = dt.dt.weekday.astype("int8")
    df["month"] = dt.dt.month.astype("int8")

    season_map = {
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Fall",10:"Fall",11:"Fall"
    }
    df["season"] = df["month"].map(season_map).astype("category")
    return df

def _parse_override_date(x: str):
    if not x:
        return None
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None

# =========================
# Akış
# =========================
def main() -> int:
    print("📂 CWD:", Path.cwd())
    print("🔧 ENV → FR_EVENTS_PATH:", _abs(EVENTS_PATH))
    print("🔧 ENV → FR_OUT_PATH   :", _abs(OUT_PATH))
    print("🔧 ENV → FR_MIRROR_DIR :", _abs(MIRROR_DIR))
    print("🔧 ENV → FR_YCOL       :", YCOL)
    print("🔧 ENV → GEOID_LEN     :", GEOID_LEN)
    print("🔧 ENV → FR_START_DATE :", FR_START_DATE)
    print("🔧 ENV → FR_END_DATE   :", FR_END_DATE)

    # 1) Olay verisini oku
    events = safe_read_csv(EVENTS_PATH)
    if events.empty:
        print(f"❌ Olay verisi boş veya yok: {_abs(EVENTS_PATH)}")
        return 0

    # 2) GEOID zorunlu
    if "GEOID" not in events.columns:
        raise ValueError("EVENTS içinde GEOID yok. Günlük grid üretilemez.")

    events = events.copy()
    events["GEOID"] = normalize_geoid(events["GEOID"], GEOID_LEN)

    # 3) Tarih kolonunu bul ve normalize et
    dt_col = find_date_col(events)
    if dt_col is None:
        raise ValueError(f"EVENTS içinde tarih kolonu yok. Adaylar: {DATE_CANDS}")

    events["date"] = pd.to_datetime(events[dt_col], errors="coerce")
    events = events.dropna(subset=["date"]).copy()
    events["date"] = events["date"].dt.date

    # 3b) Tarih aralığı override (varsa)
    o_start = _parse_override_date(FR_START_DATE)
    o_end   = _parse_override_date(FR_END_DATE)

    dmin, dmax = events["date"].min(), events["date"].max()
    if o_start is not None:
        dmin = max(dmin, o_start)
    if o_end is not None:
        dmax = min(dmax, o_end)
    if dmin > dmax:
        raise ValueError(f"Override sonrası tarih aralığı ters: {dmin} > {dmax}")

    events = events[(events["date"] >= dmin) & (events["date"] <= dmax)].copy()

    base_len = len(events)
    print(f"🧮 Olay satır sayısı (valid, windowed): {base_len:,}")
    print(f"🧊 Tarih aralığı: {dmin} → {dmax} (gün={ (pd.to_datetime(dmax)-pd.to_datetime(dmin)).days + 1 })")

    # 4) Günlük gözlem: GEOID×date crime_count
    daily_obs = (
        events.groupby(["GEOID", "date"], as_index=False)
              .size()
              .rename(columns={"size": "crime_count"})
    )
    print(f"📌 Günlük gözlem hücre sayısı (Y=1 adayı): {len(daily_obs):,}")

    # 5) Full günlük grid: her GEOID için her gün 1 satır
    geoids = events["GEOID"].dropna().unique()  # evreni events'ten al
    all_days = pd.date_range(dmin, dmax, freq="D").date
    if len(all_days) == 0 or len(geoids) == 0:
        raise ValueError("Full grid evreni boş çıktı (geoids veya days boş).")

    full_grid = pd.MultiIndex.from_product(
        [geoids, all_days],
        names=["GEOID", "date"]
    ).to_frame(index=False)

    print(f"🧱 FULL GRID satır sayısı: {len(full_grid):,}  (GEOID={len(geoids)} × gün={len(all_days)})")

    # 6) Merge ve label
    out = full_grid.merge(daily_obs, on=["GEOID", "date"], how="left")
    out["crime_count"] = pd.to_numeric(out["crime_count"], errors="coerce").fillna(0).astype("int32")
    out[YCOL] = (out["crime_count"] >= 1).astype("int8")

    # 7) Takvim sütunları ekle
    out = add_calendar_cols(out)

    # 8) İz bilgisi
    out["fr_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out["fr_label_rule"]  = "Y=1 if GEOID had >=1 crime that day else 0"
    out["fr_window_start"] = str(dmin)
    out["fr_window_end"]   = str(dmax)

    # deterministik sıralama
    out = out.sort_values(["GEOID", "date"]).reset_index(drop=True)

    # 9) Kaydet & mirror
    safe_save_csv(out, OUT_PATH)
    try:
        _abs(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(_abs(OUT_PATH), _abs(MIRROR_DIR) / _abs(OUT_PATH).name)
        print(f"📦 Mirror kopya: {_abs(MIRROR_DIR) / _abs(OUT_PATH).name}")
    except Exception as e:
        print(f"ℹ️ Mirror kopya atlandı/başarısız: {e}")

    # 10) Dağılım raporu
    vc = out[YCOL].value_counts(normalize=True, dropna=False).mul(100).round(2)
    print("\n📊 Y_label oranları (%):")
    for k, v in vc.items():
        print(f"  {k}: {v}%")

    print(f"🔢 Toplam satır (daily GEOID×date grid): {len(out):,}")
    print(f"✅ Yeni eklenen Y=0 satırları: {(out['crime_count']==0).sum():,}")

    return 0


if __name__ == "__main__":
    try:
        code = main()
        raise SystemExit(code if isinstance(code, int) else 0)
    except Exception as e:
        print(f"⚠️ FR derleme sırasında yakalanmamış hata: {e}")
        raise SystemExit(0)

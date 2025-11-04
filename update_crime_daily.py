#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_crime_daily.py — Günlük özet (GEOID × tarih) + eksik günleri 0 ile doldurma

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, date
import pandas as pd

# ========= Ayarlar (ENV ile değiştirilebilir) =========
CRIME_BASE = Path(os.getenv("CRIME_DATA_DIR", ".")).expanduser()
IN_PATH    = Path(os.getenv("SF_DAILY_IN",  os.getenv("FR_DAILY_IN",  "sf_crime_y.csv")))   # olay bazlı giriş
OUT_PATH   = Path(os.getenv("SF_DAILY_OUT", os.getenv("FR_DAILY_OUT", "daily_crime_00.csv")))  # günlük çıktı
LOCAL_TZ   = os.getenv("SF_DAILY_TZ", os.getenv("FR_DAILY_TZ", "America/Los_Angeles"))

# Opsiyonel: tarih penceresi (YYYY-MM-DD)
FORCE_START = os.getenv("SF_DAILY_START", os.getenv("FR_DAILY_START", "")).strip()
FORCE_END   = os.getenv("SF_DAILY_END",   os.getenv("FR_DAILY_END",   "")).strip()

# Zaman kolonu adayları (ilk bulunan kullanılır); ayrıca date+time fallback'ı var
DT_CANDS = [
    "datetime", "incident_datetime", "created_datetime", "occurred_at",
    "timestamp", "event_time", "t0", "t", "dt", "date", "incident_date"
]

# Adet sayımı için aday kolon (varsa sum, yoksa satır sayısı)
COUNT_CANDS = ["crime_count", "count", "n"]

# Label kolonu (varsa kullanılır ama zorunlu değil)
YCOL = os.getenv("SF_YCOL", os.getenv("FR_YCOL", "Y_label"))

# GEOID uzunluğu (normalize için)
GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ========= Yardımcılar =========
def _abs(p: Path) -> Path:
    return p.expanduser().resolve()

def _resolve_in_out(in_p: Path, out_p: Path) -> tuple[Path, Path]:
    """
    Çift/üçlü klasör tekrarını önler.
    - IN: eğer mutlak değilse önce CWD’de bakar; yoksa CRIME_BASE/in_p’ye bakar.
    - OUT: eğer mutlak değilse ve parent '.' ise CRIME_BASE/out_p altında yazar.
    """
    # IN
    if not in_p.is_absolute():
        if in_p.exists():
            pass  # CWD'de var
        elif (CRIME_BASE / in_p).exists():
            in_p = CRIME_BASE / in_p
        # değilse olduğu gibi kalsın (hata mesajı basacağız)

    # OUT
    if not out_p.is_absolute() and (str(out_p.parent) in {".", ""}):
        out_p = CRIME_BASE / out_p

    return in_p, out_p

def _autofind_input(p: Path) -> Path:
    """
    IN_PATH bulunamazsa otomatik alternatifleri sırayla dener:
    - verilen ad
    - y→csv / csv→y değişimi
    - crime_prediction_data/ öneki (tek ve çift kat)
    - CRIME_BASE + yukarıdakiler
    Bulunursa ilk eşleşeni döndürür; yoksa orijinali döndürür.
    """
    names = []
    s = str(p)

    # 1) Verilen
    names.append(s)

    # 2) y/csv swap
    if s.endswith("sf_crime_y.csv"):
        names.append(s.replace("sf_crime_y.csv", "sf_crime.csv"))
    elif s.endswith("sf_crime.csv"):
        names.append(s.replace("sf_crime.csv", "sf_crime_y.csv"))

    # 3) crime_prediction_data önekleri
    base = "crime_prediction_data"
    if not s.startswith(f"{base}/"):
        names.append(f"{base}/{s}")
        names.append(f"{base}/{base}/{s}")

    # 4) swap + önek kombinasyonları
    swaps = []
    for n in list(names):
        if n.endswith("sf_crime_y.csv"):
            swaps.append(n.replace("sf_crime_y.csv", "sf_crime.csv"))
        elif n.endswith("sf_crime.csv"):
            swaps.append(n.replace("sf_crime.csv", "sf_crime_y.csv"))
    names.extend(swaps)

    # 5) CRIME_BASE ile mutlaklaştırılan varyantlar
    cands: list[Path] = []
    for n in names:
        cands.append(Path(n))
        cands.append(CRIME_BASE / n)

    seen = set()
    for cand in cands:
        c = _abs(cand)
        if c in seen:
            continue
        seen.add(c)
        if c.exists() and c.is_file():
            return cand
    return p

def _read_any(p: Path) -> pd.DataFrame:
    p = _abs(p)
    if not p.exists():
        print(f"❌ Girdi bulunamadı: {p}")
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)
    print(f"📖 Okundu: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")
    return df

def _detect_col(cands: list[str], cols: pd.Index) -> str | None:
    for c in cands:
        if c in cols:
            return c
    return None

def _ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" not in df.columns:
        raise SystemExit("❌ 'GEOID' kolonu zorunlu ve bulunamadı.")
    out = df.copy()
    out["GEOID"] = (
        out["GEOID"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str[:GEOID_LEN]
        .str.zfill(GEOID_LEN)
    )
    return out

def _to_local_date_from_any(df: pd.DataFrame, dcol: str, tz: str) -> pd.Series:
    """
    df[dcol] bir datetime, tarih stringi veya sadece 'date' (yyyy-mm-dd) olabilir.
    Eğer sadece 'date' varsa TZ dönüşümü yapmadan tarihi döndürür.
    Eğer 'date' + 'time' kolonları varsa ikisini birleştirir.
    Çıktı: YYYY-MM-DD string
    """
    # date + time birleşimi
    if dcol in ("date", "incident_date") and "time" in df.columns:
        dt = pd.to_datetime(
            df[dcol].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
            errors="coerce",
            utc=True,
        )
        try:
            dt = dt.dt.tz_convert(tz)
        except Exception:
            dt = pd.to_datetime(
                df[dcol].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
                errors="coerce"
            ).dt.tz_localize("UTC").dt.tz_convert(tz)
        return dt.dt.strftime("%Y-%m-%d")

    # sadece 'date' veya 'incident_date'
    if dcol in ("date", "incident_date"):
        return pd.to_datetime(df[dcol], errors="coerce").dt.strftime("%Y-%m-%d")

    # datetime benzeri
    dt = pd.to_datetime(df[dcol], errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert(tz)
    except Exception:
        dt = pd.to_datetime(df[dcol], errors="coerce").dt.tz_localize("UTC").dt.tz_convert(tz)
    return dt.dt.strftime("%Y-%m-%d")

def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

# ========= Çekirdek işlev =========
def build_daily(df_src: pd.DataFrame) -> pd.DataFrame:
    # GEOID normalize
    df = _ensure_geoid(df_src)

    # Zaman + count
    dcol = _detect_col(DT_CANDS, df.columns)
    if dcol is None:
        raise SystemExit(f"❌ Zaman kolonu bulunamadı. Adaylar: {DT_CANDS}")

    ccol = _detect_col(COUNT_CANDS, df.columns)  # opsiyonel

    # Yerel tarihe indir (event_date)
    df = df.copy()
    df["event_date"] = _to_local_date_from_any(df, dcol, LOCAL_TZ)

    # Günlük agregasyon
    grp_keys = ["GEOID", "event_date"]
    if ccol:
        daily = df.groupby(grp_keys, as_index=False).agg(daily_count=(ccol, "sum"))
    else:
        daily = df.groupby(grp_keys, as_index=False).agg(daily_count=("GEOID", "size"))

    # Y_day üretimi
    if YCOL in df.columns:
        y_any = (
            df.groupby(grp_keys, as_index=False)
              .agg(_y_any=(YCOL, lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).any())))
        )
        daily = daily.merge(y_any, on=grp_keys, how="left")
        daily.rename(columns={"_y_any": "Y_day"}, inplace=True)
        daily["Y_day"] = daily["Y_day"].fillna((daily["daily_count"] > 0).astype("int8")).astype("int8")
    else:
        daily["Y_day"] = (pd.to_numeric(daily["daily_count"], errors="coerce").fillna(0) > 0).astype("int8")

    # --------- Eksik günleri 0'la doldur (tam ızgara) ---------
    all_geoids = daily["GEOID"].dropna().unique()

    existing_dates = pd.to_datetime(daily["event_date"], errors="coerce")
    auto_start = existing_dates.min().date() if not existing_dates.isna().all() else None
    auto_end   = existing_dates.max().date() if not existing_dates.isna().all() else None

    d_start = _parse_date(FORCE_START) or auto_start
    d_end   = _parse_date(FORCE_END)   or auto_end
    if d_start is None or d_end is None:
        raise SystemExit("❌ Tarih aralığı tespit edilemedi (veride hiç geçerli tarih yok).")

    all_dates = pd.date_range(start=d_start, end=d_end, freq="D").strftime("%Y-%m-%d")

    full = pd.MultiIndex.from_product(
        [all_geoids, all_dates],
        names=["GEOID", "event_date"]
    ).to_frame(index=False)

    daily_full = (
        full.merge(daily, on=["GEOID", "event_date"], how="left")
            .fillna({"daily_count": 0, "Y_day": 0})
    )

    daily_full["daily_count"] = pd.to_numeric(daily_full["daily_count"], errors="coerce").fillna(0).astype("int32")
    daily_full["Y_day"] = pd.to_numeric(daily_full["Y_day"], errors="coerce").fillna(0).astype("int8")
    daily_full["sf_daily_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    daily_full["sf_daily_tz"] = LOCAL_TZ

    daily_full = daily_full.loc[:, ~daily_full.columns.duplicated()].copy()
    return daily_full

# ========= Kaydet =========
def _save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Yazıldı: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

# ========= CLI =========
def main() -> int:
    # IN için otomatik bul
    in_guess = _autofind_input(IN_PATH)
    if in_guess != IN_PATH:
        print(f"🔎 IN otomatik bulundu → {in_guess}")
        IN = in_guess
    else:
        IN = IN_PATH

    # CWD ve path çözümlemesi
    in_path_resolved, out_path_resolved = _resolve_in_out(IN, OUT_PATH)

    print("📂 CWD:", Path.cwd())
    print("🔧 CRIME_BASE:", _abs(CRIME_BASE))
    print("🔧 SF_DAILY_IN :", _abs(in_path_resolved))
    print("🔧 SF_DAILY_OUT:", _abs(out_path_resolved))
    print("🔧 SF_DAILY_TZ :", LOCAL_TZ)
    if FORCE_START or FORCE_END:
        print(f"🔧 FORCE window: start={FORCE_START or 'auto'} end={FORCE_END or 'auto'}")

    src = _read_any(in_path_resolved)
    if src.empty:
        return 0

    # Hızlı doğrulama: kritik kolonlar
    must_have = {"GEOID"}
    if not must_have.issubset(set(src.columns)):
        print(f"⚠️ Uyarı: Giriş verisinde eksik kolon(lar): {must_have - set(src.columns)}")

    daily = build_daily(src)
    _save_csv(daily, out_path_resolved)

    # Kısa özet
    y1 = int((daily["Y_day"] == 1).sum())
    y0 = int((daily["Y_day"] == 0).sum())
    tot = len(daily)
    pct1 = round(100 * y1 / tot, 2) if tot else 0.0
    print(f"📊 Günlük satır: {tot:,} | Y_day=1: {y1:,} (%{pct1}) | Y_day=0: {y0:,}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

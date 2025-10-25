#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
post_patrol.py
- risk_hourly.csv'yi okur
- ENV'den PATROL_HORIZON_DAYS ve PATROL_TOP_K'yi alır
- (opsiyonel) yarin.csv ve week.csv'yi date üzerinden birleştirir
- Yarından başlayarak ufuk boyunca her saat için en riskli TOP_K GEOID'i seçer
- Çıktı: crime_prediction_data/patrol_recs_multi.csv
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

# --- Ortam / yollar ---
CRIME_DATA_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
CRIME_DIR = Path(CRIME_DATA_DIR)
CRIME_DIR.mkdir(parents=True, exist_ok=True)

RISK_PATH = CRIME_DIR / "risk_hourly.csv"
OUT_PATH  = CRIME_DIR / "patrol_recs_multi.csv"

# --- Parametreler (ENV) ---
SF_TZ = ZoneInfo("America/Los_Angeles")
HORIZON_DAYS = int(os.getenv("PATROL_HORIZON_DAYS", "3"))  # ufuk (gün)
TOP_K        = int(os.getenv("PATROL_TOP_K", "50"))        # saat başına öneri sayısı

# --- Yardımcılar ---
def _lower_map(cols):
    return {c.lower(): c for c in cols}

def _ensure_date_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Esnek tarih/saat kolonu tespiti:
    - 'date' + 'hour' varsa direkt kullanır
    - 'timestamp' / 'datetime' / 'dt' / 'ts' varsa bunlardan date+hour üretir
    - 'day' veya 'ds' varsa 'date' gibi kabul eder
    """
    low = _lower_map(df.columns)

    # Tarih kolonu adayı
    date_col = None
    for cand in ("date", "day", "ds"):
        if cand in low:
            date_col = low[cand]
            break

    # Saat kolonu adayı
    hour_col = None
    for cand in ("hour", "hr", "hh"):
        if cand in low:
            hour_col = low[cand]
            break

    # Zaman damgası kolonu adayı
    ts_col = None
    for cand in ("timestamp", "datetime", "dt", "ts"):
        if cand in low:
            ts_col = low[cand]
            break

    if ts_col and (date_col is None or hour_col is None):
        # Zaman damgasından date + hour türet
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
        # timezone yoksa SF olarak kabul edelim (tahmin)
        try:
            # Eğer aware değilse tz_localize yapmaz; yoksa convert
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize(SF_TZ)
            else:
                ts = ts.dt.tz_convert(SF_TZ)
        except Exception:
            pass
        df["date"] = ts.dt.strftime("%Y-%m-%d")
        df["hour"] = ts.dt.hour
        date_col, hour_col = "date", "hour"

    # Eğer hala date yoksa hata
    if date_col is None:
        raise SystemExit("risk_hourly.csv içinde tarih kolonu bulunamadı (date/day/ds ya da timestamp türetilemedi).")
    if hour_col is None:
        # Saat yoksa 0 kabul edelim (günlük aggr. durumları için)
        df["hour"] = 0
        hour_col = "hour"

    # Normalize tipler
    df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
    # date'i iso string halinde tutalım
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype("string")

    # risk skor kolonu tespiti
    risk_col = None
    for cand in ("risk_score", "prob", "score", "p", "yhat"):
        if cand in low:
            risk_col = low[cand]
            break
    if risk_col is None:
        # olası tek bir sayısal skoru bulmayı deneyelim
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        # hour ve muhtemel id/sayac kolonlarını çıkar
        bad = set([hour_col])
        cand_list = [c for c in numeric_cols if c not in bad]
        if not cand_list:
            raise SystemExit("risk_hourly.csv içinde risk skoru için sayısal bir kolon bulunamadı.")
        risk_col = cand_list[0]

    # geoid kolonu
    geoid_col = None
    for cand in ("geoid", "GEOID", "id", "cell_id", "geo"):
        if cand.lower() in low:
            geoid_col = low[cand.lower()]
            break
    if geoid_col is None:
        raise SystemExit("risk_hourly.csv içinde GEOID kolonu bulunamadı (geoid/GEOID/id...).")

    # Standart isimlerle döndür
    out = df.rename(columns={
        date_col: "date",
        hour_col: "hour",
        risk_col: "risk_score",
        geoid_col: "geoid",
    }).copy()

    # türler
    out["date"] = out["date"].astype(str)  # "YYYY-MM-DD"
    out["hour"] = out["hour"].astype(int)
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    out = out.dropna(subset=["risk_score"])
    return out

def _merge_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    yarin.csv ve week.csv varsa, 'date' üstünden left-join yapar.
    Çakışan kolon adlarını otomatik ayırmak için suffix kullanır.
    Yoksa dokunmaz.
    """
    ypath = CRIME_DIR / "yarin.csv"
    wpath = CRIME_DIR / "week.csv"
    merged = df

    def _safe_merge(left, right_path, suffix):
        if right_path.exists():
            try:
                r = pd.read_csv(right_path)
                if "date" in r.columns:
                    r["date"] = pd.to_datetime(r["date"], errors="coerce").dt.date.astype("string")
                return left.merge(r, on="date", how="left", suffixes=("", suffix))
            except Exception:
                return left
        return left

    merged = _safe_merge(merged, ypath, "_t")
    merged = _safe_merge(merged, wpath, "_w")
    return merged

def main():
    if not RISK_PATH.exists():
        raise SystemExit(f"risk_hourly.csv bulunamadı: {RISK_PATH}")

    print(f"[post_patrol] HORIZON_DAYS={HORIZON_DAYS}, TOP_K={TOP_K}")
    print(f"[post_patrol] RISK_PATH={RISK_PATH}")

    # 1) Risk verisini oku ve standartlaştır
    risk = pd.read_csv(RISK_PATH, low_memory=False)
    risk = _ensure_date_hour(risk)

    # 2) Ufku hesapla (yarından başlayarak)
    today_sf  = datetime.now(SF_TZ).date()
    start_day = today_sf + timedelta(days=1)
    end_day   = start_day + timedelta(days=HORIZON_DAYS - 1)

    mask = (risk["date"] >= start_day.isoformat()) & (risk["date"] <= end_day.isoformat())
    risk_h = risk.loc[mask].copy()
    if risk_h.empty:
        # veri yoksa son çare: mevcut günleri kullan (geliştirme/deneme kolaylığı)
        risk_h = risk.copy()

    # 3) (Opsiyonel) hava verisini ekle
    risk_h = _merge_weather(risk_h)

    # 4) Saat saat sıralama ve TOP_K seçimi
    out_parts = []
    for (d, h), g in risk_h.groupby(["date", "hour"], sort=True):
        g = g.sort_values("risk_score", ascending=False).head(TOP_K).copy()
        if g.empty:
            continue
        g["rank"] = range(1, len(g) + 1)
        out_parts.append(g)

    if not out_parts:
        # hiç parça çıkmadıysa boş CSV yazalım
        pd.DataFrame(columns=["date", "hour", "geoid", "risk_score", "rank"]).to_csv(OUT_PATH, index=False)
        print(f"[post_patrol] UYARI: Seçim çıkmadı, boş dosya yazıldı → {OUT_PATH}")
        return

    patrol = pd.concat(out_parts, ignore_index=True)
    # Sıralı çıktı (d, h, rank)
    patrol = patrol.sort_values(["date", "hour", "rank"]).reset_index(drop=True)

    # 5) Yaz
    patrol.to_csv(OUT_PATH, index=False)
    print(f"[post_patrol] ✅ Yazıldı → {OUT_PATH} | rows={len(patrol)}")
    try:
        print(patrol.head(10).to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()

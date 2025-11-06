# scripts/update_311.py
import os, re, time, shutil, zipfile, requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from urllib.parse import quote
from pathlib import Path

# ============================== HELPERS & ENV ===============================

def log(msg: str): 
    print(msg, flush=True)

def save_atomic(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# Sanity: tek bir SAVE_DIR
SAVE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).expanduser().resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACT_ZIP = Path(os.getenv("ARTIFACT_ZIP", "artifact/sf-crime-pipeline-output.zip")).expanduser()
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifact_unzipped")).expanduser()

RAW_311_NAME_Y = os.getenv("RAW_311_NAME", "sf_311_last_5_years_y.csv")
AGG_BASENAME   = os.getenv("AGG_311_NAME", "sf_311_last_5_years.csv")
AGG_ALIAS      = os.getenv("AGG_311_ALIAS", "sf_311_last_5_years_3h.csv")

# Legacy adlar (workflow geriye dönük isim arıyor olabilir)
LEGACY_311_Y = os.getenv("LEGACY_311_Y", "sf_311_last_5_year_y.csv")
LEGACY_311   = os.getenv("LEGACY_311",   "sf_311_last_5_year.csv")

# Socrata dataset & limits
DATASET_BASE = os.getenv("SF311_DATASET", "https://data.sfgov.org/resource/vw6y-z8j6.json")
SOCRATA_APP_TOKEN = os.getenv("SOCS_APP_TOKEN", "").strip()
PAGE_LIMIT   = int(os.getenv("SF_SODA_PAGE_LIMIT", "50000"))
SODA_TIMEOUT = int(os.getenv("SODA_TIMEOUT", "90"))
SODA_RETRIES = int(os.getenv("SODA_RETRIES", "5"))
SLEEP_SEC    = float(os.getenv("SF_SODA_THROTTLE_SEC", "0.25"))
CHUNK_DAYS   = int(os.getenv("SF311_CHUNK_DAYS", "31"))
MAX_PAGES_PER_CHUNK = int(os.getenv("SF311_MAX_PAGES_PER_CHUNK", "40"))
MAX_CONSEC_EMPTY_CHUNKS = int(os.getenv("SF311_MAX_EMPTY_CHUNKS", "8"))

# Window
TODAY = datetime.utcnow().date()
FIVE_YEARS = 5 * 365
DEFAULT_START = TODAY - timedelta(days=FIVE_YEARS)
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "0"))

# GEO
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
GEOJSON_NAME = os.getenv("SF_BLOCKS_GEOJSON", "sf_census_blocks_with_population.geojson")

# ============================== ZIP (ARTIFACT) ==============================

def ensure_unzip_artifact():
    if ARTIFACT_ZIP.exists():
        try:
            if ARTIFACT_DIR.exists():
                shutil.rmtree(ARTIFACT_DIR)
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ARTIFACT_ZIP, "r") as z:
                z.extractall(ARTIFACT_DIR)
            log(f"📦 Artifact açıldı: {ARTIFACT_ZIP} → {ARTIFACT_DIR}")
        except Exception as e:
            log(f"⚠️ Artifact açılamadı: {e}")

# ============================== GEO UTILS ===================================

def normalize_geoid(series: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

def normalize_geoid_11(x):
    if pd.isna(x): return pd.NA
    digits = re.sub(r"\D", "", str(x))
    return digits[:11] if len(digits) >= 11 else pd.NA

def find_geojson_candidates():
    cands = []
    # 1) Artifact
    cands.append(ARTIFACT_DIR / GEOJSON_NAME)
    # 2) SAVE_DIR
    cands.append(SAVE_DIR / GEOJSON_NAME)
    # 3) repo kök
    cands.append(Path(".").resolve() / GEOJSON_NAME)
    # 4) SAVE_DIR/crime_prediction_data
    cands.append((SAVE_DIR / "crime_prediction_data" / GEOJSON_NAME))
    return [Path(c) for c in cands]

def ensure_blocks_gdf():
    for cand in find_geojson_candidates():
        if cand.exists():
            gdf = gpd.read_file(cand)
            if "GEOID" not in gdf.columns:
                poss = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
                if poss:
                    gdf["GEOID"] = gdf[poss[0]].astype(str)
            if "GEOID" not in gdf.columns:
                continue
            gdf["TRACT11"] = gdf["GEOID"].apply(normalize_geoid_11)
            gdf = gdf[["TRACT11", "geometry"]].dropna(subset=["TRACT11"])
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif str(gdf.crs).lower() not in ("epsg:4326", "wgs84", "wgs 84"):
                gdf = gdf.to_crs(epsg=4326)
            log(f"🧭 GEOJSON: {cand}")
            return gdf
    log("⚠️ GEOJSON yok; GEOID eşleme atlanabilir.")
    return None

def geotag_to_geoid11(df_new: pd.DataFrame) -> pd.DataFrame:
    df = df_new.copy()
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
    if "longitude" not in df.columns and "long" in df.columns:
        df["longitude"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        df_new["GEOID"] = pd.NA
        return df_new

    blocks = ensure_blocks_gdf()
    if blocks is None:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_pts = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    try:
        joined = gpd.sjoin(gdf_pts, blocks, how="left", predicate="within")
    except Exception:
        try:
            joined = gpd.sjoin_nearest(gdf_pts, blocks, how="left", max_distance=5)
        except Exception:
            gdf_pts["GEOID"] = pd.NA
            return pd.DataFrame(gdf_pts.drop(columns=["geometry"]))
    out = pd.DataFrame(joined.drop(columns=["geometry"]))
    out.rename(columns={"TRACT11": "GEOID"}, inplace=True)
    out["GEOID"] = out["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:11]
    return out

# ============================== IO CANDIDATES ================================

def candidate_dirs():
    # Öncelik: artifact → SAVE_DIR → repo kök → SAVE_DIR/crime_prediction_data
    return [
        ARTIFACT_DIR,
        SAVE_DIR,
        Path(".").resolve(),
        SAVE_DIR / "crime_prediction_data",
    ]

def find_existing(path_name: str) -> Path | None:
    for base in candidate_dirs():
        p = (base / path_name).resolve()
        if p.exists():
            return p
    return None

def read_csv_safe(p: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(p, **kw)
    except Exception as e:
        log(f"⚠️ CSV okunamadı ({p}): {e}")
        return pd.DataFrame()

# ============================== SOCRATA =====================================

def socrata_get(session: requests.Session, url, params):
    headers = {"Accept": "application/json"}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN
    last = None
    for i in range(SODA_RETRIES):
        try:
            r = session.get(url, params=params, headers=headers, timeout=SODA_TIMEOUT)
            if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"status={r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(max(SLEEP_SEC, SLEEP_SEC * (2 ** i)))
    raise last

def download_by_date_chunks(start_date: datetime.date) -> pd.DataFrame:
    log(f"🧩 İndirme modu: DATE-CHUNKS ({CHUNK_DAYS}gün) + paging")
    session = requests.Session()
    police_filter = "(agency_responsible like '%Police%' OR agency_responsible like '%SFPD%')"
    cols = ",".join([
        "service_request_id","requested_datetime","lat","long",
        "service_name","service_subtype","agency_responsible"
    ])

    all_chunks = []
    consec_empty = 0
    cur = start_date
    end = TODAY

    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        start_iso = f"{cur.isoformat()}T00:00:00.000"
        end_iso   = f"{chunk_end.isoformat()}T23:59:59.999"
        where = f"$where=requested_datetime between '{start_iso}' and '{end_iso}' AND {police_filter}"
        log(f"⛏️  {cur} → {chunk_end} aralığı çekiliyor…")

        offset, pages = 0, 0
        chunk_rows = []
        while True:
            params = {"$select": cols, "$order": "requested_datetime ASC", "$limit": PAGE_LIMIT, "$offset": offset}
            q = f"{DATASET_BASE}?{quote(where, safe='=&()>< ')}"
            try:
                data = socrata_get(session, q, params)
            except Exception as e:
                log(f"❌ Chunk hata ({cur}→{chunk_end}, offset={offset}): {e} → chunk geçiliyor.")
                break

            df = pd.DataFrame(data)
            if df.empty: break
            if pages == 0: log("   • kolonlar:", list(df.columns))
            chunk_rows.append(df)
            offset += len(df); pages += 1
            log(f"   + {offset} kayıt (sayfa={pages})")
            if len(df) < PAGE_LIMIT or pages >= MAX_PAGES_PER_CHUNK:
                if pages >= MAX_PAGES_PER_CHUNK:
                    log(f"   ↪️ MAX_PAGES_PER_CHUNK={MAX_PAGES_PER_CHUNK} doldu, chunk kesildi.")
                break
            time.sleep(SLEEP_SEC)

        if chunk_rows:
            consec_empty = 0
            all_chunks.append(pd.concat(chunk_rows, ignore_index=True))
            log(f"✅ Chunk bitti: satır={sum(len(x) for x in chunk_rows)}")
        else:
            consec_empty += 1
            log(f"ℹ️ Chunk boş (ardışık boş={consec_empty}).")
            if consec_empty >= MAX_CONSEC_EMPTY_CHUNKS and cur > start_date:
                log("⏹️ Çok fazla ardışık boş; erken durdurma.")
                break

        cur = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_SEC)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

# ============================== MAIN ========================================

def main():
    log(f"🔎 CWD: {os.getcwd()}")
    log(f"🔎 SAVE_DIR: {SAVE_DIR}")

    # 0) Artifact varsa aç
    ensure_unzip_artifact()

    # 1) Mevcut ÖZET dosyasını en öncelikli yerden bul (artifact → SAVE_DIR → . → SAVE_DIR/crime_prediction_data)
    agg_existing_path = find_existing(AGG_BASENAME)
    if not agg_existing_path:
        # alias’ı da dene
        agg_existing_path = find_existing(AGG_ALIAS) or find_existing("sf_311_last_5_years_3h.csv")
    raw_existing_path = find_existing(RAW_311_NAME_Y) or find_existing(LEGACY_311_Y) or find_existing(LEGACY_311)

    # 2) Başlangıç tarihi kararını özetten yap (varsa), yoksa hamdan yap; hiçbiri yoksa default/backfill
    start_mode = "full-5y"
    if BACKFILL_DAYS > 0:
        start_date = TODAY - timedelta(days=BACKFILL_DAYS)
        start_mode = "backfill"
    else:
        if agg_existing_path:
            df_agg_old = read_csv_safe(agg_existing_path, dtype={"GEOID": str}, low_memory=False)
            if not df_agg_old.empty and "date" in df_agg_old.columns:
                last_date = pd.to_datetime(df_agg_old["date"], errors="coerce").dt.date.max()
                if pd.notna(last_date):
                    start_date = last_date + timedelta(days=1)
                    start_mode = f"incremental-from-agg({last_date})"
                else:
                    start_date = DEFAULT_START
            else:
                start_date = DEFAULT_START
        elif raw_existing_path:
            df_raw_old = read_csv_safe(raw_existing_path, dtype={"GEOID": str}, low_memory=False)
            mx = pd.to_datetime(df_raw_old.get("datetime"), errors="coerce", utc=True).max()
            if pd.notna(mx):
                start_date = mx.date() + timedelta(days=1)
                start_mode = f"incremental-from-raw({mx.date()})"
            else:
                start_date = DEFAULT_START
        else:
            start_date = DEFAULT_START

    log(f"📌 Mod: {start_mode} | start={start_date} | window ≥ {DEFAULT_START}")

    # 3) Yeni veri indir (tarih-chunk)
    df_new = pd.DataFrame()
    if start_date <= TODAY:
        df_new = download_by_date_chunks(start_date)
    if df_new.empty:
        log("ℹ️ Yeni 311 kaydı bulunamadı veya erişilemedi.")
    else:
        log(f"➕ Yeni indirilen satır: {len(df_new):,}")
        df_new = df_new.rename(columns={
            "service_request_id": "id",
            "requested_datetime": "datetime",
            "service_name": "category",
            "service_subtype": "subcategory"
        })
        df_new["datetime"] = pd.to_datetime(df_new["datetime"], errors="coerce", utc=True)
        df_new["date"] = pd.to_datetime(df_new["datetime"]).dt.date
        df_new["time"] = pd.to_datetime(df_new["datetime"]).dt.time

    # 4) HAM (_y) güncelle (varsa ekle, yoksa yaz)
    raw_out_path = SAVE_DIR / RAW_311_NAME_Y
    if not df_new.empty:
        df_new_geo = geotag_to_geoid11(df_new)
        keep = ["id","datetime","date","time","lat","long","category","subcategory",
                "agency_responsible","latitude","longitude","GEOID"]
        for c in keep:
            if c not in df_new_geo.columns:
                df_new_geo[c] = pd.NA
        df_new_geo = df_new_geo[keep]
        df_new_geo["GEOID"] = normalize_geoid(df_new_geo["GEOID"], DEFAULT_GEOID_LEN)

        if raw_existing_path:
            df_raw_old = read_csv_safe(raw_existing_path, dtype={"GEOID": str}, low_memory=False)
            df_raw = pd.concat([df_raw_old, df_new_geo], ignore_index=True)
        else:
            df_raw = df_new_geo

        # Tekilleştir & sırala & pencere
        df_raw["id"] = df_raw["id"].astype(str)
        df_raw.drop_duplicates(subset=["id"], keep="last", inplace=True)
        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce", utc=True)
        df_raw.sort_values("datetime", inplace=True)

        min_date = start_date if BACKFILL_DAYS > 0 else DEFAULT_START
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce").dt.date
        df_raw = df_raw[df_raw["date"] >= min_date]

        save_atomic(df_raw, raw_out_path.as_posix())
        # legacy aliaslar da güncellensin
        for alias in (LEGACY_311_Y, LEGACY_311):
            save_atomic(df_raw, (SAVE_DIR / alias).as_posix())

        mx = pd.to_datetime(df_raw["datetime"], errors="coerce").max()
        log(f"✅ Ham (_y) yazıldı: {raw_out_path} | rows={len(df_raw):,} | max_date={mx.date() if pd.notna(mx) else 'NA'}")
        try:
            log(df_raw.head(3).to_string(index=False))
        except Exception:
            pass
    else:
        # Yeni yoksa ama ham dosya yoksa şemalı boş üret
        if not raw_out_path.exists():
            empty_cols = ["id","datetime","date","time","lat","long",
                          "category","subcategory","agency_responsible","latitude","longitude","GEOID"]
            save_atomic(pd.DataFrame(columns=empty_cols), raw_out_path.as_posix())
            for alias in (LEGACY_311_Y, LEGACY_311):
                save_atomic(pd.DataFrame(columns=empty_cols), (SAVE_DIR / alias).as_posix())
            log("ℹ️ Ham (_y) için şemalı boş dosya yazıldı (yeni kayıt yoktu).")

    # 5) 3 SAATLİK ÖZET — append veya full yaz
    agg_out = SAVE_DIR / AGG_BASENAME
    agg_alias_out = SAVE_DIR / AGG_ALIAS
    if not df_new.empty:
        # Yalnızca yeni blok için özet üret
        df_ok = df_new.copy()
        h = pd.to_datetime(df_ok["datetime"], errors="coerce").dt.hour.fillna(0).astype(int)
        start_h = (h // 3) * 3
        end_h = (start_h + 3) % 24
        df_ok["hour_range"] = start_h.astype(str).str.zfill(2) + "-" + end_h.astype(str).str.zfill(2)

        df_new_geo = geotag_to_geoid11(df_ok)
        if "GEOID" not in df_new_geo.columns or df_new_geo["GEOID"].isna().all():
            # GEOID üretilememişse yeni özet boş geç, ama dosyaları en azından dokunma
            new_grouped = pd.DataFrame(columns=["GEOID","date","hour_range","311_request_count"])
        else:
            new_grouped = (
                df_new_geo.dropna(subset=["GEOID"])
                          .groupby(["GEOID","date","hour_range"])
                          .size()
                          .reset_index(name="311_request_count")
            )
            new_grouped["GEOID"] = normalize_geoid(new_grouped["GEOID"], DEFAULT_GEOID_LEN)

        if agg_existing_path:
            df_agg_old = read_csv_safe(agg_existing_path, dtype={"GEOID": str}, low_memory=False)
        else:
            df_agg_old = pd.DataFrame(columns=["GEOID","date","hour_range","311_request_count"])

        # append & dedup
        df_agg_old["date"] = pd.to_datetime(df_agg_old.get("date"), errors="coerce").dt.date
        new_grouped["date"] = pd.to_datetime(new_grouped.get("date"), errors="coerce").dt.date
        combined = pd.concat([df_agg_old, new_grouped], ignore_index=True)
        combined.drop_duplicates(subset=["GEOID","date","hour_range"], keep="last", inplace=True)
        combined.sort_values(["GEOID","date","hour_range"], inplace=True)

        save_atomic(combined, agg_out.as_posix())
        if AGG_ALIAS and AGG_ALIAS != AGG_BASENAME:
            save_atomic(combined, agg_alias_out.as_posix())
        log(f"📁 Özet güncellendi: {agg_out} (rows={len(combined):,})")
        try:
            log(combined.tail(5).to_string(index=False))
        except Exception:
            pass
    else:
        # Yeni yok; ama özet hiç yoksa şemalı boş oluştur
        if not agg_out.exists():
            empty = pd.DataFrame(columns=["GEOID","date","hour_range","311_request_count"])
            save_atomic(empty, agg_out.as_posix())
            if AGG_ALIAS and AGG_ALIAS != AGG_BASENAME:
                save_atomic(empty, agg_alias_out.as_posix())
            log("ℹ️ Özet için şemalı boş dosya yazıldı (yeni kayıt yoktu).")

    # 6) 311 ÖZET + SUÇ merge (sf_crime_01.csv → sf_crime_02.csv)
    try:
        crime_01 = SAVE_DIR / "sf_crime_01.csv"
        if not crime_01.exists():
            log(f"ℹ️ {crime_01} yok → 311 merge atlandı.")
            return

        log("🔗 sf_crime_01 ile 311 özet birleştiriliyor…")
        crime = read_csv_safe(crime_01, dtype={"GEOID": str}, low_memory=False)

        # Özet için en taze yolu yeniden çöz (az önce yazmış olabiliriz)
        summary_path = agg_out if agg_out.exists() else (find_existing(AGG_BASENAME) or find_existing(AGG_ALIAS))
        if not summary_path or not Path(summary_path).exists():
            log("⚠️ 311 özet yok → PASSTHROUGH (311_request_count=0)")
            crime["311_request_count"] = 0
            save_atomic(crime, (SAVE_DIR / "sf_crime_02.csv").as_posix())
            log("✅ Passthrough yazıldı.")
            return

        summary = read_csv_safe(Path(summary_path), dtype={"GEOID": str}, low_memory=False)

        # GEOID uzunluk hizalama
        def _mode_len(s: pd.Series) -> int:
            s2 = s.dropna().astype(str).str.extract(r"(\d+)")[0]
            return int(s2.str.len().mode().iat[0]) if len(s2) else DEFAULT_GEOID_LEN
        tgt_len = min(_mode_len(crime["GEOID"]), _mode_len(summary["GEOID"]))
        def _left(series, n):
            s = series.astype(str).str.extract(r"(\d+)")[0]
            return s.str[:n]
        crime["GEOID"]   = _left(crime["GEOID"],   tgt_len)
        summary["GEOID"] = _left(summary["GEOID"], tgt_len)

        if "hour_range" not in summary.columns:
            if "hr_key" in summary.columns:
                hr = pd.to_numeric(summary["hr_key"], errors="coerce").fillna(0).astype(int)
                summary["hour_range"] = hr.astype(str).str.zfill(2) + "-" + (hr + 3).astype(str).str.zfill(2)
            else:
                summary["hour_range"] = "00-03"

        summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date
        hrp = summary["hour_range"].astype(str).str.extract(r"(\d{1,2})\s*-\s*(\d{1,2})")
        summary["hr_key"] = pd.to_numeric(hrp[0], errors="coerce").fillna(0).astype(int)
        _dt = pd.to_datetime(summary["date"], errors="coerce")
        summary["day_of_week"] = _dt.dt.weekday
        summary["month"] = _dt.dt.month
        _smap = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                 6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
        summary["season"] = summary["month"].map(_smap)

        # crime tarafı saat anahtarı
        if "hour_range" not in crime.columns:
            if "event_hour" not in crime.columns:
                raise ValueError("❌ sf_crime_01.csv için hour_range/event_hour yok.")
            hr = (pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int) // 3) * 3
            crime["hour_range"] = hr.astype(str).str.zfill(2) + "-" + (hr + 3).astype(str).str.zfill(2)
        hrp2 = crime["hour_range"].astype(str).str.extract(r"(\d{1,2})")
        crime["hr_key"] = pd.to_numeric(hrp2[0], errors="coerce").fillna(0).astype(int)

        # Tarihli birleşim tercih edilir
        if "date" in crime.columns or "datetime" in crime.columns:
            if "date" not in crime.columns:
                crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
            else:
                crime["date"] = pd.to_datetime(crime["date"], errors="coerce").dt.date
            keys = ["GEOID", "date", "hour_range"]
            merged = crime.merge(
                summary[["GEOID","date","hour_range","311_request_count"]],
                on=keys, how="left"
            )
            log(f"🔗 Join: DATE-BASED ({keys})")
        else:
            cal_keys = ["GEOID", "hr_key", "day_of_week", "season"]
            cal_agg = (summary.groupby(cal_keys, as_index=False)["311_request_count"].median())
            if "day_of_week" not in crime.columns:
                crime["day_of_week"] = 0
            if "season" not in crime.columns:
                if "month" in crime.columns:
                    crime["season"] = pd.to_numeric(crime["month"], errors="coerce").map(_smap).fillna("Summer")
                else:
                    crime["season"] = "Summer"
            merged = crime.merge(cal_agg, on=cal_keys, how="left")
            log(f"🔗 Join: CALENDAR-BASED ({cal_keys})")

        merged["311_request_count"] = pd.to_numeric(merged.get("311_request_count"), errors="coerce").fillna(0).astype(int)
        save_atomic(merged, (SAVE_DIR / "sf_crime_02.csv").as_posix())
        log("✅ Suç + 311 birleştirmesi tamamlandı.")
        try:
            log(merged.head(5).to_string(index=False))
        except Exception:
            pass

    except Exception as e:
        log(f"⚠️ 311 merge hatası: {e} → PASSTHROUGH")
        try:
            p = SAVE_DIR / "sf_crime_01.csv"
            if p.exists():
                crime = read_csv_safe(p, dtype={"GEOID": str}, low_memory=False)
                crime["311_request_count"] = 0
                save_atomic(crime, (SAVE_DIR / "sf_crime_02.csv").as_posix())
                log("✅ Passthrough yazıldı (exception fallback).")
        except Exception as ee:
            log(f"❌ Passthrough da başarısız: {ee}")

if __name__ == "__main__":
    main()

from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import numpy as np
from io import BytesIO
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = boto3.client('s3', region_name='us-east-1', config=Config(signature_version=UNSIGNED))
BUCKET = 'unidata-nexrad-level2'


# ===== CHASER-STYLE COLOR TABLES (GR2Analyst inspired) =====
# Reflectivity: full chaser palette with vibrant pinks/purples for hail cores
REFL_COLORS = [
    (-30, '#646464'), (-25, '#9696c8'), (-20, '#785084'), (-15, '#643c8c'),
    (-10, '#3232a0'), (-5, '#0000d2'), (0, '#0064d2'), (5, '#00afff'),
    (10, '#00d2ff'), (15, '#00ffd2'), (20, '#00d200'), (25, '#00a000'),
    (30, '#008c00'), (35, '#ffff00'), (40, '#ffaa00'), (45, '#ff7800'),
    (50, '#ff0000'), (55, '#c80000'), (60, '#a00000'), (65, '#ff00ff'),
    (70, '#9b51c5'), (75, '#ffffff'),
]

# Velocity: classic green/red with high-end purples for tornadic rotation
VEL_COLORS = [
    (-100, '#00ffff'), (-80, '#0080ff'), (-60, '#0040ff'), (-40, '#00d000'),
    (-30, '#00a000'), (-20, '#008000'), (-10, '#005000'), (-5, '#003800'),
    (0, '#000000'), (5, '#380000'), (10, '#500000'), (20, '#800000'),
    (30, '#a00000'), (40, '#d00000'), (60, '#ff0080'), (80, '#ff00ff'),
    (100, '#ffffff'),
]


def make_cmap(color_list):
    """Build a matplotlib colormap from (value, hex) pairs."""
    values = [c[0] for c in color_list]
    colors = [c[1] for c in color_list]
    vmin, vmax = values[0], values[-1]
    # Normalize values to 0-1 for the colormap
    normalized = [(v - vmin) / (vmax - vmin) for v in values]
    return LinearSegmentedColormap.from_list('custom', list(zip(normalized, colors))), vmin, vmax


REFL_CMAP, REFL_MIN, REFL_MAX = make_cmap(REFL_COLORS)
VEL_CMAP, VEL_MIN, VEL_MAX = make_cmap(VEL_COLORS)


def find_storm_center(radar, sweep=0):
    """Find the lat/lon center of the strongest reflectivity in the sweep."""
    try:
        refl = radar.get_field(sweep, 'reflectivity')
        if hasattr(refl, 'filled'):
            refl_arr = refl.filled(-999)
        else:
            refl_arr = np.array(refl)

        # Threshold at 50 dBZ to find storm cores
        strong = refl_arr > 50
        if not np.any(strong):
            strong = refl_arr > 35
        if not np.any(strong):
            return None, None, None

        # Get range and azimuth indices of strong returns
        ranges = radar.range['data']
        azimuths = radar.get_azimuth(sweep)

        # Get indices of strong points
        az_idx, rng_idx = np.where(strong)

        # Average position of strong returns (weighted by intensity)
        weights = refl_arr[az_idx, rng_idx]
        weights = np.maximum(weights, 0)

        avg_az = np.average(azimuths[az_idx], weights=weights)
        avg_rng = np.average(ranges[rng_idx], weights=weights)

        # Convert to lat/lon offset from radar
        radar_lat = radar.latitude['data'][0]
        radar_lon = radar.longitude['data'][0]

        # Approximate offset (good enough for centering)
        az_rad = np.radians(avg_az)
        # 1 degree lat ~ 111km, 1 degree lon varies with latitude
        d_north = avg_rng * np.cos(az_rad) / 1000.0  # km
        d_east = avg_rng * np.sin(az_rad) / 1000.0
        d_lat = d_north / 111.0
        d_lon = d_east / (111.0 * np.cos(np.radians(radar_lat)))

        storm_lat = radar_lat + d_lat
        storm_lon = radar_lon + d_lon

        # Use a 60km zoom box around the storm
        return storm_lat, storm_lon, 0.6  # 0.6 degrees ~ 67km
    except Exception:
        return None, None, None


@app.get("/")
def root():
    return {"status": "online", "message": "Radar backend v3 — chaser colors + auto-zoom"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scans/{site}/{date}")
def list_scans(site: str, date: str):
    try:
        y, m, d = date.split('-')
        prefix = f"{y}/{m}/{d}/{site.upper()}/"
        response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, MaxKeys=300)
        if 'Contents' not in response:
            return {"site": site, "date": date, "scans": []}
        scans = [
            obj['Key'].split('/')[-1]
            for obj in response['Contents']
            if not obj['Key'].endswith('_MDM')
        ]
        return {"site": site.upper(), "date": date, "scans": scans, "count": len(scans)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/render/{site}/{filename}")
def render(site: str, filename: str, product: str = "reflectivity", zoom: str = "auto"):
    """Render a NEXRAD scan.
    product: 'reflectivity' or 'velocity'
    zoom: 'auto' (focus on storm) or 'full' (entire radar range)"""
    try:
        date_part = filename[4:12]
        y, m, d = date_part[0:4], date_part[4:6], date_part[6:8]
        s3_key = f"{y}/{m}/{d}/{site.upper()}/{filename}"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.ar2v') as tmp:
            s3.download_fileobj(BUCKET, s3_key, tmp)
            tmp_path = tmp.name

        radar = pyart.io.read_nexrad_archive(tmp_path)

        # Lowest sweep for both products (where tornadoes live)
        sweep = 0

        # For velocity, sweep 1 is usually the matching velocity scan to sweep 0 reflectivity
        if product == "velocity":
            # Find first sweep that has velocity data
            for s in range(radar.nsweeps):
                try:
                    radar.get_field(s, 'velocity')
                    sweep = s
                    break
                except Exception:
                    continue

        fig = plt.figure(figsize=(10, 10), facecolor='black')
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')

        display = pyart.graph.RadarDisplay(radar)

        if product == "velocity":
            display.plot('velocity', sweep=sweep, ax=ax,
                         vmin=VEL_MIN, vmax=VEL_MAX, cmap=VEL_CMAP,
                         colorbar_flag=False, title_flag=False, axislabels_flag=False)
        else:
            display.plot('reflectivity', sweep=sweep, ax=ax,
                         vmin=REFL_MIN, vmax=REFL_MAX, cmap=REFL_CMAP,
                         colorbar_flag=False, title_flag=False, axislabels_flag=False)

        # Auto-zoom to the storm
        if zoom == "auto":
            storm_lat, storm_lon, box = find_storm_center(radar, sweep=0)
            if storm_lat is not None:
                ax.set_xlim(storm_lon - box, storm_lon + box)
                ax.set_ylim(storm_lat - box, storm_lat + box)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)

        os.unlink(tmp_path)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render error: {str(e)}")


# ===== Geo-referenced Level 2 rendering (Viper HD / AWIPS palettes) =====
# Serves map-ready overlays for RadarReplay. Bounds are deterministic (a
# HALF_BOX_DEG square around the radar), so the frontend places the overlay
# from the radar's own lat/lon — no need to ship bounds with each image.
import time
from datetime import datetime, timedelta
from threading import Lock
from geo_render import render_geo, HALF_BOX_DEG

_radar_cache = {}   # s3_key -> (radar, atime)   — big objects, keep few
_png_cache = {}     # (s3_key, product) -> (png, atime) — small, keep many
_cache_lock = Lock()
RADAR_CACHE_MAX = 3
PNG_CACHE_MAX = 128


def _evict(d, maxn):
    while len(d) > maxn:
        del d[min(d, key=lambda k: d[k][1])]


def _load_radar(s3_key):
    with _cache_lock:
        hit = _radar_cache.get(s3_key)
        if hit:
            _radar_cache[s3_key] = (hit[0], time.time())
            return hit[0]
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ar2v') as tmp:
        s3.download_fileobj(BUCKET, s3_key, tmp)
        path = tmp.name
    radar = pyart.io.read_nexrad_archive(path)
    os.unlink(path)
    with _cache_lock:
        _radar_cache[s3_key] = (radar, time.time())
        _evict(_radar_cache, RADAR_CACHE_MAX)
    return radar


def _parse_ts(key):
    """SITEYYYYMMDD_HHMMSS_V06 -> ISO 'YYYY-MM-DDTHH:MM:SSZ' (or None).
    e.g. KVNX20240507_000608_V06 : date is glued to the site in field 0."""
    name = key.split('/')[-1]
    parts = name.split('_')
    if len(parts) < 2:
        return None
    try:
        dt = datetime.strptime(parts[0][-8:] + parts[1], "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


@app.get("/geo/scans/{site}")
def geo_scans(site: str, start: str, end: str):
    """Available Level 2 scan keys+times for a 4-letter site over [start,end]
    (ISO, e.g. 2024-05-07T00:00Z). Frontend maps frames to the nearest ts."""
    try:
        s = datetime.fromisoformat(start.replace('Z', '+00:00')).replace(tzinfo=None)
        e = datetime.fromisoformat(end.replace('Z', '+00:00')).replace(tzinfo=None)
        site = site.upper()
        scans, day = [], datetime(s.year, s.month, s.day)
        while day <= e:
            prefix = f"{day:%Y/%m/%d}/{site}/"
            token = None
            while True:
                kw = dict(Bucket=BUCKET, Prefix=prefix, MaxKeys=1000)
                if token:
                    kw['ContinuationToken'] = token
                resp = s3.list_objects_v2(**kw)
                for obj in resp.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('_MDM'):
                        continue
                    iso = _parse_ts(key)
                    if iso:
                        t = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")
                        if s <= t <= e:
                            scans.append({"ts": iso, "key": key})
                if resp.get('IsTruncated'):
                    token = resp.get('NextContinuationToken')
                else:
                    break
            day += timedelta(days=1)
        scans.sort(key=lambda x: x['ts'])
        return {"site": site, "count": len(scans), "scans": scans}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/geo/render/{product}")
def geo_render_endpoint(product: str, key: str):
    """Render one product from a Level 2 scan (by its S3 key) as a transparent,
    geo-referenced PNG. Cached so replay/re-view is instant."""
    cache_id = (key, product)
    with _cache_lock:
        hit = _png_cache.get(cache_id)
        if hit:
            _png_cache[cache_id] = (hit[0], time.time())
    if hit:
        png = hit[0]
    else:
        try:
            radar = _load_radar(key)
            png, _bounds = render_geo(radar, product)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as ex:
            raise HTTPException(status_code=500, detail=f"Render error: {ex}")
        with _cache_lock:
            _png_cache[cache_id] = (png, time.time())
            _evict(_png_cache, PNG_CACHE_MAX)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})

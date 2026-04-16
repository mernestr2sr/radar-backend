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

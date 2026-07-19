"""Geo-referenced Level 2 rendering.

Renders a NEXRAD moment (reflectivity / velocity / CC / ZDR) to a transparent
PNG aligned to a lat/lon bounding box, so RadarReplay can drop it straight onto
Leaflet as an imageOverlay. Returns (png_bytes, bounds) where bounds is
[[south, west], [north, east]] — the same shape the frontend already uses for
the IEM velocity overlays.
"""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from palettes import (
    VIPER_HD_REFL_CMAP, VIPER_HD_REFL_MIN, VIPER_HD_REFL_MAX,
    AWIPS_EVANS_VEL_CMAP, AWIPS_EVANS_VEL_MIN, AWIPS_EVANS_VEL_MAX,
    AWIPS_RHO_CC_CMAP, AWIPS_RHO_CC_MIN, AWIPS_RHO_CC_MAX,
    build_cmap, MS_TO_KTS,
)

HALF_BOX_DEG = 2.5  # render a 5-degree square centered on the radar

# Per-product multiplier applied to the raw Py-ART field before rendering.
# Velocity: m/s -> knots (AWIPS Evans is a knots table; forecasters read knots).
DATA_SCALE = {'velocity': MS_TO_KTS}

# product -> (level2 field name, cmap, vmin, vmax)
# reflectivity: Viper HD | velocity: AWIPS Evans | CC: AWIPS Rho (Matt's picks)
PRODUCTS = {
    'reflectivity': ('reflectivity', VIPER_HD_REFL_CMAP, VIPER_HD_REFL_MIN, VIPER_HD_REFL_MAX),
    'velocity':     ('velocity',     AWIPS_EVANS_VEL_CMAP, AWIPS_EVANS_VEL_MIN, AWIPS_EVANS_VEL_MAX),
    'cc':           ('cross_correlation_ratio', AWIPS_RHO_CC_CMAP, AWIPS_RHO_CC_MIN, AWIPS_RHO_CC_MAX),
    'zdr':          ('differential_reflectivity', None, -4, 8),  # cmap filled below
}
# ZDR: simple perceptual ramp
_ZDR = [(-4, 40, 40, 40), (0, 0, 120, 200), (1, 0, 200, 120), (3, 240, 220, 40), (6, 240, 60, 40), (8, 255, 0, 255)]
ZDR_CMAP, _, _ = build_cmap(_ZDR)
PRODUCTS['zdr'] = ('differential_reflectivity', ZDR_CMAP, -4, 8)


def pick_sweep(radar, field):
    """Lowest sweep with REAL data for this field. NEXRAD split cuts carry the
    field on every sweep but leave it fully masked on the wrong one (velocity is
    masked on the surveillance sweep, present on the Doppler sweep), so we must
    check for unmasked values, not just that get_field succeeds."""
    for s in range(radar.nsweeps):
        try:
            d = radar.get_field(s, field)
        except Exception:
            continue
        if np.ma.count(d) > 0:  # has at least some unmasked gates
            return s
    return 0


def render_geo(radar, product, range_km=300.0):
    """Render one product from an open Py-ART radar. Returns (png_bytes, bounds).
    Renders to a FIXED radar-centered box of `range_km` radius so bounds are
    deterministic — the frontend can place the overlay from the radar's lat/lon
    (no need to read them back), and the data circle sits exactly inside."""
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product}")
    field, cmap, vmin, vmax = PRODUCTS[product]
    if field not in radar.fields:
        raise ValueError(f"field {field} not in scan (products: {list(radar.fields)})")

    sweep = pick_sweep(radar, field)
    lat, lon, _ = radar.get_gate_lat_lon_alt(sweep)
    data = radar.get_field(sweep, field)
    scale = DATA_SCALE.get(product)
    if scale:
        data = data * scale  # e.g. velocity m/s -> knots to match the knots palette

    # Fixed radar-centered box (deterministic). range_km -> degrees; lon scaled
    # by latitude. The data circle fits inside; corners are transparent.
    rlat = float(radar.latitude['data'][0])
    rlon = float(radar.longitude['data'][0])
    r_lat = range_km / 111.0
    r_lon = range_km / (111.0 * np.cos(np.radians(rlat)))
    south, north = rlat - r_lat, rlat + r_lat
    west, east = rlon - r_lon, rlon + r_lon

    # 800x800 (8in @ 100dpi). Lower than before to cut render CPU + memory on
    # small instances; plenty for a map overlay that gets scaled by Leaflet anyway.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    buf = io.BytesIO()
    # NO bbox_inches='tight' — the axes fill the whole figure (add_axes([0,0,1,1])),
    # so the saved canvas maps exactly to xlim/ylim. Cropping would misalign bounds.
    fig.savefig(buf, format='png', transparent=True, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), [[south, west], [north, east]]


def _metpy_radial(f):
    """(data, start_az) from a MetPy Level3File radial packet."""
    p = f.sym_block[0][0]
    return np.ma.asarray(f.map_data(np.asarray(p['data']))), np.asarray(p['start_az'])


def _align_to(src, src_az, dst_az, ngate):
    """Resample a product's radials to dst_az (nearest azimuth), slice to ngate.
    Returns a plain array filled with -999 where missing (for thresholding)."""
    idx = np.array([np.argmin(np.abs(((src_az - a + 180) % 360) - 180)) for a in dst_az])
    out = np.ma.filled(src[idx], -999.0)
    if out.shape[1] >= ngate:
        return out[:, :ngate]
    pad = np.full((out.shape[0], ngate - out.shape[1]), -999.0)
    return np.concatenate([out, pad], axis=1)


def render_l3_reflectivity_metpy(f, range_km=300.0, px=1600):
    """Render N0B super-res base reflectivity (decoded by MetPy) with the Viper HD
    palette, to the fixed radar-centered box. Sliced to range_km so it shares the
    same box as velocity/CC. Returns (png, bounds)."""
    data, az = _metpy_radial(f)
    ng = min(int(round(range_km / 0.25)), data.shape[1])  # 250m super-res gates
    data = data[:, :ng]
    rng = (np.arange(ng) + 0.5) * (range_km * 1000.0 / ng)
    rlat, rlon = float(f.lat), float(f.lon)
    x = rng[None, :] * np.sin(np.radians(az)[:, None])
    y = rng[None, :] * np.cos(np.radians(az)[:, None])
    lat = rlat + (y / 1000.0) / 111.0
    lon = rlon + (x / 1000.0) / (111.0 * np.cos(np.radians(rlat)))

    r_lat = range_km / 111.0
    r_lon = range_km / (111.0 * np.cos(np.radians(rlat)))
    south, north = rlat - r_lat, rlat + r_lat
    west, east = rlon - r_lon, rlon + r_lon

    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(lon, lat, data, cmap=VIPER_HD_REFL_CMAP,
                  vmin=VIPER_HD_REFL_MIN, vmax=VIPER_HD_REFL_MAX, shading='auto')
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), [[south, west], [north, east]]


def render_l3_velocity_metpy(f, refl_f=None, cc_f=None, range_km=300.0,
                             refl_min=15.0, cc_min=0.80, px=1600):
    """Render N0G TRUE BASE VELOCITY (super-res, 720 radials) decoded by MetPy.
    If refl_f (N0B) and/or cc_f (N0C) are supplied, mask velocity to only real
    precip (refl >= refl_min AND CC >= cc_min) — removes bug/clutter speckle.
    px controls output resolution (sharpness when zoomed). Returns (png, bounds)."""
    vel, vaz = _metpy_radial(f)
    ngate = vel.shape[1]
    mask = np.ma.getmaskarray(vel).copy()
    if refl_f is not None:
        rd, raz = _metpy_radial(refl_f)
        mask |= (_align_to(rd, raz, vaz, ngate) < refl_min)
    if cc_f is not None:
        cd, caz = _metpy_radial(cc_f)
        mask |= (_align_to(cd, caz, vaz, ngate) < cc_min)
    vel = np.ma.masked_array(np.ma.getdata(vel), mask)

    az = np.radians(vaz)
    rng = (np.arange(ngate) + 0.5) * (range_km * 1000.0 / ngate)
    rlat, rlon = float(f.lat), float(f.lon)
    x = rng[None, :] * np.sin(az[:, None])
    y = rng[None, :] * np.cos(az[:, None])
    lat = rlat + (y / 1000.0) / 111.0
    lon = rlon + (x / 1000.0) / (111.0 * np.cos(np.radians(rlat)))
    kts = vel * MS_TO_KTS

    r_lat = range_km / 111.0
    r_lon = range_km / (111.0 * np.cos(np.radians(rlat)))
    south, north = rlat - r_lat, rlat + r_lat
    west, east = rlon - r_lon, rlon + r_lon

    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(lon, lat, kts, cmap=AWIPS_EVANS_VEL_CMAP,
                  vmin=AWIPS_EVANS_VEL_MIN, vmax=AWIPS_EVANS_VEL_MAX, shading='auto')
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), [[south, west], [north, east]]

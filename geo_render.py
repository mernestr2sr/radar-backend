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

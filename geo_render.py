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
    VIPER_HD_REFL_CMAP, VIPER_HD_REFL_MIN, VIPER_HD_REFL_MAX, build_cmap,
)

HALF_BOX_DEG = 2.5  # render a 5-degree square centered on the radar

# --- default velocity palette (green inbound / red outbound), black at zero.
# Placeholder until the Viper HD velocity table is supplied. m/s.
VEL_ANCHORS = [
    (-40, 0, 255, 255), (-30, 0, 128, 255), (-20, 0, 200, 0), (-10, 0, 110, 0),
    (-1, 0, 40, 0), (0, 0, 0, 0), (1, 40, 0, 0), (10, 130, 0, 0),
    (20, 220, 0, 0), (30, 255, 40, 40), (40, 255, 0, 255),
]
VEL_CMAP, VEL_MIN, VEL_MAX = build_cmap(VEL_ANCHORS)

# --- correlation coefficient: low CC (debris/non-met) stands out dark/purple,
# meteorological high CC ramps blue->green->yellow->red. Unitless (~0.2-1.05).
CC_ANCHORS = [
    (0.20, 20, 20, 20), (0.45, 90, 40, 130), (0.65, 40, 60, 200),
    (0.80, 0, 180, 200), (0.90, 40, 210, 90), (0.95, 240, 240, 40),
    (0.98, 255, 140, 0), (1.05, 220, 0, 0),
]
CC_CMAP, CC_MIN, CC_MAX = build_cmap(CC_ANCHORS)

# product -> (level2 field name, cmap, vmin, vmax)
PRODUCTS = {
    'reflectivity': ('reflectivity', VIPER_HD_REFL_CMAP, VIPER_HD_REFL_MIN, VIPER_HD_REFL_MAX),
    'velocity':     ('velocity',     VEL_CMAP, VEL_MIN, VEL_MAX),
    'cc':           ('cross_correlation_ratio', CC_CMAP, CC_MIN, CC_MAX),
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


def render_geo(radar, product, half_box=HALF_BOX_DEG):
    """Render one product from an open Py-ART radar. Returns (png_bytes, bounds)."""
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product}")
    field, cmap, vmin, vmax = PRODUCTS[product]
    if field not in radar.fields:
        raise ValueError(f"field {field} not in scan (products: {list(radar.fields)})")

    sweep = pick_sweep(radar, field)
    lat, lon, _ = radar.get_gate_lat_lon_alt(sweep)
    data = radar.get_field(sweep, field)

    rlat = float(radar.latitude['data'][0])
    rlon = float(radar.longitude['data'][0])
    west, east = rlon - half_box, rlon + half_box
    south, north = rlat - half_box, rlat + half_box

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    buf = io.BytesIO()
    # NO bbox_inches='tight' — the axes fill the whole figure (add_axes([0,0,1,1])),
    # so the saved canvas maps exactly to xlim/ylim. Cropping would misalign bounds.
    fig.savefig(buf, format='png', transparent=True, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), [[south, west], [north, east]]

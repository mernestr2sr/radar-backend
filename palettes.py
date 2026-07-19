"""Radar color palettes for Level 2 rendering.

Viper HD is a popular community reflectivity palette (Wx Tools, compatible with
RadarScope / GRLevelX). Source: https://www.wxtools.org/reflectivity/viper-hd
Defined here as (dBZ, #hex) anchor points at the palette's color transitions —
matplotlib interpolates the smooth ramp between them, reproducing the look
without hand-listing every 0.5 dBZ step.
"""
from matplotlib.colors import LinearSegmentedColormap

# Viper HD reflectivity — anchor colors at each transition (dBZ, R, G, B)
VIPER_HD_REFL = [
    (0,   1, 243, 247),   # near-white cyan (lightest returns)
    (5,   21, 123, 167),  # blue
    (10,  12, 137, 201),
    (13.5, 5, 158, 235),  # bright blue (top of the blue band)
    (14,  21, 191, 180),  # blue->green transition
    (14.5, 37, 225, 125), # green
    (20,  26, 187, 90),
    (25,  17, 154, 59),
    (30,  8, 120, 27),    # dark green (top of green band)
    (34,  128, 175, 19),  # green->yellow transition
    (34.5, 255, 255, 33), # yellow
    (38,  255, 199, 0),
    (40,  255, 157, 0),   # orange
    (43,  255, 52, 0),
    (44,  255, 17, 0),    # into red
    (44.5, 255, 0, 0),    # red
    (50,  195, 0, 0),
    (54,  154, 0, 0),     # dark red (top of red band)
    (54.5, 180, 0, 180),  # red->magenta transition
    (57,  210, 49, 206),
    (60,  247, 108, 237), # pink/magenta
    (60.5, 253, 117, 243),
    (61,  232, 109, 232),
    (63,  151, 77, 146),
    (65,  70, 45, 68),    # dark purple (extreme cores)
    (66,  29, 30, 29),    # near-black
    (70,  61, 62, 61),    # gray ramp for the very highest values
    (75,  101, 102, 101),
    (80,  142, 142, 142),
    (85,  182, 182, 182),
    (94,  254, 254, 254), # white cap
]


def build_cmap(anchors):
    """Anchor list [(dBZ, r, g, b), ...] -> (LinearSegmentedColormap, vmin, vmax)."""
    vals = [a[0] for a in anchors]
    vmin, vmax = vals[0], vals[-1]
    stops = []
    for dbz, r, g, b in anchors:
        pos = (dbz - vmin) / (vmax - vmin)
        stops.append((pos, (r / 255.0, g / 255.0, b / 255.0)))
    return LinearSegmentedColormap.from_list("viper_hd", stops), vmin, vmax


VIPER_HD_REFL_CMAP, VIPER_HD_REFL_MIN, VIPER_HD_REFL_MAX = build_cmap(VIPER_HD_REFL)

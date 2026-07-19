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


# ---------------------------------------------------------------------------
# AWIPS Evans base velocity (operational NWS table). Source: Wx Tools.
# Native units are KNOTS — velocity is rendered in knots (Py-ART data is m/s and
# gets converted with MS_TO_KTS in geo_render), so this table stays in knots.
MS_TO_KTS = 1.943844
AWIPS_EVANS_VEL_KT = [
    (-120, 255, 0, 128), (-90.5, 0, 0, 160), (-70, 0, 224, 255),
    (-69.99, 0, 255, 224), (-60, 0, 255, 225), (-59.99, 160, 255, 208),
    (-50, 160, 255, 208), (-49.99, 160, 255, 208), (-40, 0, 255, 0),
    (-10, 16, 96, 16), (-9.99, 16, 96, 16), (-0.01, 112, 128, 112),
    (0, 144, 128, 144), (10, 112, 0, 0), (40, 255, 0, 0),
    (48.6, 255, 0, 128), (49.5, 255, 0, 144), (69.99, 255, 196, 255),
    (70, 255, 96, 0), (120, 255, 255, 0),
]
AWIPS_EVANS_VEL_CMAP, AWIPS_EVANS_VEL_MIN, AWIPS_EVANS_VEL_MAX = build_cmap(AWIPS_EVANS_VEL_KT)

# ---------------------------------------------------------------------------
# AWIPS Rho correlation coefficient (operational NWS table). Source: Wx Tools.
# Ascending order; low CC (debris / non-met) reads blue, met rain red/pink.
AWIPS_RHO_CC = [
    (0.00, 15, 15, 140), (0.45, 15, 15, 140), (0.60, 10, 10, 190),
    (0.75, 120, 120, 255), (0.80, 95, 245, 100), (0.85, 135, 215, 10),
    (0.90, 255, 255, 0), (0.95, 255, 140, 0), (0.97, 225, 3, 0),
    (0.99, 139, 30, 77), (1.00, 255, 180, 215), (1.05, 164, 54, 150),
]
AWIPS_RHO_CC_CMAP, AWIPS_RHO_CC_MIN, AWIPS_RHO_CC_MAX = build_cmap(AWIPS_RHO_CC)

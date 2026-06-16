import re
from typing import Optional
from fastapi import FastAPI, Body

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION MATRICES & VALUES (Ported from VBA)
# ─────────────────────────────────────────────────────────────────────────────

# Hardcoded database table (Sequential ID -> CAD Block, Public Zone Values, Occupational Zone Values)
_RAW = [
    (1,  "C_001", 25.7,2.2,3.6,0.7,0.1,3.6,5.3,4,0.1,0.4,4.4,6.4,24.2,   11,1.3,1.5,0.2,0.1,0.9,2,1.3,0.1,0.1,1.6,3.6,10.6),
    (2,  "C_002", 22.2,1.6,3.4,0.4,0.1,2.8,4.8,4,0.1,0.3,3.5,5.5,17.7,   9.5,0.8,1.3,0.2,0.1,0.6,1.9,1.2,0.1,0.1,1.2,3.2,7.9),
    (3,  "C_003", 28.6,2.4,3.9,0.8,0.1,3.8,5.3,4,0.1,0.4,4.7,6.7,26.2,   12.2,1.3,1.1,0.2,0.1,1.1,1.9,1.3,0.1,0.1,1.7,3.7,11.4),
    (4,  "C_004", 47.5,4.5,5,2.1,0.5,5.7,8.2,7.2,0.3,1.4,9.4,11.4,50.4,  20.3,2,1.9,0.2,0.1,3.2,3.1,2.2,0.1,0.1,3.5,5.5,21.6),
    (5,  "C_005", 40.8,2.8,6,1.3,0.1,5,7.5,6.5,0.1,0.9,7.3,9.3,36,        17.4,1.5,2.4,0.2,0.1,1.9,3,2.1,0.1,0.1,2.7,4.7,15.6),
    (6,  "C_006", 25.7,1.9,3.6,0.5,0.1,3.3,4.9,4,0.1,0.3,4,6,20.3,        11,0.9,1.5,0.2,0.1,0.9,1.9,1.4,0.1,0.1,1.3,3.3,8.9),
    (7,  "C_007", 45.7,3.8,4.6,1.8,0.5,5.5,8.1,7,0.2,1.3,8.9,10.9,47.7,  19.5,1.8,2.7,0.2,0.1,2.8,3.2,2.5,0.1,0.1,3.4,5.4,20.6),
    (8,  "C_008", 38.7,2.3,5.3,1,0.1,4.8,7.4,6,0.1,0.7,6.7,8.7,32.2,     16.5,1.4,2.4,0.2,0.1,1.7,2.9,2,0.1,0.1,2.5,4.5,13.9),
    (9,  "C_009", 43.8,4.2,4.1,1.7,0.5,5.5,8.2,6.7,0.2,1.4,8.8,10.8,46.7, 18.7,1.9,2.7,0.2,0.1,2.7,3.2,2.5,0.1,0.1,3.3,5.3,20.1),
    (10, "C_010", 36.4,2.4,5.1,0.8,0.2,4.8,7.4,6,0.1,0.7,6.4,8.4,30.5,   15.5,1.4,2.1,0.2,0.1,1.5,2.8,1.9,0.1,0.1,2.4,4.4,13.2),
    (11, "C_011", 45.6,3.7,4.5,2,0.4,5.5,8.4,6.8,0.3,1.3,9.3,11.3,49.6,  19.5,2,2.7,0.2,0.1,2.9,3.3,2.6,0.1,0.1,3.5,5.5,21.3),
    (12, "C_012", 38.5,2.7,5.1,1.1,0.1,5,7.7,6,0.1,0.9,6.9,8.9,34.6,     16.5,1.5,2.4,0.2,0.1,1.8,3,2.4,0.1,0.1,2.7,4.7,15),
    (13, "C_013", 14.5,6.8,0.1,0.2,0.1,1.2,0.7,0.2,0.1,0.1,1.8,3.8,11.9, 6.2,3.9,0.1,0.2,0.1,0.1,0.3,0.2,0.1,0.1,0.7,2.7,5.1),
    (14, "C_014", 29.4,2.5,3.9,0.8,0.1,3.9,5.3,4,0.1,0.5,4.8,6.8,26.6,   12.5,1.3,1.1,0.2,0.1,1.1,1.9,1.3,0.1,0.1,1.7,3.7,11.6),
    (15, "C_015", 26.4,2,3.6,0.6,0.1,3.2,4.9,3.9,0.1,0.3,4.1,6.1,20.9,   11.3,1,1.5,0.2,0.1,0.8,1.9,1.4,0.1,0.1,1.3,3.3,9.2),
    (16, "C_016", 46.2,4.4,4.6,1.8,0.6,5.6,8.2,7,0.2,1.4,9.1,11.1,48.1,  19.7,1.8,2.7,0.2,0.1,2.9,3.2,2.5,0.1,0.1,3.4,5.4,20.7),
    (17, "C_017", 39.2,2.8,5.3,1.1,0.3,4.9,7.4,6,0.1,0.8,6.8,8.8,32.6,   16.7,1.4,2.4,0.2,0.1,1.8,2.9,2,0.1,0.1,2.5,4.5,14.1),
    (18, "C_018", 47.8,4,4.9,2.3,0.4,5.3,8.2,7.2,0.4,1.3,9.5,11.5,50.6,  20.4,2,1.9,0.2,0.1,3.1,3.2,2.2,0.1,0.1,3.5,5.5,21.7),
    (19, "C_019", 41.2,3,6,1.3,0.2,5,7.5,6.7,0.1,0.9,7.3,9.3,36.4,        17.5,1.6,2.4,0.2,0.1,2,3,2.1,0.1,0.1,2.7,4.7,15.7),
    (20, "C_020", 40.7,3,2.6,1.5,0.4,5.3,6.7,5.1,0.2,1.1,8.2,10.2,44.8,  17.3,2.1,1.2,0.2,0.1,2.3,2.4,0.7,0.1,0.1,3.1,5.1,19.3),
    (21, "C_021", 32.6,1.9,3.1,0.6,0.1,4.4,5.5,4.5,0.1,0.5,5.4,7.4,28.3, 13.8,1.8,1.1,0.2,0.1,1,1.5,0.9,0.1,0.1,2.1,4.1,12.3),
    (22, "C_022", 41.3,4.5,3.9,2.9,0.1,3.5,6.6,4.3,0.3,1.2,7.4,9.4,42.6, 17.6,2.3,1.5,0.4,0.1,2.3,2.3,1.4,0.1,0.2,3,5,18.4),
    (23, "C_023.1",39.5,4,4,2.5,0.1,3.4,6.2,4.4,0.2,1,6.9,8.9,40,         16.9,2.1,1.6,0.2,0.1,2.4,2.3,1.4,0.1,0.1,2.8,4.8,17),
    (24, "C_024", 54.6,5.9,5.3,3.4,0.7,4.9,8.8,7.6,0.6,1.9,10.6,12.6,58.1, 23.4,2.5,2.1,0.3,0.1,3.9,3.5,2.3,0.1,0.3,4.1,6.1,25),
    (25, "C_025", 49,3.6,5.5,2.5,0.1,4.9,8.2,6.7,0.3,1.3,8.8,10.8,46.4,  20.8,2.1,1.9,0.4,0.1,2.7,3.1,1.9,0.1,0.2,3.4,5.4,20),
    (26, "C_026", 55.9,3.2,6.4,3.7,0.2,4.8,9.1,7.9,0.7,1.9,11.1,13.1,60.2, 23.8,2.6,2.1,0.5,0.1,3.7,3.6,2.3,0.1,0.3,4.3,6.3,25.9),
    (27, "C_027", 50.1,4,5.9,3.2,0.2,4.2,8.5,6.7,0.4,1.5,9.1,11.1,49.3,  21.5,2.1,1.9,0.5,0.1,2.9,3.2,1.8,0.1,0.2,3.5,5.5,21.2),
    (28, "C_028", 30.8,2.5,3.4,0.8,0.1,3.7,4.9,4.1,0.1,0.4,4.9,6.9,24.4, 13.2,1.2,1.1,0.2,0.1,1.2,1.9,1.3,0.1,0.1,1.6,3.6,10.7),
    (29, "C_029", 48.9,1.9,5.6,2,0.2,5.7,8.1,7.3,0.2,1.3,9.1,11.1,49.3,  20.8,2.4,1.9,0.2,0.1,3.1,3.1,2.2,0.1,0.1,3.4,5.4,21.3),
    (30, "C_030", 42.4,2.1,6,1.3,0.1,5.1,7.5,6.1,0.1,0.8,7.3,9.3,35,     18,1.4,2.4,0.2,0.1,2,2.9,2,0.1,0.1,2.6,4.6,15.1),
    (31, "C_031", 49.6,4.6,3.7,3.3,0.7,4.5,7.7,5.6,0.6,1.5,10,12,55.4,   21.3,2.7,1.6,0.3,0.1,3.4,2.7,1.4,0.1,0.2,3.9,5.9,23.8),
    (32, "C_032", 43.4,3.4,4.1,2.5,0.1,4,6.6,5.8,0.3,1.3,7.8,9.8,43.4,   18.5,2.1,1.7,0.2,0.1,2.5,2.2,1.5,0.1,0.1,3.1,5.1,18.7),
    (33, "C_033", 25.5,2.5,2.6,0.6,0.1,1.3,5.5,2.3,0.1,0.1,4,6,22,        10.9,0.9,1.5,0.2,0.1,0.4,2.2,1.7,0.1,0.1,1.2,3.2,9.4),
    (34, "C_034", 34.8,2.4,3.9,0.9,0.1,1.6,7.4,4,0.1,0.2,5.8,7.8,28.8,   14.8,1.7,1.3,0.2,0.1,0.7,2.9,1.4,0.1,0.1,2,4,12.5),
    (35, "C_035", 32.6,2.3,3.6,0.2,0.1,2.5,7,3.8,0.1,0.1,5.4,7.4,27,      13.8,1.6,1.3,0.2,0.1,0.7,2.8,1.3,0.1,0.1,1.9,3.9,11.8),
    (36, "C_036", 49.8,2.5,4.9,1.3,0.4,5.7,9.7,7,0.2,1.2,9.4,11.4,49.2,  21.2,2,1.9,0.2,0.1,2.5,4,1.9,0.1,0.1,3.4,5.4,21.3),
    (37, "C_037", 43.2,2.3,5.2,1.1,0.1,3.6,9,6.4,0.1,0.6,7.7,9.7,36.4,   18.4,1.7,2,0.2,0.1,1.5,3.7,2.3,0.1,0.1,2.7,4.7,15.7),
    (38, "C_038", 51.1,2.5,4.9,1.3,0.4,5.8,10.2,6,0.2,1.2,9.7,11.7,50.3, 21.7,2,2.3,0.2,0.1,2.5,4.1,2.8,0.1,0.1,3.5,5.5,21.6),
    (39, "C_039", 45,3.8,5.5,1.2,0.5,3.6,9.4,6.9,0.1,0.6,8,10,37.8,       19.2,1.7,1.9,0.2,0.1,1.6,3.9,1.7,0.1,0.1,2.8,4.8,16.3),
    (40, "C_040", 45.4,3.2,4.3,1.2,0.4,5.8,8.6,6,0.1,1.2,8.8,10.8,47.1,  19.4,2,1.7,0.2,0.1,2.4,3.5,1.6,0.1,0.1,3.2,5.2,20.3),
    (41, "C_041", 38.4,2.6,5.7,1,0.3,3.7,7.7,7,0.1,0.5,6.7,8.7,32.8,     16.3,1.5,2.4,0.2,0.1,1.2,3,2.7,0.1,0.1,2.3,4.3,14.1),
    (42, "C_042", 38.1,3.5,3.7,1.3,0.3,5.3,6.5,5.8,0.1,1.3,8.1,10.1,43.7, 16.2,2.3,1.3,0.2,0.1,2.1,1.9,1.3,0.1,0.1,3,5,18.9),
    (43, "C_043", 29.1,1.9,3.9,0.5,0.1,4.1,5.5,4.9,0.1,0.5,5.2,7.2,25.9, 12.4,1.9,0.7,0.2,0.1,0.7,1.3,0.6,0.1,0.1,2,4,11.2),
    (44, "C_044", 58,3.9,6.6,3.9,0.5,5.4,9.6,8.4,0.9,1.9,12.2,14.2,63.5, 24.7,3,2.6,0.4,0.1,4.1,3.8,3.3,0.1,0.5,4.8,6.8,27.3),
    (45, "C_045", 51.3,3.8,5.7,3.3,0.1,4.1,8.7,7.5,0.4,1.6,9.8,11.8,49.9, 21.8,2.1,2.9,0.5,0.1,2.9,3.5,2.8,0.1,0.3,3.9,5.9,21.5),
]

_ZONE_DB = {}
for row in _RAW:
    idx, cad, p, o = row[0], row[1], row[2:15], row[15:28]
    _ZONE_DB[idx] = {
        "cad_block": cad,
        "public": {
            "Front": p[0], "Front2": p[1], "Front3": p[2], "Back": p[3], "Back2": p[4], "Side": p[5],
            "Top": p[6], "Top1": p[7], "Top2": p[8], "Bottom1": p[9], "Bottom2": p[10], "BuildHeight": p[11], "Width": p[12]
        },
        "occupational": {
            "Front": o[0], "Front2": o[1], "Front3": o[2], "Back": o[3], "Back2": o[4], "Side": o[5],
            "Top": o[6], "Top1": o[7], "Top2": o[8], "Bottom1": o[9], "Bottom2": o[10], "BuildHeight": o[11], "Width": o[12]
        }
    }

# Binary bit weights
CTIL726uni, CTIL726MoranW, CTIL726MoranE, CTILVF34, CTILTEF34 = 1, 2, 4, 8, 16
MBNL726, MBNLBrunel, EE726, EE34, H3G81821 = 32, 64, 128, 256, 512
IncEE726, IncEE34, H3GStep1, H3GStep4, H3Gconfig2 = 1024, 2048, 4096, 8192, 16384

_INDEX = {}
def _i(val: int, idx: int): _INDEX[val] = idx

_i(EE726 + EE34, 1); _i(EE726, 2); _i(EE726 + EE34 + H3G81821, 3)
_i(EE726 + EE34 + H3G81821 + CTIL726MoranW + CTILVF34 + CTILTEF34, 4); _i(EE726 + EE34 + H3G81821 + CTIL726MoranW, 5)
_i(EE726 + H3G81821, 6); _i(EE726 + H3G81821 + CTIL726MoranW + CTILVF34 + CTILTEF34, 7); _i(EE726 + H3G81821 + CTIL726MoranW, 8)
_i(EE726 + CTIL726MoranW + CTILVF34 + CTILTEF34, 9); _i(EE726 + CTIL726MoranW, 10)
_i(EE726 + EE34 + CTIL726MoranW + CTILVF34 + CTILTEF34, 11); _i(EE726 + EE34 + CTIL726MoranW, 12); _i(H3Gconfig2, 13)
_i(H3Gconfig2 + IncEE726 + IncEE34, 14); _i(H3Gconfig2 + IncEE726, 15)
_i(H3Gconfig2 + IncEE726 + CTIL726MoranW + CTILVF34 + CTILTEF34, 16); _i(H3Gconfig2 + IncEE726 + CTIL726MoranW, 17)
_i(H3Gconfig2 + IncEE726 + IncEE34 + CTIL726MoranW + CTILVF34 + CTILTEF34, 18); _i(H3Gconfig2 + IncEE726 + IncEE34 + CTIL726MoranW, 19)
_i(H3Gconfig2 + CTIL726MoranW + CTILVF34 + CTILTEF34, 20); _i(H3Gconfig2 + CTIL726MoranW, 21); _i(H3GStep1 + IncEE726 + IncEE34, 22)
_i(H3GStep1 + IncEE726, 23); _i(H3GStep1 + IncEE726 + CTIL726MoranW + CTILVF34 + CTILTEF34, 24); _i(H3GStep1 + IncEE726 + CTIL726MoranW, 25)
_i(H3GStep1 + IncEE726 + IncEE34 + CTIL726MoranW + CTILVF34 + CTILTEF34, 26); _i(H3GStep1 + IncEE726 + IncEE34 + CTIL726MoranW, 27)
_i(MBNL726, 28); _i(MBNL726 + CTIL726MoranW + CTILVF34 + CTILTEF34, 29); _i(MBNL726 + CTIL726MoranW, 30)
_i(H3GStep1 + CTIL726MoranW + CTILVF34 + CTILTEF34, 31); _i(H3GStep1 + CTIL726MoranW, 32); _i(H3GStep4, 33)
_i(H3GStep4 + IncEE726 + IncEE34, 34); _i(H3GStep4 + IncEE726, 35)
_i(H3GStep4 + IncEE726 + CTIL726MoranW + CTILVF34 + CTILTEF34, 36); _i(H3GStep4 + IncEE726 + CTIL726MoranW, 37)
_i(H3GStep4 + IncEE726 + IncEE34 + CTIL726MoranW + CTILVF34 + CTILTEF34, 38); _i(H3GStep4 + IncEE726 + IncEE34 + CTIL726MoranW, 39)
_i(H3GStep4 + CTIL726MoranW + CTILVF34 + CTILTEF34, 40); _i(H3GStep4 + CTIL726MoranW, 41); _i(CTIL726MoranW + CTILVF34 + CTILTEF34, 42)
_i(CTIL726MoranW, 43); _i(H3GStep1 + IncEE726 + IncEE34 + CTIL726uni + CTILVF34 + CTILTEF34, 44); _i(H3GStep1 + IncEE726 + IncEE34 + CTIL726uni, 45)

# ─────────────────────────────────────────────────────────────────────────────
# PARSING & COMPUTATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

_OPERATOR_ALIASES = {
    "bt": "EE", "ee": "EE", "bt/ee": "EE", "ee/bt": "EE", "btee": "EE", "eebr": "EE", "bt ee": "EE", "ee bt": "EE",
    "h3g": "H3G", "three": "H3G", "3": "H3G", "h3": "H3G", "three uk": "H3G", "3uk": "H3G",
    "vmo2": "VMO2", "vmo": "VMO2", "virgin": "VMO2", "o2": "VMO2", "virgin media o2": "VMO2", "virginmediao2": "VMO2",
    "vf": "VF", "vodafone": "VF", "tef": "TEF", "telefonica": "TEF", "ctil": "CTIL"
}

def _parse_bands(freq_str: str) -> set[int]:
    cleaned = re.sub(r'@[\dx]+', '', freq_str, flags=re.IGNORECASE)
    return {int(num) for num in re.findall(r'\d{3,4}', cleaned)}

def icnirp_cal(input_text: str):
    cfg = {
        "ctil": False, "ctil_site_type": "uni", "ctil_moran_operator": "VF", "ctil_vf34": False, "ctil_tef34": False,
        "mbnl": False, "mbnl_mode": "", "ee_5g": False, "inc_h3g_1800_2100": False, "inc_ee_700_2600": False, "inc_ee_3400": False
    }
    warnings = []
    operator_bands = {}

    for line in input_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        op_raw, freq_raw = line.split(":", 1)
        op = _OPERATOR_ALIASES.get(op_raw.strip().lower(), op_raw.strip().upper())
        bands = _parse_bands(freq_raw)
        if bands:
            operator_bands[op] = operator_bands.get(op, set()) | bands

    # 2. Determine configuration pathways
    if "EE" in operator_bands:
        ee_bands = operator_bands["EE"]
        cfg["mbnl"] = True
        if (bool(ee_bands & {700, 800}) and bool(ee_bands & {1800, 2100, 2600})) or ee_bands & {700, 800, 1800, 2100, 2600}:
            cfg["mbnl_mode"] = "EE_700_2600"
            cfg["ee_5g"] = bool(ee_bands & {3400, 3500, 3600})
        elif ee_bands & {3400, 3500, 3600}:
            cfg["mbnl_mode"] = "EE_700_2600"
            cfg["ee_5g"] = True

    if "H3G" in operator_bands:
        h3g_bands = operator_bands["H3G"]
        cfg["mbnl"] = True
        if cfg["mbnl_mode"] == "EE_700_2600":
            if h3g_bands & {1800, 2100} or h3g_bands & {700, 800}:
                cfg["inc_h3g_1800_2100"] = True
        else:
            cfg["mbnl_mode"] = "H3G_Step1"

    ctil_ops = {"VMO2", "VF", "TEF", "CTIL"} & set(operator_bands.keys())
    if ctil_ops:
        cfg["ctil"] = True
        has_vf = "VF" in ctil_ops or "VMO2" in ctil_ops
        has_tef = "TEF" in ctil_ops
        if "CTIL" in ctil_ops or (has_vf and has_tef):
            cfg["ctil_site_type"] = "moran"
            cfg["ctil_moran_operator"] = "VF"
        else:
            cfg["ctil_site_type"] = "uni"

        all_ctil_bands = set().union(*(operator_bands.get(op, set()) for op in ctil_ops))
        if all_ctil_bands & {3400, 3500, 3600}:
            if has_vf: cfg["ctil_vf34"] = True
            if has_tef: cfg["ctil_tef34"] = True

    # 3. Compute distinct binary bit-sum integer
    ctil, mbnl = int(cfg["ctil"]), int(cfg["mbnl"])
    use_ctil_726uni = ctil * int(cfg["ctil_site_type"] == "uni") * CTIL726uni
    use_ctil_726moranW = ctil * int(cfg["ctil_site_type"] == "moran") * int(cfg["ctil_moran_operator"] == "VF") * CTIL726MoranW
    use_ctil_726moranE = ctil * int(cfg["ctil_site_type"] == "moran") * int(cfg["ctil_moran_operator"] == "TEF") * CTIL726MoranE
    use_ctil_vf34 = ctil * int(cfg["ctil_vf34"]) * CTILVF34
    use_ctil_tef34 = ctil * int(cfg["ctil_tef34"]) * CTILTEF34
    use_mbnl_726 = mbnl * int(cfg["mbnl_mode"] == "EE_H3G_800_2600") * MBNL726
    use_mbnl_brunel = mbnl * int(cfg["mbnl_mode"] == "Brunel") * MBNLBrunel
    use_ee726 = mbnl * int(cfg["mbnl_mode"] == "EE_700_2600") * EE726
    use_ee34 = mbnl * int(cfg["mbnl_mode"] == "EE_700_2600") * int(cfg["ee_5g"]) * EE34
    use_h3g81821 = mbnl * int(cfg["mbnl_mode"] == "EE_700_2600") * int(cfg["inc_h3g_1800_2100"]) * H3G81821
    is_h3g_path = int(cfg["mbnl_mode"] in ("H3G_Step1", "H3G_Step4", "H3G_Config2"))
    use_inc_ee726 = mbnl * is_h3g_path * int(cfg["inc_ee_700_2600"]) * IncEE726
    use_inc_ee34 = mbnl * is_h3g_path * int(cfg["inc_ee_700_2600"]) * int(cfg["inc_ee_3400"]) * IncEE34
    use_h3g_step1 = mbnl * int(cfg["mbnl_mode"] == "H3G_Step1") * H3GStep1
    use_h3g_step4 = mbnl * int(cfg["mbnl_mode"] == "H3G_Step4") * H3GStep4
    use_h3g_config2 = mbnl * int(cfg["mbnl_mode"] == "H3G_Config2") * H3Gconfig2

    result_sum = (use_ctil_726uni + use_ctil_726moranW + use_ctil_726moranE + use_ctil_vf34 + use_ctil_tef34 +
                  use_mbnl_726 + use_mbnl_brunel + use_ee726 + use_ee34 + use_h3g81821 + use_inc_ee726 + use_inc_ee34 +
                  use_h3g_step1 + use_h3g_step4 + use_h3g_config2)

    # 4. Resolve Database Match
    seq_idx = _INDEX.get(result_sum, 100)
    zone = _ZONE_DB.get(seq_idx)

    if not zone or seq_idx == 100:
        return {
            "cad_block": "XXX", "public_zones": {}, "occupational_zones": {},
            "binary_sum": result_sum, "sequential_index": seq_idx, "warnings": warnings,
            "error": f"Option not available for configuration combination (sum={result_sum})."
        }

    # 5. Return basic nested Python dictionaries directly
    return {
        "cad_block": zone["cad_block"],
        "public_zones": zone["public"],
        "occupational_zones": zone["occupational"],
        "binary_sum": result_sum,
        "sequential_index": seq_idx,
        "warnings": warnings,
        "error": None
    }
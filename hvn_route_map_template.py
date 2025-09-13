#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVN Route Volume Map — Avelo (XP) & Breeze (MX)

--------------
1) Load route-level data for Tweed–New Haven (HVN) from one of:
   A. BTS T-100 Domestic Segment CSV(s): columns may be either
      - Standard: YEAR, MONTH, ORIGIN, DEST, CARRIER, PASSENGERS
      - BTS variant: MONTH, ORIGIN, DEST, UNIQUE_CARRIER, PASSENGERS  (no YEAR)
        -> If YEAR is missing, the script stamps YEAR from the file name (e.g., ...2024...) or,
           if exactly one --year is provided, from that value.
   B. A simple "routes.csv": columns = [carrier, origin, dest, passengers, year, month]

2) Aggregate to annual one-way passenger volumes per HVN route by carrier.

3) Plot a great-circle map from HVN to each destination. Line color & width track volume.
   - Uses matplotlib; if cartopy is installed you’ll get coastlines automatically.
   - Uses airportsdata for lat/lon if available; otherwise falls back to a built-in dict.

CLI example
-----------
python hvn_route_map_template.py ^
  --input-type t100 ^
  --csv-path T_T100D_SEGMENT_2024_ALL_CARRIER.csv T_T100D_SEGMENT_2025_ALL_CARRIER.csv ^
  --year 2024 2025 ^
  --only-carriers XP MX ^
  --output hvn_routes_2024_2025.png ^
  --kml-output hvn_routes_2024_2025.kml ^
  --summary --summary-csv hvn_routes_summary_2024_2025.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# -------------------------
# Built-in airport fallback
# -------------------------
FALLBACK_AIRPORTS: Dict[str, Tuple[float, float]] = {
    "HVN": (41.265278, -72.888333),
    "ATL": (33.640411, -84.419853),
    "CHS": (32.898647, -80.040528),
    "USA": (35.387798, -80.709099),
    "ORD": (41.974163, -87.907321),
    "DFW": (32.899809, -97.040335),
    "DAB": (29.182045, -81.048751),
    "VPS": (30.48325, -86.5254),
    "DTW": (42.216172, -83.355384),
    "FLL": (26.072583, -80.15275),
    "RSW": (26.536199, -81.755203),
    "GSP": (34.895699, -82.218857),
    "HOU": (29.645419, -95.278889),
    "JAX": (30.494055, -81.687861),
    "EYW": (24.556111, -81.759556),
    "TYS": (35.811, -83.994),
    "LAL": (27.9889, -82.0186),
    "MYR": (33.682676, -78.9275),
    "BNA": (36.126, -86.678),
    "MSY": (29.9934, -90.258),
    "MCO": (28.428889, -81.316028),
    "RDU": (35.8801, -78.788),
    "SJU": (18.4394, -66.0018),
    "SRQ": (27.3954, -82.5544),
    "SAV": (32.1276, -81.2021),
    "TPA": (27.975472, -82.533249),
    "TVC": (44.741444, -85.582235),
    "BWI": (39.1754, -76.6683),
    "IAD": (38.953116, -77.456539),
    "PBI": (26.683161, -80.095589),
    "ILM": (34.2706, -77.9026),
    "VRB": (27.6556, -80.4179),
    "RIC": (37.5052, -77.3197),
    "ORF": (36.8946, -76.2012),
}

def load_airports_latlon() -> Dict[str, Tuple[float, float]]:
    mapping = FALLBACK_AIRPORTS.copy()
    try:
        import airportsdata  # type: ignore
        db = airportsdata.load("IATA")
        for iata, row in db.items():
            try:
                lat = float(row.get("lat"))
                lon = float(row.get("lon"))
                mapping[iata.upper()] = (lat, lon)
            except Exception:
                continue
    except Exception:
        pass
    return mapping

# -------------------------
# Helpers
# -------------------------
_YEAR_REGEX = re.compile(r"(20\d{2})")

def infer_year_from_filename(p: Path) -> Optional[int]:
    """Try to find a 4-digit year like 2024 in the filename."""
    m = _YEAR_REGEX.findall(p.name)
    if not m:
        return None
    # Pick the first occurrence; filenames usually include only one year
    try:
        return int(m[0])
    except Exception:
        return None

# -------------------------
# Data ingestion
# -------------------------
def _normalize_t100_columns(df: pd.DataFrame, assume_year: Optional[int], path: Path) -> pd.DataFrame:
    """
    Accept either:
      - YEAR, MONTH, ORIGIN, DEST, CARRIER, PASSENGERS
      - MONTH, ORIGIN, DEST, UNIQUE_CARRIER, PASSENGERS (+ inject YEAR)
    Normalize to: YEAR, MONTH, ORIGIN, DEST, CARRIER, PASSENGERS
    """
    cols = set(df.columns)
    std_required = {"YEAR", "MONTH", "ORIGIN", "DEST", "CARRIER", "PASSENGERS"}
    alt_required = {"MONTH", "ORIGIN", "DEST", "UNIQUE_CARRIER", "PASSENGERS"}

    if std_required.issubset(cols):
        pass  # already good
    elif alt_required.issubset(cols):
        df = df.rename(columns={"UNIQUE_CARRIER": "CARRIER"})
        if "YEAR" not in df.columns:
            if assume_year is None:
                raise ValueError(
                    f"{path.name}: YEAR column missing and could not infer year. "
                    f"Include a year in the filename (e.g., ...2024...) or pass exactly one --year."
                )
            df["YEAR"] = assume_year
    else:
        raise ValueError(
            f"{path.name}: Columns must match either "
            f"{sorted(std_required)} or {sorted(alt_required)}; got {sorted(cols)}"
        )

    # Normalize string columns
    for c in ["ORIGIN", "DEST", "CARRIER"]:
        df[c] = df[c].astype(str).str.upper().str.strip()

    # Coerce numerics
    for c in ["YEAR", "MONTH", "PASSENGERS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["YEAR", "MONTH", "ORIGIN", "DEST", "CARRIER", "PASSENGERS"]]

def read_t100_csvs(paths: Iterable[Path], year_hints: Optional[List[int]]) -> pd.DataFrame:
    """
    Read one or more T-100 Domestic Segment CSV files and normalize columns.
    If a file lacks YEAR, this tries (in order):
      1) a single value in --year (if provided),
      2) infer year from the filename (e.g., ...2024...).
    """
    single_hint: Optional[int] = None
    if year_hints and len(year_hints) == 1:
        single_hint = year_hints[0]

    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str, low_memory=False)
        df.columns = [c.upper().strip() for c in df.columns]

        # Decide per-file assumed year if needed
        assumed = single_hint
        if "YEAR" not in df.columns and assumed is None:
            inferred = infer_year_from_filename(p)
            assumed = inferred

        df = _normalize_t100_columns(df, assume_year=assumed, path=p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out

def read_simple_routes_csv(path: Path) -> pd.DataFrame:
    """
    Read a simple routes CSV with columns:
    carrier, origin, dest, passengers, year, month  (case-insensitive)
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    needed = ["carrier", "origin", "dest", "passengers"]
    missing = [c for c in needed if c not in cols]
    if missing:
        raise ValueError(f"routes.csv missing columns: {missing}")
    df["CARRIER"] = df[cols["carrier"]].astype(str).str.upper().str.strip()
    df["ORIGIN"] = df[cols["origin"]].astype(str).str.upper().str.strip()
    df["DEST"] = df[cols["dest"]].astype(str).str.upper().str.strip()
    df["PASSENGERS"] = pd.to_numeric(df[cols["passengers"]], errors="coerce")
    df["YEAR"] = pd.to_numeric(df.get(cols.get("year", "YEAR"), pd.NA), errors="coerce")
    df["MONTH"] = pd.to_numeric(df.get(cols.get("month", "MONTH"), pd.NA), errors="coerce")
    return df[["YEAR", "MONTH", "ORIGIN", "DEST", "CARRIER", "PASSENGERS"]]

def aggregate_hvn_routes(df: pd.DataFrame,
                         year_filter: Optional[List[int]] = None,
                         only_carriers: Optional[List[str]] = None,
                         month_filter: Optional[List[int]] = None,
                         direction: str = "both") -> pd.DataFrame:
    """
    Keep only rows where ORIGIN == 'HVN' or DEST == 'HVN'.
    Filters:
      - year_filter: list of years (optional)
      - month_filter: list of months 1..12 (optional)
      - direction:
          'outbound' -> ORIGIN==HVN only (departures)
          'inbound'  -> DEST==HVN only   (arrivals)
          'both'     -> combine inbound+outbound as HVN->DEST totals
    Returns columns: YEAR, CARRIER, ORIGIN='HVN', DEST=<other>, PASSENGERS
    """
    df = df.copy()
    df["ORIGIN"]   = df["ORIGIN"].astype(str).str.upper()
    df["DEST"]     = df["DEST"].astype(str).str.upper()
    df["CARRIER"]  = df["CARRIER"].astype(str).str.upper()
    df["YEAR"]     = pd.to_numeric(df["YEAR"], errors="coerce")
    df["MONTH"]    = pd.to_numeric(df["MONTH"], errors="coerce")
    df["PASSENGERS"] = pd.to_numeric(df["PASSENGERS"], errors="coerce")

    if year_filter:
        df = df[df["YEAR"].isin(year_filter)]
    if month_filter:
        df = df[df["MONTH"].isin(month_filter)]
    if only_carriers:
        only = [c.upper() for c in only_carriers]
        df = df[df["CARRIER"].isin(only)]

    if direction == "outbound":
        df = df[df["ORIGIN"] == "HVN"].dropna(subset=["PASSENGERS"])
        df["PLOT_ORIGIN"] = "HVN"
        df["PLOT_DEST"]   = df["DEST"]

    elif direction == "inbound":
        df = df[df["DEST"] == "HVN"].dropna(subset=["PASSENGERS"])
        df["PLOT_ORIGIN"] = "HVN"
        df["PLOT_DEST"]   = df["ORIGIN"]

    else:  # both
        df = df[(df["ORIGIN"] == "HVN") | (df["DEST"] == "HVN")].dropna(subset=["PASSENGERS"])
        df["PLOT_ORIGIN"] = "HVN"
        df["PLOT_DEST"]   = np.where(df["ORIGIN"] == "HVN", df["DEST"], df["ORIGIN"])

    g = (df.groupby(["YEAR", "CARRIER", "PLOT_ORIGIN", "PLOT_DEST"], dropna=True)["PASSENGERS"]
            .sum()
            .reset_index()
            .rename(columns={"PLOT_ORIGIN": "ORIGIN", "PLOT_DEST": "DEST"}))

    g = g[(g["DEST"].notna()) & (g["DEST"] != "HVN")]
    return g[["YEAR","CARRIER","ORIGIN","DEST","PASSENGERS"]]


# -------------------------
# Great-circle utilities
# -------------------------
def great_circle_points(lat1, lon1, lat2, lon2, n=64):
    phi1, lam1, phi2, lam2 = map(np.radians, [lat1, lon1, lat2, lon2])

    def sph2cart(phi, lam):
        x = np.cos(phi) * np.cos(lam)
        y = np.cos(phi) * np.sin(lam)
        z = np.sin(phi)
        return np.array([x, y, z])

    p1 = sph2cart(phi1, lam1)
    p2 = sph2cart(phi2, lam2)
    dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.isclose(omega, 0):
        return np.linspace([lat1, lon1], [lat2, lon2], n)

    ts = np.linspace(0, 1, n)
    s1 = np.sin((1 - ts) * omega) / np.sin(omega)
    s2 = np.sin(ts * omega) / np.sin(omega)
    pts = (p1[:, None] * s1 + p2[:, None] * s2).T
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    lats = np.degrees(np.arctan2(zs, np.sqrt(xs**2 + ys**2)))
    lons = np.degrees(np.arctan2(ys, xs))
    return np.column_stack([lats, lons])

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_hvn_routes_single_year(routes_year: pd.DataFrame,
                                airports: Dict[str, Tuple[float, float]],
                                title: str,
                                output: Path,
                                focus_extent: Optional[Tuple[float, float, float, float]] = None):
    """
    Render ONE year to a figure with an inset colorbar under the map.
    This avoids extra bottom whitespace and produces a tight image.
    """
    if routes_year.empty:
        raise ValueError("No routes to plot for this year.")

    year_val = int(routes_year["YEAR"].iloc[0])
    vols = routes_year["PASSENGERS"].astype(float).values
    vmin = float(np.nanmin(vols)) if np.isfinite(np.nanmin(vols)) else 0.0
    vmax = float(np.nanmax(vols)) if np.isfinite(np.nanmax(vols)) else 1.0
    if vmin == vmax:
        vmax = max(1.0, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_cmap("viridis")

    # Try cartopy
    use_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
        proj = ccrs.PlateCarree()
        geod = ccrs.Geodetic()
        data_crs = ccrs.PlateCarree()
    except Exception:
        proj = None
        geod = None
        data_crs = None

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(1, 1, 1, projection=proj if use_cartopy else None)

    if use_cartopy:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"))
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0f0f0")
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
        try:
            ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="#777777")
        except Exception:
            pass
        ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3)
        if focus_extent is None:
            ax.set_extent([-125, -60, 18, 50], crs=data_crs)
        else:
            ax.set_extent(list(focus_extent), crs=data_crs)
    else:
        if focus_extent is None:
            ax.set_xlim([-125, -60]); ax.set_ylim([18, 50])
        else:
            xmin, xmax, ymin, ymax = focus_extent
            ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", linewidth=0.3)
        ax.set_aspect("equal", adjustable="box")

    # HVN point
    hvn_lat, hvn_lon = airports["HVN"]
    if use_cartopy:
        ax.plot(hvn_lon, hvn_lat, marker="o", markersize=6, transform=data_crs, color="#444")
    else:
        ax.plot(hvn_lon, hvn_lat, marker="o", markersize=6, color="#444")

    # Routes
    for _, r in routes_year.iterrows():
        dest = str(r["DEST"])
        if dest not in airports:
            continue
        dlat, dlon = airports[dest]
        pts = great_circle_points(hvn_lat, hvn_lon, dlat, dlon, n=80)
        color = cmap(norm(float(r["PASSENGERS"])))
        lw = 0.5 + 4.0 * norm(float(r["PASSENGERS"]))
        if use_cartopy:
            ax.plot(pts[:, 1], pts[:, 0], linewidth=lw, alpha=0.95, transform=geod, color=color)
            ax.plot(dlon, dlat, marker=".", markersize=4, transform=data_crs, color=color)
        else:
            ax.plot(pts[:, 1], pts[:, 0], linewidth=lw, alpha=0.95, color=color)
            ax.plot(dlon, dlat, marker=".", markersize=4, color=color)

    # Year label inside the map
    ax.text(0.01, 0.98, f"{year_val}", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, weight="bold",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    # Colorbar attached to the axis (Cartopy-safe)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax,
        orientation="horizontal",
        pad=0.08,       # space between map and bar
        fraction=0.05,  # bar size relative to axis
        aspect=40       # length/width ratio of the bar
    )
    cbar.set_label("One-way passengers (total in selection)")


    fig.suptitle(title, y=0.96, fontsize=14)
    # Save with tight bbox to clip excess margins
    fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved map to: {output}")

# -------------------------
# Plotting
# -------------------------
def plot_hvn_routes(routes: pd.DataFrame,
                    airports: Dict[str, Tuple[float, float]],
                    title: str,
                    output: Path,
                    focus_extent: Optional[Tuple[float, float, float, float]] = None,
                    split_by_year: bool = False):
    """
    If split_by_year is False (default): one figure with one row per year + shared colorbar row.
    If split_by_year is True: save one figure PER YEAR (suffix _YYYY) using an inset colorbar and tight bbox.
    """
    years = sorted(routes["YEAR"].dropna().astype(int).unique().tolist())
    if not years:
        raise ValueError("No routes to plot after filtering.")

    if split_by_year:
        stem = output.stem
        suffix = output.suffix or ".png"
        for yr in years:
            sub = routes[routes["YEAR"] == yr].copy()
            out_y = output.with_name(f"{stem}_{yr}{suffix}")
            plot_hvn_routes_single_year(sub, airports, f"{title} [Year {yr}]", out_y, focus_extent)
        return

    # -------- original stacked layout (kept for convenience) --------
    vols = routes["PASSENGERS"].astype(float).values
    vmin = float(np.nanmin(vols)) if np.isfinite(np.nanmin(vols)) else 0.0
    vmax = float(np.nanmax(vols)) if np.isfinite(np.nanmax(vols)) else 1.0
    if vmin == vmax:
        vmax = max(1.0, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_cmap("viridis")

    use_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        use_cartopy = True
        proj = ccrs.PlateCarree()
        geod = ccrs.Geodetic()
        data_crs = ccrs.PlateCarree()
    except Exception:
        proj = None
        geod = None
        data_crs = None

    import matplotlib.gridspec as gridspec
    nrows = len(years)
    fig = plt.figure(figsize=(13, 7 * nrows + 2))
    gs  = gridspec.GridSpec(nrows + 1, 1,
                            height_ratios=[1] * nrows + [0.08],
                            hspace=0.25)

    axes = []
    for i, yr in enumerate(years):
        if use_cartopy:
            ax = fig.add_subplot(gs[i, 0], projection=proj)
            ax.add_feature(cfeature.OCEAN.with_scale("50m"))
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0f0f0")
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
            try:
                ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3, edgecolor="#777777")
            except Exception:
                pass
            ax.gridlines(draw_labels=True, linestyle=":", linewidth=0.3)
            if focus_extent is None:
                ax.set_extent([-125, -60, 18, 50], crs=data_crs)
            else:
                ax.set_extent(list(focus_extent), crs=data_crs)
        else:
            ax = fig.add_subplot(gs[i, 0])
            if focus_extent is None:
                ax.set_xlim([-125, -60]); ax.set_ylim([18, 50])
            else:
                xmin, xmax, ymin, ymax = focus_extent
                ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax])
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.grid(True, linestyle=":", linewidth=0.3)
            ax.set_aspect("equal", adjustable="box")
        axes.append(ax)

        sub = routes[routes["YEAR"] == yr].copy()
        hvn_lat, hvn_lon = airports["HVN"]
        if use_cartopy:
            ax.plot(hvn_lon, hvn_lat, marker="o", markersize=6, transform=data_crs, color="#444")
        else:
            ax.plot(hvn_lon, hvn_lat, marker="o", markersize=6, color="#444")

        for _, r in sub.iterrows():
            dest = str(r["DEST"])
            if dest not in airports:
                continue
            dlat, dlon = airports[dest]
            pts = great_circle_points(hvn_lat, hvn_lon, dlat, dlon, n=80)
            color = cmap(norm(float(r["PASSENGERS"])))
            lw = 0.5 + 4.0 * norm(float(r["PASSENGERS"]))
            if use_cartopy:
                ax.plot(pts[:, 1], pts[:, 0], linewidth=lw, alpha=0.95, transform=geod, color=color)
                ax.plot(dlon, dlat, marker=".", markersize=4, transform=data_crs, color=color)
            else:
                ax.plot(pts[:, 1], pts[:, 0], linewidth=lw, alpha=0.95, color=color)
                ax.plot(dlon, dlat, marker=".", markersize=4, color=color)

        ax.text(0.01, 0.98, f"{yr}", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_subplot(gs[-1, 0])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("One-way passengers (total in selection)")

    fig.suptitle(title, y=0.98, fontsize=14)
    fig.subplots_adjust(top=0.92, bottom=0.18)
    fig.savefig(output, dpi=200)
    print(f"Saved map to: {output}")


# -------------------------
# Summaries
# -------------------------
def summarize_routes(routes: pd.DataFrame, save_csv: Optional[Path] = None, top_n: Optional[int] = None) -> None:
    table = (routes
             .rename(columns={"YEAR":"year","CARRIER":"carrier","ORIGIN":"origin","DEST":"dest","PASSENGERS":"passengers"})
             .sort_values(["year","passengers"], ascending=[True, False])
             .reset_index(drop=True))
    counts = table.groupby("year")["dest"].nunique().reset_index(name="num_routes")
    print("\n=== HVN routes: count by year ===")
    print(counts.to_string(index=False))
    print("\n=== HVN routes: ordered by passengers (one-way, annual) ===")
    for yr, g in table.groupby("year", sort=True):
        print(f"\nYear {yr} — {g.shape[0]} routes")
        show = g.head(top_n) if top_n else g
        if top_n:
            print(f"(showing top {top_n}; total {g.shape[0]})")
        print(show.loc[:, ["carrier","dest","passengers"]].to_string(index=False))
    if save_csv is not None:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_csv, index=False)
        print(f"\nSaved route summary CSV → {save_csv}")

# -------------------------
# KML Export
# -------------------------
def _rgba_to_kml_color(rgba, alpha_override: Optional[int] = None) -> str:
    r, g, b, a = rgba
    if alpha_override is not None:
        a = alpha_override / 255.0
    R = int(round(r * 255)); G = int(round(g * 255)); B = int(round(b * 255)); A = int(round(a * 255))
    return f"{A:02x}{B:02x}{G:02x}{R:02x}"

def export_routes_to_kml(routes: pd.DataFrame,
                         airports: Dict[str, Tuple[float, float]],
                         outfile: Path,
                         cmap_name: str = "viridis"):
    cmap = get_cmap(cmap_name)
    vols = routes["PASSENGERS"].astype(float).values
    vmin, vmax = float(np.nanmin(vols)), float(np.nanmax(vols))
    norm = Normalize(vmin=vmin if np.isfinite(vmin) else 0.0,
                     vmax=vmax if np.isfinite(vmax) and vmax != vmin else max(1.0, float(vmax)))
    hvn_lat, hvn_lon = airports["HVN"]

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("<Document>")
    lines.append("<name>HVN Routes</name>")

    for _, row in routes.iterrows():
        dest = str(row["DEST"])
        if dest not in airports:
            continue
        dlat, dlon = airports[dest]
        vol = float(row["PASSENGERS"])
        rgba = cmap(norm(vol))
        kml_color = _rgba_to_kml_color(rgba, alpha_override=220)
        style_id = f"line_{dest}_{int(vol)}"

        lines.append(f'<Style id="{style_id}">')
        lines.append("  <LineStyle>")
        lines.append(f"    <color>{kml_color}</color>")
        width = 1 + int(round(5 * norm(vol)))
        lines.append(f"    <width>{width}</width>")
        lines.append("  </LineStyle>")
        lines.append("</Style>")

        lines.append("<Placemark>")
        lines.append(f"  <name>HVN → {dest} ({int(vol):,})</name>")
        lines.append(f"  <styleUrl>#{style_id}</styleUrl>")
        lines.append("  <LineString>")
        lines.append("    <tessellate>1</tessellate>")
        lines.append("    <coordinates>")
        pts = great_circle_points(hvn_lat, hvn_lon, dlat, dlon, n=80)
        for lat, lon in pts:
            lines.append(f"      {lon:.6f},{lat:.6f},0")
        lines.append("    </coordinates>")
        lines.append("  </LineString>")
        lines.append("</Placemark>")

    lines.append("</Document>")
    lines.append("</kml>")
    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved KML to: {outfile}")

_MONTH_NAME = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# -------------------------
# Main
# -------------------------
def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Build an HVN route volume map (Avelo/Breeze) from T-100 or a simple CSV.")
    p.add_argument("--input-type", choices=["t100", "simple"], required=True,
                   help="t100 = BTS T-100 Domestic Segment CSV(s); simple = your curated routes.csv")
    p.add_argument("--csv-path", nargs="+", type=str,
                   help="One or more T-100 CSV file paths (for input-type=t100)")
    p.add_argument("--routes-csv", type=str, help="Path to simple routes CSV (for input-type=simple)")
    p.add_argument("--year", nargs="+", type=int, help="Year(s) to include, e.g., 2024 2025")
    p.add_argument("--only-carriers", nargs="*", default=["XP", "MX"],
                   help="Filter to these IATA 2-letter carrier codes (default: XP MX)")
    p.add_argument("--output", type=str, default="hvn_routes.png", help="Output image path")
    p.add_argument("--kml-output", type=str, default=None, help="Optional path to write a colored KML of routes")
    p.add_argument("--summary", action="store_true",
                   help="Print per-year route counts and ordered tables to stdout")
    p.add_argument("--summary-csv", type=str, default=None,
                   help="Optional path to save the full ordered routes table as CSV")
    p.add_argument("--top-n", type=int, default=None,
                   help="If set, only show the top N rows per year in the console summary")
    p.add_argument("--split-by-year", action="store_true",
               help="Save a separate PNG per year (e.g., *_2024.png, *_2025.png). Recommended for month/direction slices.")
    p.add_argument("--direction", choices=["both", "outbound", "inbound"], default="both",
               help="Aggregate direction: outbound=HVN departures only, inbound=arrivals to HVN only, both=combined (default)")
    p.add_argument("--month", nargs="+", type=int, default=None,
               help="Filter to specific months by number (1=Jan ... 12=Dec). Can pass multiple.")
    p.add_argument("--academic-scenarios", action="store_true",
               help="Run preset scenarios: Aug inbound, Dec outbound, Jan inbound, May outbound (creates separate files).")

    args = p.parse_args(argv)

    airports = load_airports_latlon()

    if args.input_type == "t100":
        if not args.csv_path:
            print("ERROR: Provide at least one --csv-path for input-type=t100", file=sys.stderr)
            sys.exit(2)
        paths = [Path(x) for x in args.csv_path]
        df = read_t100_csvs(paths, year_hints=args.year)
    else:
        if not args.routes_csv:
            print("ERROR: Provide --routes-csv for input-type=simple", file=sys.stderr)
            sys.exit(2)
        df = read_simple_routes_csv(Path(args.routes_csv))

    if args.academic_scenarios:
        # (month, direction, label)
        scenarios = [
            ([8],  "inbound",  "aug_inbound"),
            ([12], "outbound", "dec_outbound"),
            ([1],  "inbound",  "jan_inbound"),
            ([5],  "outbound", "may_outbound"),
        ]
        for months, direction, tag in scenarios:
            routes = aggregate_hvn_routes(
                df,
                year_filter=args.year,
                only_carriers=args.only_carriers,
                month_filter=months,
                direction=direction
            )
            if routes.empty:
                print(f"[{tag}] No HVN routes after filtering.")
                continue

            mnames = ",".join(_MONTH_NAME.get(m, str(m)) for m in months)
            ttl = f"Tweed–New Haven (HVN) route volumes — Avelo (XP) & Breeze (MX) [{direction}] [Months: {mnames}]"

            out_png = Path(args.output).with_name(Path(args.output).stem + f"_{tag}.png")
            plot_hvn_routes(routes, airports, ttl, out_png, split_by_year=True)

            if args.kml_output:
                out_kml = Path(args.kml_output).with_name(Path(args.kml_output).stem + f"_{tag}.kml")
                export_routes_to_kml(routes, airports, out_kml)

            if args.summary or args.summary_csv:
                csv_path = Path(args.summary_csv).with_name(Path(args.summary_csv).stem + f"_{tag}.csv") if args.summary_csv else None
                summarize_routes(routes, save_csv=csv_path, top_n=args.top_n)

        return  # don't fall through to the single-run path


    routes = aggregate_hvn_routes(
        df,
        year_filter=args.year,
        only_carriers=args.only_carriers,
        month_filter=args.month,
        direction=args.direction
    )

    if routes.empty:
        print("No HVN routes found after filtering. Check your input files and parameters.", file=sys.stderr)
        sys.exit(1)

    title_bits = ["Tweed–New Haven (HVN) route volumes — Avelo (XP) & Breeze (MX)"]
    if args.direction != "both":
        title_bits.append(f"[{args.direction}]")
    if args.month:
        mnames = ",".join(_MONTH_NAME.get(m, str(m)) for m in sorted(set(args.month)))
        title_bits.append(f"[Months: {mnames}]")
    title = " ".join(title_bits)

    output = Path(args.output)
    plot_hvn_routes(routes, airports, title, output, split_by_year=args.split_by_year)

    if args.kml_output:
        export_routes_to_kml(routes, airports, Path(args.kml_output))

    if args.summary or args.summary_csv:
        save_path = Path(args.summary_csv) if args.summary_csv else None
        summarize_routes(routes, save_csv=save_path, top_n=args.top_n)

if __name__ == "__main__":
    main()

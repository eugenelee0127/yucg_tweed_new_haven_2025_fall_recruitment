HVN Route Volume Map — Quick Start

Files created:
- hvn_route_map_template.py  -> main script

What you need:
- At least one CSV of BTS T-100 Domestic Segment data that includes:
  YEAR, MONTH, ORIGIN, DEST, CARRIER, PASSENGERS
  (You can export these from BTS TranStats for the years you want.)

OR

- A simple "routes.csv" with the columns:
  carrier, origin, dest, passengers, year, month

Example 1 — T-100 input
-----------------------
python hvn_route_map_template.py \
  --input-type t100 \
  --csv-path T_T100D_SEGMENT_2024.csv T_T100D_SEGMENT_2025_YTD.csv \
  --year 2024 2025 \
  --only-carriers XP MX \
  --output hvn_routes_2024_2025.png

Example 2 — Simple input
------------------------
python hvn_route_map_template.py \
  --input-type simple \
  --routes-csv routes.csv \
  --year 2025 \
  --only-carriers XP MX \
  --output hvn_routes_2025.png

Notes
-----
- The script tries to use the `airportsdata` package to look up airport lat/lon.
  If it's not installed, it falls back to a built-in dictionary that covers HVN and
  current destinations listed on Tweed’s site (as of Sep 2025).

- If `cartopy` is installed, the map will include coastlines and labeled grids.
  Without `cartopy`, the script will plot lat/lon arcs on simple axes.

- Volumes are plotted as one-way passenger totals per route and per year.
  Line color and width both scale with volume.

- Carriers default to Avelo (XP) and Breeze (MX). Pass `--only-carriers` to change.

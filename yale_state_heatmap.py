import pandas as pd
import plotly.express as px
from pathlib import Path

# --- config ---
CSV_PATH = "yale_oir_2024_sorted.csv"     # file
COLOR_BY = "2024"                          # or "percent_2024"
TITLE = "Where Yale Students Come From (First-Year Origins, 2024)"

# USPS code map (includes DC + PR). Keys as they appear in OIR CSV.
STATE_TO_FIPS = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    "Puerto Rico":"PR"  # if present
}

# --- load & clean ---
df = pd.read_csv(CSV_PATH)

# normalize column names just in case
df.columns = [c.strip() for c in df.columns]
if "State" not in df.columns:
    raise ValueError("Expected a 'State' column in the Yale OIR file.")
if COLOR_BY not in df.columns:
    raise ValueError(f"Expected a '{COLOR_BY}' column in the Yale OIR file.")

# Trim and standardize state names
df["State"] = df["State"].astype(str).str.strip()

# Drop rows we cannot map (e.g., "International")
df["code"] = df["State"].map(STATE_TO_FIPS)
unmapped = df[df["code"].isna()]["State"].unique().tolist()
df = df.dropna(subset=["code"])

# If prefer percentage but it's missing, compute it
if COLOR_BY == "percent_2024" and "percent_2024" not in df.columns:
    total = df["2024"].sum()
    df["percent_2024"] = df["2024"] / total * 100.0

# --- plot ---
fig = px.choropleth(
    df,
    locations="code",
    locationmode="USA-states",
    color=COLOR_BY,
    color_continuous_scale="Viridis",
    scope="usa",
    labels={"2024": "Students", "percent_2024": "% of class"},
    hover_name="State",
    hover_data={"code": False}
)

fig.update_layout(
    title=TITLE,
    margin=dict(l=10, r=10, t=60, b=10),
    coloraxis_colorbar=dict(title="Students" if COLOR_BY=="2024" else "% of class")
)

# --- save outputs ---
out_html = Path("yale_students_2024_heatmap.html")
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"Saved interactive map → {out_html.resolve()}")

# Optional static PNG (needs kaleido)
try:
    out_png = Path("yale_students_2024_heatmap.png")
    fig.write_image(out_png, scale=2, width=1200, height=800)
    print(f"Saved static PNG → {out_png.resolve()}")
except Exception as e:
    print("(PNG export skipped — install 'kaleido' if you want a static image)")

# Tells which rows were excluded (e.g., International)
if unmapped:
    print("Unmapped rows (excluded from the map):", ", ".join(unmapped))

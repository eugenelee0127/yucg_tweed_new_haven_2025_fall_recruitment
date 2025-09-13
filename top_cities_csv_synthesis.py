import pandas as pd

# --------------------
# Airport → (City, State) mapping
# --------------------
airport_map = {
    "MCO": ("Orlando", "Florida"),
    "FLL": ("Fort Lauderdale", "Florida"),
    "PBI": ("West Palm Beach", "Florida"),
    "RSW": ("Fort Myers", "Florida"),
    "RDU": ("Raleigh/Durham", "North Carolina"),
    "TPA": ("Tampa", "Florida"),
    "MYR": ("Myrtle Beach", "South Carolina"),
    "SRQ": ("Sarasota", "Florida"),
    "ATL": ("Atlanta", "Georgia"),
    "USA": ("Concord/Charlotte", "North Carolina"),
    "ILM": ("Wilmington", "North Carolina"),
    "SJU": ("San Juan", "Puerto Rico"),
    "BNA": ("Nashville", "Tennessee"),
    "DAB": ("Daytona Beach", "Florida"),
    "CHS": ("Charleston", "South Carolina"),
    "GSP": ("Greenville/Spartanburg", "South Carolina"),
    "LAL": ("Lakeland", "Florida"),
    "BWI": ("Baltimore", "Maryland"),
    "IAD": ("Washington Dulles", "Virginia"),
    "SAV": ("Savannah", "Georgia"),
    "TYS": ("Knoxville", "Tennessee"),
    "HOU": ("Houston Hobby", "Texas"),
    "MDW": ("Chicago Midway", "Illinois"),
    "MSY": ("New Orleans", "Louisiana"),
    "VRB": ("Vero Beach", "Florida"),
    "STL": ("St. Louis", "Missouri"),
    "VPS": ("Destin/Fort Walton Beach", "Florida"),
    "ORD": ("Chicago O'Hare", "Illinois"),
    "JAX": ("Jacksonville", "Florida"),
    "DFW": ("Dallas/Fort Worth", "Texas"),
    "DTW": ("Detroit", "Michigan"),
    "ORF": ("Norfolk", "Virginia"),
    "TVC": ("Traverse City", "Michigan"),
    "RIC": ("Richmond", "Virginia"),
    "MLB": ("Melbourne", "Florida"),
    "CKB": ("Clarksburg", "West Virginia"),
    "PWM": ("Portland", "Maine"),
    "BDL": ("Hartford", "Connecticut"),
    "BED": ("Bedford", "Massachusetts"),
    "CMH": ("Columbus", "Ohio"),
    "HPN": ("White Plains", "New York"),
    "ILG": ("Wilmington", "Delaware"),
    "PIT": ("Pittsburgh", "Pennsylvania"),
    "PSM": ("Portsmouth", "New Hampshire"),
    "PVD": ("Providence", "Rhode Island"),
    "ROA": ("Roanoke", "Virginia"),
    "SYR": ("Syracuse", "New York"),
}

# --------------------
# Load scenario CSVs
# --------------------
files = {
    "aug_inbound": "hvn_routes_summary_aug_inbound.csv",
    "dec_outbound": "hvn_routes_summary_dec_outbound.csv",
    "jan_inbound": "hvn_routes_summary_jan_inbound.csv",
    "may_outbound": "hvn_routes_summary_may_outbound.csv"
}

dfs = []
for label, path in files.items():
    df = pd.read_csv(path)
    df["scenario"] = label
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# --------------------
# Aggregate across scenarios
# --------------------
agg = (
    all_data.groupby("dest")
    .agg(total_passengers=("passengers", "sum"),
         appearances=("scenario", "nunique"))
    .reset_index()
    .sort_values(["appearances", "total_passengers"], ascending=[False, False])
    .reset_index(drop=True)
)

# --------------------
# Add city and state columns
# --------------------
agg["city"] = agg["dest"].map(lambda x: airport_map.get(x, ("Unknown", "Unknown"))[0])
agg["state"] = agg["dest"].map(lambda x: airport_map.get(x, ("Unknown", "Unknown"))[1])

# Save result
agg.to_csv("hvn_routes_summary_overall_with_city_state.csv", index=False)
print(agg.head(20))
